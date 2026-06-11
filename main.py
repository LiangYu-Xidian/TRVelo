import argparse
import os
import torch
import scanpy as sc
import scvelo as scv
from torch.utils.data import DataLoader as TorchDataLoader
from src.utils import seed_everything, load_config, get_cluster_edges
from src.dataset import TRVeloDataset
from src.model import VelocityGAT, LatentTime
from src.trainer import Trainer
from src.metrics import compute_cbdir_old, compute_icvcoh

def main(args):
    # 1. Config & Setup
    config = load_config()
    if args.dataset: config['dataset_name'] = args.dataset
    if args.seed: config['seed'] = int(args.seed)

    seed_everything(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 2. Data
    data_obj = TRVeloDataset(config, config['dataset_name'], device)
    adata = data_obj.load_data()
    tfs, edges = data_obj.process_tfs()
    adj_matrix = data_obj.construct_graph()
    graph, x_high = data_obj.get_tensors(adj_matrix)
    x_high_raw = x_high.clone()

    # 3. Model
    vel_model = VelocityGAT(graph.x.shape[1], len(tfs), heads=config['heads'], num_layers=config['vel_num_layers']).to(device)
    time_model = LatentTime(x_high.shape[0], x_high.shape[1], num_layers=config['time_num_layers']).to(device)

    vel_total_param = sum([param.nelement() for param in vel_model.parameters()])
    time_total_param = sum([param.nelement() for param in time_model.parameters()])
    print(graph.x.shape[1])
    print("vel_total_param: "+str(vel_total_param/1e6) + "M")
    print("time_total_param: "+str(time_total_param/1e6) + "M")

    latent_loader = TorchDataLoader(x_high, batch_size=config['batch_size'], shuffle=False)
    trainer = Trainer(vel_model, time_model, graph, x_high, x_high_raw, config, data_obj)

    if args.load_model:
        # ---- Inference-only path ----
        model_dir = args.load_model
        ds = config['dataset_name']
        vel_path = f"{model_dir}/{ds}_velocity_model_weights.pth"
        time_path = f"{model_dir}/{ds}_latent_time_model_weights.pth"
        print(f"--- Loading pre-trained models ---")
        print(f"  vel: {vel_path}")
        print(f"  time: {time_path}")

        # Load with version detection + key remapping
        vel_ckpt = torch.load(vel_path, map_location=device)
        # Auto-detect PyG version mismatch and remap
        all_keys = ''.join(vel_ckpt.keys())
        ckpt_v1 = 'lin_l.' in all_keys   # PyG v1 uses lin_l/lin_r
        ckpt_v2 = 'att_src' in all_keys  # PyG v2 uses att_src
        model_v1 = hasattr(vel_model.input_layer, 'att_l')
        model_v2 = hasattr(vel_model.input_layer, 'att_src')
        print(f"  ckpt=PyGv{'1' if ckpt_v1 else '2'} model=PyGv{'1' if model_v1 else '2'}")

        if ckpt_v1 and model_v2:
            # Remap v1→v2: att_l/r→att_src/dst, average lin_l+lin_r→lin
            remapped = {}
            lin_buf = {}
            for k, v in vel_ckpt.items():
                if 'lin_l.' in k:
                    lin_buf[k.replace('lin_l.', 'lin.')] = v
                elif 'lin_r.' in k:
                    b = k.replace('lin_r.', 'lin.')
                    lin_buf[b] = (lin_buf[b] + v) / 2 if b in lin_buf else v
                elif 'att_l' in k:
                    remapped[k.replace('att_l', 'att_src')] = v
                elif 'att_r' in k:
                    remapped[k.replace('att_r', 'att_dst')] = v
                else:
                    remapped[k] = v
            remapped.update(lin_buf)
        elif ckpt_v2 and model_v1:
            # Remap v2→v1
            remapped = {}
            for k, v in vel_ckpt.items():
                if 'lin.' in k and 'lin_l' not in k and 'lin_r' not in k:
                    remapped[k.replace('lin.', 'lin_l.')] = v
                    remapped[k.replace('lin.', 'lin_r.')] = v
                elif 'att_src' in k:
                    remapped[k.replace('att_src', 'att_l')] = v
                elif 'att_dst' in k:
                    remapped[k.replace('att_dst', 'att_r')] = v
                else:
                    remapped[k] = v
        else:
            remapped = vel_ckpt

        vel_model.load_state_dict(remapped)
        vel_model.load_state_dict(remapped)
        time_model.load_state_dict(torch.load(time_path, map_location=device))
        v0, v1, v2, v3, latent_time, sine_target = trainer.inference(latent_loader)

    else:
        # ---- Training path ----
        trainer.train_phase(latent_loader, mode='pretrain', phase_name='Pretrain')
        trainer.train_phase(latent_loader, mode='train_phase_1', phase_name='Train_I')
        trainer.train_phase(latent_loader, mode='train_phase_2', phase_name='Train_II')
        v0, v1, v2, v3, latent_time, sine_target = trainer.train_phase(latent_loader, mode='train_phase_3', phase_name='Train_III')

    # 5. Result Injection & Evaluation
    print("--- Post-Processing & Evaluation ---")

    lt_np = latent_time.detach().cpu().numpy()
    adata.obs['latent_time'] = (lt_np - lt_np.min()) / (lt_np.max() - lt_np.min())
    adata.layers['velocity'] = v1.detach().cpu().numpy()
    scv.tl.velocity_graph(adata)

    # Save
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    if not args.load_model:
        torch.save(vel_model.state_dict(), f"{config['model_dir']}/{config['dataset_name']}_vel.pth")
    scv.pl.velocity_embedding_stream(adata, color='clusters', save=f"./output/{config['dataset_name']}_velocity.png")

    return v0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='human_cd34_bone_marrow')
    parser.add_argument('--seed', type=str, default='59510')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to checkpoint .pth for inference-only mode')
    args = parser.parse_args()
    main(args)