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
    
    seed_everything(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 2. Data
    data_obj = TRVeloDataset(config, config['dataset_name'], device)
    adata = data_obj.load_data()
    tfs, edges = data_obj.process_tfs()
    adj_matrix = data_obj.construct_graph()
    graph, x_high = data_obj.get_tensors(adj_matrix)
    
    # Store Raw X for init logic
    x_high_raw = x_high.clone()

    # 3. Model
    # Input dim is Graph X dim (Expr + Node2Vec)
    vel_model = VelocityGAT(graph.x.shape[1], len(tfs), heads=config['heads'], num_layers=config['vel_num_layers']).to(device)
    time_model = LatentTime(x_high.shape[0], x_high.shape[1], num_layers=config['time_num_layers']).to(device)
    

    vel_total_param = sum([param.nelement() for param in vel_model.parameters()])
    time_total_param = sum([param.nelement() for param in time_model.parameters()])
    print(graph.x.shape[1])
    print("vel_total_param: "+str(vel_total_param/1e6) + "M")
    print("time_total_param: "+str(time_total_param/1e6) + "M")
    # 4. Train
    # Dataloader for cell embeddings
    latent_loader = TorchDataLoader(x_high, batch_size=config['batch_size'], shuffle=False)
    
    trainer = Trainer(vel_model, time_model, graph, x_high, x_high_raw, config, data_obj)
    
    # Pretrain
    trainer.train_phase(latent_loader, mode='pretrain', phase_name='Pretrain')
    
    # Train Phases
    # You can call this multiple times or add logic in trainer to reset optimizers
    trainer.train_phase(latent_loader, mode='train_phase_1', phase_name='Train_I')
    trainer.train_phase(latent_loader, mode='train_phase_2', phase_name='Train_II')
    v0, v1, v2, v3, latent_time, sine_target = trainer.train_phase(latent_loader, mode='train_phase_3', phase_name='Train_III')

    # 5. Result Injection & Evaluation
    print("--- Post-Processing & Evaluation ---")
    
    # Inject results back to AnnData for metrics/plotting
    # Normalized latent time (0-1)
    lt_np = latent_time.detach().cpu().numpy()
    adata.obs['latent_time'] = (lt_np - lt_np.min()) / (lt_np.max() - lt_np.min())
    
    # Inject velocity
    adata.layers['velocity'] = v0.detach().cpu().numpy()
    
    # Calculate Velocity Graph
    scv.tl.velocity_graph(adata)
    
    # Metrics
#     edges = get_cluster_edges(config['dataset_name'])
#     if edges:
#         # Need X_emb and velocity_emb
#         scv.tl.velocity_embedding(adata, basis=adata.uns['embedding_key'])
        
#         compute_cbdir_old(
#             adata, 
#             adata.obsm[f"X_{adata.uns['embedding_key']}"],
#             adata.obsm[f"velocity_{adata.uns['embedding_key']}"],
#             edges
#         )
        
#     compute_icvcoh(adata, list(adata.obs['clusters'].unique()))
    
    # Save
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    torch.save(vel_model.state_dict(), f"{config['model_dir']}/{config['dataset_name']}_vel.pth")
    scv.pl.velocity_embedding_stream(adata, color='clusters', save=f"./output/{config['dataset_name']}_velocity.png")
    adata.layers["velocity"] = v1.detach().cpu().numpy()
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata)
    
    
    return v0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='human_cd34_bone_marrow')
    parser.add_argument('--seed', type=str, default='59510')
    args = parser.parse_args()
    main(args)