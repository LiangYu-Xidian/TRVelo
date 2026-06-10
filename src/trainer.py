import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools
import numpy as np
from scipy.stats import spearmanr

class Trainer:
    def __init__(self, velocity_model, latent_time_model, graph, x_high, x_high_raw, config, dataset_obj):
        self.velocity_model = velocity_model
        self.latent_time_model = latent_time_model
        self.graph = graph
        self.x_high = x_high
        self.x_high_raw = x_high_raw
        self.config = config
        self.device = dataset_obj.device

        self.TFs_index_list = dataset_obj.TFs_index_list
        self.TFs = dataset_obj.TFs
        self.var_names = dataset_obj.var_names
        self.edges = dataset_obj.edges
        self.num_cells = x_high.shape[0]
        self.num_genes = x_high.shape[1]

        if dataset_obj.neighbor_indices is not None:
            self.neighbors = torch.LongTensor(dataset_obj.neighbor_indices).to(self.device)
        else:
            self.neighbors = None

        self._init_parameters()

    def _init_parameters(self):
        print("--- Initializing Parameters (Spearman & Steady State) ---")
        max_top_K = int(self.num_cells/500)
        min_top_K = int(self.num_cells/500)
        skip_indices = int(self.num_cells/500)

        x_high_nonzero = torch.where(self.x_high == 0, torch.tensor(float('inf'), dtype=self.x_high_raw.dtype).to(self.device), self.x_high_raw)
        _, max_indices = torch.topk(self.x_high, max_top_K + skip_indices, dim=0)
        _, min_indices = torch.topk(-x_high_nonzero, min_top_K + skip_indices, dim=0)

        self.max_indices = max_indices[skip_indices:]
        self.min_indices = min_indices[skip_indices:]
        self.max_weights = torch.ones(max_top_K, 1, dtype=torch.float32).to(self.device)
        self.min_weights = torch.ones(min_top_K, 1, dtype=torch.float32).to(self.device)

        adata_TFs = self.x_high[:, self.TFs_index_list].cpu().numpy()
        adata_x_np = self.x_high.cpu().numpy()
        combined_matrix = np.hstack((adata_TFs, adata_x_np))
        correlation_matrix, _ = spearmanr(combined_matrix, axis=0)
        correlation_matrix = correlation_matrix[:adata_TFs.shape[1], adata_TFs.shape[1]:]
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0)
        self.init_weight_TFs = torch.FloatTensor(correlation_matrix.T).to(self.device)

        self.init_weight_TFs_mask = torch.zeros_like(self.init_weight_TFs, device=self.device)
        for edge in self.edges:
            tmpTF = edge[0]
            tmpGene = edge[1]
            if tmpTF in self.TFs and tmpGene in self.var_names:
                self.init_weight_TFs_mask[self.var_names.index(tmpGene)][self.TFs.index(tmpTF)] = 1

        self.init_weight_TFs = self.init_weight_TFs * self.init_weight_TFs_mask

        self.init_weight_self = torch.zeros_like(self.x_high, device=self.device)
        combined_indices = torch.cat((self.min_indices, self.max_indices), dim=0)
        J, I = combined_indices.size(0), combined_indices.size(1)
        row_indices = combined_indices.t().reshape(-1)
        col_indices = torch.arange(I, device=self.device).repeat_interleave(J)
        x_vals = self.x_high_raw[row_indices, col_indices]
        nonzero_mask = torch.abs(x_vals) > 1e-3
        row_nz = row_indices[nonzero_mask]
        col_nz = col_indices[nonzero_mask]
        x_nz = x_vals[nonzero_mask]
        x_selected = self.x_high[row_nz][:, self.TFs_index_list]
        w_selected = self.init_weight_TFs[col_nz]
        dot = (x_selected * w_selected).sum(dim=1)
        self.init_weight_self[row_nz, col_nz] = -dot / x_nz

        mask = self.x_high != 0
        max_expression = torch.max(self.x_high, dim=0).values
        masked_x_high = torch.where(mask, self.x_high, torch.tensor(float('inf')).to(self.device))
        min_expression = torch.min(masked_x_high, dim=0).values
        min_expression[min_expression == float('inf')] = float(0)

        self.init_a = (max_expression - min_expression) / 2
        self.init_b = torch.full((self.num_genes,), 1.0).to(self.device)
        self.init_c = torch.full((self.num_genes,), -0.5).to(self.device)
        self.init_d = (max_expression + min_expression) / 2

    def train_phase(self, latent_time_loader, mode='pretrain', phase_name='Phase'):
        epochs = self.config['epochs']
        cfg = self.config
        is_pretrain = (mode == 'pretrain')

        lr = float(cfg['learning_rate']) if is_pretrain else float(cfg['learning_rate_train'])
        optimizer = optim.Adam(
            itertools.chain(self.velocity_model.parameters(), self.latent_time_model.parameters()),
            lr=lr
        )
        print(f"--- Starting {phase_name} (Mode: {mode}) | Epochs: {epochs} | LR: {lr} ---")

        for epoch in range(epochs):
            self.velocity_model.train()
            self.latent_time_model.train()

            # --- Forward Pass ---
            velocity_weight = self.velocity_model(self.graph)
            scale_factor = velocity_weight[:, 0:1].squeeze(dim=1)
            velocity_weight_TFs = velocity_weight[:, 1:-5]
            velocity_weight_self = velocity_weight[:, -5:-4].squeeze(dim=1)
            a = velocity_weight[:, -4:-3].squeeze(dim=1)
            b = velocity_weight[:, -3:-2].squeeze(dim=1)
            c = velocity_weight[:, -2:-1].squeeze(dim=1)
            d = velocity_weight[:, -1]

            latent_time_list = []
            for batch_cell in latent_time_loader:
                batch_latent_time = self.latent_time_model(batch_cell)
                latent_time_list.append(batch_latent_time)
            latent_time = torch.cat(latent_time_list, dim=0)[:, -1]
            latent_time_expanded = latent_time.unsqueeze(1).expand(-1, self.num_genes)

            sine_target = (a * torch.sin(torch.pi*b*(latent_time_expanded+c)) + d)

            # Scale Logic — matches notebook exactly
            if is_pretrain and epoch < 500:
                velocity_weight_TFs = velocity_weight_TFs * torch.ones_like(scale_factor).unsqueeze(1)
            else:
                velocity_weight_TFs = velocity_weight_TFs * scale_factor.unsqueeze(1)

            velocity_weight_TFs = velocity_weight_TFs * self.init_weight_TFs_mask

            # Velocity Equations
            v0 = F.relu(sine_target[:, self.TFs_index_list] @ velocity_weight_TFs.t()) + sine_target * velocity_weight_self
            v1 = torch.autograd.grad(sine_target.sum(), latent_time_expanded, retain_graph=True)[0]
            v2 = F.relu(self.x_high[:, self.TFs_index_list] @ velocity_weight_TFs.t()) + self.x_high * velocity_weight_self
            v3 = F.relu(self.x_high[:, self.TFs_index_list] @ velocity_weight_TFs.t()) + sine_target * velocity_weight_self

            # --- LOSS CALCULATION ---
            losses = []
            loss_start_epoch = 100 if is_pretrain else 0

            if epoch >= loss_start_epoch:
                # Velocity Consistency v3-v1: factor=1e-8 in pretrain, 1e-9 in train
                lv1 = float(cfg['lambda_velocity']) * 10 if is_pretrain else float(cfg['lambda_velocity'])
                loss_v01 = (torch.sum((v3 - v1)**2)/self.num_cells) * lv1
                losses.append(loss_v01)

                # Velocity Consistency v2-v0: factor=1e-9 always
                loss_v20 = (torch.sum((v2 - v0)**2)/self.num_cells) * float(cfg['lambda_velocity'])
                losses.append(loss_v20)

                # Steady State — disabled in train_phase_3 (epoch >= 99990 = never)
                enable_steady = True
                if mode == 'train_phase_3':
                    enable_steady = False

                if enable_steady:
                    ls_v = float(cfg['lambda_steady'])
                    for v in [v0, v3]:
                        max_steady = torch.gather(v, 0, self.max_indices) * self.max_weights
                        min_steady = torch.gather(v, 0, self.min_indices) * self.min_weights
                        losses.append((torch.sum(max_steady**2) / self.num_genes) * ls_v)
                        losses.append((torch.sum(min_steady**2) / self.num_genes) * ls_v)

            # --- Pretrain Init Losses ---
            if is_pretrain:
                f0 = 0.1
                f1 = 0.0

                # TF loss: active epoch < 100
                curr = f0 if epoch < 100 else f1
                init_scaled = self.init_weight_TFs * torch.ones_like(scale_factor).unsqueeze(1)
                losses.append(F.mse_loss(velocity_weight_TFs, init_scaled.detach()) * curr)

                # Self loss: active epoch < 100
                curr = f0 if epoch < 100 else f1
                losses.append(F.mse_loss(velocity_weight_self, self.init_weight_self.detach()) * curr)

                # Param losses: active epoch < 300
                curr = f0 if epoch < 300 else f1
                losses.append(F.mse_loss(a, self.init_a) * curr)
                losses.append(F.mse_loss(b, self.init_b) * curr)
                losses.append(F.mse_loss(c, self.init_c) * curr)
                losses.append(F.mse_loss(d, self.init_d) * curr)

                # Scale loss: active epoch < 500 (but epochs=300 so always active)
                curr = f0 if epoch < 500 else f1
                losses.append(F.mse_loss(scale_factor, torch.ones_like(scale_factor)) * curr)

            # --- Train Phase Losses ---
            else:
                # Scale loss: 1e-1 for I/II, 1e-3 for III
                slf = 0.001 if mode == 'train_phase_3' else 0.1
                losses.append(F.mse_loss(scale_factor, torch.ones_like(scale_factor)) * slf)

                # Smoothness: phase 2 and 3 only, 1 random neighbor
                if self.neighbors is not None and (mode == 'train_phase_2' or mode == 'train_phase_3'):
                    rand_idx = torch.randint(0, 30, (self.neighbors.size(0),)).to(self.device)
                    sel = self.neighbors[torch.arange(self.neighbors.size(0)), rand_idx]
                    losses.append(torch.sum((latent_time - latent_time[sel])**2) * 0.001 * float(cfg['lambda_smooth']))

            # --- Optimization ---
            if losses:
                total_loss = torch.stack(losses).sum()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        return v0, v1, v2, v3, latent_time, sine_target
