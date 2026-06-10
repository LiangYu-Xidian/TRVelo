import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import Node2Vec
from scipy import sparse
from sklearn.decomposition import PCA

class TRVeloDataset:
    def __init__(self, config, dataset_name, device):
        self.config = config
        self.dataset_name = dataset_name
        self.device = device
        self.data_path = os.path.join(config['data_dir'], f"{dataset_name}_processed.h5ad")
        
    def load_data(self):
        print(f"--- Loading adata: {self.dataset_name} ---")
        adata = sc.read(self.data_path)
        
        if self.dataset_name == 'human_cd34_bone_marrow':
            adata.uns['embedding_key'] = 'tsne'
            species = 'human'
        elif self.dataset_name == 'endocrinogenesis_day15':
            adata.uns['embedding_key'] = 'umap'
            adata.uns['species'] = 'mouse'
        elif self.dataset_name == 'DentateGyrus':
            adata.uns['embedding_key'] = 'umap'
            sc.tl.umap(adata)
            adata.uns['species'] = 'mouse'
            adata.obs["clusters"] = adata.obs["ClusterName"].values
        elif self.dataset_name == 'postpro':
            adata.uns['embedding_key'] = 'umap'
            species = 'human'
            adata.obs["clusters"] = adata.obs["leiden"].values
            
        if 'neighbors' in adata.uns and 'indices' in adata.uns['neighbors']:
            self.neighbor_indices = adata.uns['neighbors']['indices']
            print(f"--- Neighbors indices loaded. Shape: {self.neighbor_indices.shape} ---")
        else:
            print("[Warning] 'neighbors' not found in adata.uns. Smoothness loss will be disabled.")
        
        self.adata = adata
        self.var_names = [n.lower() for n in adata.var_names]
        return adata

    def process_tfs(self):
        # Logic from Notebook "加载TF" cell
        print("--- Processing TFs ---")
        edges = []
        TFs = []
        
        tf_path = os.path.join(self.config['tf_dir'], "gene_attribute_edges.txt")
        if os.path.exists(tf_path):
            df = pd.read_csv(tf_path, sep="\t", usecols=[0, 3], header=1)
            df.iloc[:, 0] = df.iloc[:, 0].str.lower()
            df.iloc[:, 1] = df.iloc[:, 1].str.lower()
            filtered_df = df[df.iloc[:, 0].isin(self.var_names) & df.iloc[:, 1].isin(self.var_names)]
            edges = list(zip(filtered_df.iloc[:, 1], filtered_df.iloc[:, 0]))
            TFs = filtered_df.iloc[:, 1].unique().tolist()
        
        TFs = list(set(TFs) & set(self.var_names))
        TFs_index_list = [self.var_names.index(tf) for tf in TFs]
        TFs_index_list.sort()
        
        self.TFs = TFs
        self.edges = edges
        self.TFs_index_list = TFs_index_list
        return TFs, edges

    def construct_graph(self):
        # Logic from Notebook "构建调控网络" cell
        print("--- Constructing GRN ---")
        N = len(self.var_names)
        var_to_index = {var: idx for idx, var in enumerate(self.var_names)}
        adj_matrix_np = torch.zeros((N, N), dtype=torch.long)

        for src, dst in self.edges:
            if src in var_to_index and dst in var_to_index:
                adj_matrix_np[var_to_index[src], var_to_index[dst]] = 1
        
        adj_matrix_dense = torch.LongTensor(adj_matrix_np).to(self.device)
        n = adj_matrix_dense.size(0)
        adj_matrix_dense[torch.arange(n), torch.arange(n)] = 1 # Self loops

        indices = adj_matrix_dense.nonzero().t()
        values = adj_matrix_dense[indices[0], indices[1]]
        adj_matrix_sparse = torch.sparse_coo_tensor(indices, values, adj_matrix_dense.size()).coalesce()
        
        return adj_matrix_sparse

    def get_tensors(self, adj_matrix_sparse):
        # Logic from Notebook "加载tensor" cell
        print("--- Generating Tensors ---")
        # MAGIC preprocessing as in notebook — must succeed
        sc.external.pp.magic(self.adata)
            
        adata_x = self.adata.X
        if sparse.issparse(adata_x):
            adata_x = adata_x.todense().A
        
        x_high = torch.FloatTensor(adata_x).to(self.device)
        
        # Node2Vec + Expression Concatenation
        print("--- Running Node2Vec ---")
        node2vec = Node2Vec(adj_matrix_sparse.indices().cpu(), embedding_dim=128, walk_length=20, context_size=10)
        n2v_emb = node2vec.forward().detach().numpy()
        
        combined = np.concatenate([adata_x.T, n2v_emb], axis=1)
        graph_x = torch.tensor(combined, dtype=torch.float).to(self.device)
        
        graph = Data(x=graph_x, edge_index=adj_matrix_sparse.indices().to(self.device))
        graph.edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        
        return graph, x_high