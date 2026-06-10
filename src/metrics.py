import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc

def compute_cbdir_old(adata, cell_pos, cell_velocitys, cluster_edges):
    """Port of CBDir_old from Notebook"""
    scores = {}
    print("--- Computing CBDir (Old) ---")
    
    if 'neighbors' not in adata.uns:
        print("Neighbors not found in adata.uns")
        return scores

    for u, v in cluster_edges:
        sel = adata.obs["clusters"] == u
        nbs = adata.uns['neighbors']['indices'][sel]
        
        # This map lambda logic from notebook
        boundary_nodes = [
            nodes[adata.obs["clusters"].iloc[nodes].values == v] 
            for nodes in nbs
        ]

        x_points = cell_pos[sel]
        x_velocities = cell_velocitys[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue
            
            position_dif = cell_pos[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))
            
        if type_score:
            scores[(u, v)] = np.mean(type_score)
            print(f"{u} -> {v}: {scores[(u,v)]}")
            
    cbdir_val = np.mean(list(scores.values())) if scores else 0
    print(f"Global CBDir (Old): {cbdir_val}")
    return scores

def compute_icvcoh(adata, cluster_names):
    """Port of ICVCoh from Notebook"""
    print("--- Computing ICVCoh ---")
    scores = {}
    
    # Ensure velocity layer exists
    if 'velocity' not in adata.layers:
        print("Velocity layer not found.")
        return {}

    for cat in cluster_names:
        sel = adata.obs["clusters"] == cat
        if not sel.any(): continue
        
        nbs = adata.uns['neighbors']['indices'][sel]
        
        same_cat_nodes = [
             nodes[adata.obs["clusters"].iloc[nodes].values == cat]
             for nodes in nbs
        ]
        
        cat_vels = adata.layers['velocity'][sel]
        
        cat_score = []
        for ith, nodes in enumerate(same_cat_nodes):
            if len(nodes) > 0:
                sim = cosine_similarity(cat_vels[[ith]], adata.layers['velocity'][nodes]).mean()
                cat_score.append(sim)
        
        if cat_score:
            scores[cat] = np.mean(cat_score)
            print(f"{cat}: {scores[cat]}")

    mean_score = np.mean(list(scores.values())) if scores else 0
    print(f"Global ICVCoh: {mean_score}")
    return scores