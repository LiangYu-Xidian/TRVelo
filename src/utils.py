import random
import os
import numpy as np
import torch
import yaml

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"Seed set to: {seed}")

def load_config(path='configs/default.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_cluster_edges(dataset_name):
    # Hardcoded edges from Notebook logic
    if dataset_name == 'endocrinogenesis_day15':
        return [('Ductal','Ngn3 low EP'), ('Ngn3 low EP', 'Ngn3 high EP'), 
                ('Ngn3 high EP', 'Pre-endocrine'), ('Pre-endocrine', 'Alpha'), 
                ('Pre-endocrine', 'Beta')]
    # Add others as needed
    return []