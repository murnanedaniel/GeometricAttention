import os
import warnings
import time
import itertools
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch import nn

warnings.simplefilter(action='ignore', category=FutureWarning)

def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=True,
    dropout=0.0,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers with dropout
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)



def open_processed_files(input_dir, num_jets):
    
    jet_files = os.listdir(input_dir)
    num_files = (num_jets // 100000) + 1 if num_jets > 0 else 0
    jet_paths = [os.path.join(input_dir, file) for file in jet_files][:num_files]
    opened_files = [torch.load(file) for file in tqdm(jet_paths)]
    
    opened_files = list(itertools.chain.from_iterable(opened_files))
    
    return opened_files

def load_processed_datasets(input_dir,  data_split, graph_construction):
    
    print("Loading torch files")
    print(time.ctime())
    train_jets = open_processed_files(os.path.join(input_dir, "train"), data_split[0])
    val_jets = open_processed_files(os.path.join(input_dir, "val"), data_split[1])
    test_jets = open_processed_files(os.path.join(input_dir, "test"), data_split[2])
    
    print("Building events")
    print(time.ctime())
    train_dataset = build_processed_dataset(train_jets, graph_construction,  data_split[0])
    val_dataset = build_processed_dataset(val_jets, graph_construction, data_split[1])
    test_dataset = build_processed_dataset(test_jets, graph_construction, data_split[2])
    
    return train_dataset, val_dataset, test_dataset

def build_processed_dataset(jetlist, graph_construction, num_jets = None):
    
    subsample = jetlist[:num_jets] if num_jets is not None else jetlist

    try:
        _ = subsample[0].px
    except Exception:
        for i, data in enumerate(subsample):
            subsample[i] = Data.from_dict(data.__dict__)

    if (graph_construction == "fully_connected"):        
        for jet in subsample:
            jet.edge_index = get_fully_connected_edges(jet.x)

    print("Testing sample quality")
    for sample in tqdm(subsample):
        sample.x = sample.px

        # Check if any nan values in sample
        for key in sample.keys:
            assert not torch.isnan(sample[key]).any(), "Nan value found in sample"
            
    return subsample

"""
Returns an array of edge links corresponding to a fully-connected graph - NEW VERSION
"""
def get_fully_connected_edges(x):
    
    n_nodes = len(x)
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    
    return torch.cat([edges, edges.flip(0)], axis=1)