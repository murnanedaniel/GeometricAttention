from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, knn_graph, radius_graph
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..jet_gnn_base import JetGNNBase
from ..utils import make_mlp


class GravConv(nn.Module):
    def __init__(self, hparams, input_size=None, output_size=None):
        super().__init__()
        self.hparams = hparams
        self.feature_dropout = hparams["feature_dropout"] if "feature_dropout" in hparams else 0.0
        self.spatial_dropout = hparams["spatial_dropout"] if "spatial_dropout" in hparams else 0.0
        self.input_size = hparams["hidden"] if input_size is None else input_size
        self.output_size = hparams["hidden"] if output_size is None else output_size
        

        self.feature_network = make_mlp(
                2*(self.input_size + 1),
                [self.output_size] * hparams["nb_node_layer"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.feature_dropout
        )

        self.spatial_network = make_mlp(
                self.input_size + 1,
                [self.input_size] * hparams["nb_node_layer"] + [hparams["emb_dims"]],
                hidden_activation=hparams["hidden_activation"],
                layer_norm=hparams["layernorm"],
                batch_norm=hparams["batchnorm"],
                dropout=self.spatial_dropout
        )

        # This handles the various r, k, and random edge options
        self.setup_neighborhood_configuration()

    def get_neighbors(self, spatial_features):
        
        edge_index = torch.empty([2, 0], dtype=torch.int64, device=spatial_features.device)
 
        if self.use_radius:
            radius_edges = radius_graph(spatial_features, r=self.r, max_num_neighbors=self.hparams["max_knn"], batch=self.batch, loop=self.hparams["self_loop"])
            edge_index = torch.cat([edge_index, radius_edges], dim=1)
        
        if self.use_knn and self.knn > 0:
            k_edges = knn_graph(spatial_features, k=self.knn, batch=self.batch, loop=True)
            edge_index = torch.cat([edge_index, k_edges], dim=1)

        if self.use_rand_k and self.rand_k > 0:
            random_edges = knn_graph(torch.rand(spatial_features.shape[0], 2, device=spatial_features.device), k=self.rand_k, batch=self.batch, loop=True) 
            edge_index = torch.cat([edge_index, random_edges], dim=1)
        
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index

    def get_grav_function(self, d):
        grav_weight = self.grav_weight
        grav_function = - grav_weight * d / self.r**2
        
        return grav_function, grav_weight

    def get_attention_weight(self, spatial_features, hidden_features, edge_index):
        start, end = edge_index
        d = torch.sum((spatial_features[start] - spatial_features[end])**2, dim=-1) 
        grav_function, grav_fact = self.get_grav_function(d)

        return torch.exp(grav_function), grav_fact

    def grav_pooling(self, spatial_features, hidden_features):
        edge_index = self.get_neighbors(spatial_features)
        start, end = edge_index
        d_weight, grav_fact = self.get_attention_weight(spatial_features, hidden_features, edge_index)

        if "norm_hidden" in self.hparams and self.hparams["norm_hidden"]:
            hidden_features = F.normalize(hidden_features, p=1, dim=-1)

        return scatter_add(hidden_features[start] * d_weight.unsqueeze(1), end, dim=0, dim_size=hidden_features.shape[0]), edge_index, grav_fact

    def forward(self, hidden_features, batch, current_epoch):
        self.current_epoch = current_epoch
        self.batch = batch

        hidden_features = torch.cat([hidden_features, hidden_features.mean(dim=1).unsqueeze(-1)], dim=-1)
        spatial_features = self.spatial_network(hidden_features)

        if "norm_embedding" in self.hparams and self.hparams["norm_embedding"]:
            spatial_features = F.normalize(spatial_features, p=2, dim=-1)

        aggregated_hidden, edge_index, grav_fact = self.grav_pooling(spatial_features, hidden_features)
        concatenated_hidden = torch.cat([aggregated_hidden, hidden_features], dim=-1)
        return self.feature_network(concatenated_hidden), edge_index, spatial_features, grav_fact

    def setup_neighborhood_configuration(self):
        self.current_epoch = 0
        self.use_radius = bool("r" in self.hparams and self.hparams["r"])
        # A fix here for the case where there is dropout and a large embedded space, model initially can't find neighbors: Enforce self-loop
        if not self.hparams["knn"] and self.hparams["emb_dims"] > 4 and (self.hparams["feature_dropout"] or self.hparams["spatial_dropout"]):
            self.hparams["self_loop"] = True
        self.use_knn = bool("knn" in self.hparams and self.hparams["knn"])
        self.use_rand_k = bool("rand_k" in self.hparams and self.hparams["rand_k"])

    @property
    def r(self):
        if isinstance(self.hparams["r"], list):
            if len(self.hparams["r"]) == 2:
                return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / self.hparams["max_epochs"] )
            elif len(self.hparams["r"]) == 3:
                if self.current_epoch < self.hparams["max_epochs"]/2:
                    return self.hparams["r"][0] + ( (self.hparams["r"][1] - self.hparams["r"][0]) * self.current_epoch / (self.hparams["max_epochs"]/2) )
                else:
                    return self.hparams["r"][1] + ( (self.hparams["r"][2] - self.hparams["r"][1]) * (self.current_epoch - self.hparams["max_epochs"]/2) / (self.hparams["max_epochs"]/2) )
        elif isinstance(self.hparams["r"], float):
            return self.hparams["r"]
        else:
            return 0.3

    @property
    def knn(self):
        if not isinstance(self.hparams["knn"], list):
            return self.hparams["knn"]
        if len(self.hparams["knn"]) == 2:
            return int( self.hparams["knn"][0] + ( (self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
        elif len(self.hparams["knn"]) == 3:
            return int(self.hparams["knn"][0] + ((self.hparams["knn"][1] - self.hparams["knn"][0]) * self.current_epoch / (self.hparams["max_epochs"] / 2))) if self.current_epoch < self.hparams["max_epochs"] / 2 else int(self.hparams["knn"][1] + ((self.hparams["knn"][2] - self.hparams["knn"][1]) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"] / 2)))
        else:
            raise ValueError("knn must be a list of length 2 or 3")

    @property
    def rand_k(self):        
        if not isinstance(self.hparams["rand_k"], list):
            return self.hparams["rand_k"]
        if len(self.hparams["knn"]) == 2:
            return int( self.hparams["rand_k"][0] + ( (self.hparams["rand_k"][1] - self.hparams["rand_k"][0]) * self.current_epoch / self.hparams["max_epochs"] ) )
        elif len(self.hparams["rand_k"]) == 3:
            return int(self.hparams["rand_k"][0] + ((self.hparams["rand_k"][1] - self.hparams["rand_k"][0]) * self.current_epoch / (self.hparams["max_epochs"] / 2))) if self.current_epoch < self.hparams["max_epochs"] / 2 else int(self.hparams["rand_k"][1] + ((self.hparams["rand_k"][2] - self.hparams["rand_k"][1]) * (self.current_epoch - self.hparams["max_epochs"] / 2) / (self.hparams["max_epochs"] / 2)))
        else:
            raise ValueError("rand_k must be a list of length 2 or 3")

    @property
    def grav_weight(self):        
        if isinstance(self.hparams["grav_weight"], list) and len(self.hparams["grav_weight"]) == 2:
            return (self.hparams["grav_weight"][0] + (self.hparams["grav_weight"][1] - self.hparams["grav_weight"][0]) * self.current_epoch / self.hparams["max_epochs"])
        elif isinstance(self.hparams["grav_weight"], float):
            return self.hparams["grav_weight"]
        else:
            raise ValueError("grav_weight must be a list of length 2 or a float")
        