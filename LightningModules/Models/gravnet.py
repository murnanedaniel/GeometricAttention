from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, knn_graph, radius_graph
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..jet_gnn_base import JetGNNBase
from ..utils import make_mlp
from .gravconv import GravConv

class GravNet(JetGNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Construct architecture
        # -------------------------

        if "spatial_channels" in hparams and hparams["spatial_channels"] is not None:
            self.spatial_channels = hparams["spatial_channels"]
        else:
            self.spatial_channels = len(self.hparams["feature_set"])

        # Encode input features to hidden features
        self.get_layer_structure()
        self.feature_encoder = make_mlp(
            self.spatial_channels,
            [self.layer_structure[0][0]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Construct the GravNet convolution modules 
        self.grav_convs = nn.ModuleList([
            GravConv(hparams, input_size, output_size) for input_size, output_size in self.layer_structure
        ])

        # Decode hidden features to output features
        self.get_output_structure()
        self.output_network = make_mlp(
            self.aggregation_factor*self.output_size,
            [self.output_size] * hparams["nb_node_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"]
        )

    def output_step(self, x, batch, all_x = None):

        if all_x is not None and self.hparams["concat_all_layers"]:
            all_x = torch.cat(all_x, dim=-1)
        else:
            all_x = x

        if self.hparams["aggregation"] == "mean_sum":
            graph_level_inputs = torch.cat([global_add_pool(all_x, batch), global_mean_pool(all_x, batch), global_max_pool(all_x, batch)], dim=1)
        elif self.hparams["aggregation"] == "sum":
            graph_level_inputs = global_add_pool(all_x, batch)
        elif self.hparams["aggregation"] == "mean":
            graph_level_inputs = global_mean_pool(all_x, batch)

        # Add dropout
        if "final_dropout" in self.hparams and self.hparams["final_dropout"] > 0.0:
            graph_level_inputs = F.dropout(graph_level_inputs, p=self.hparams["final_dropout"], training=self.training)

        return self.output_network(graph_level_inputs)

    def forward(self, batch, log_attention=False):

        x = self.concat_feature_set(batch)

        # Encode all features
        hidden_features = self.feature_encoder(x)

        # If concating, keep list of all output features
        all_hidden_features = []
        for i, grav_conv in enumerate(self.grav_convs):

            hidden_features, spatial_edges, _, grav_fact = checkpoint(grav_conv, hidden_features, batch.batch, self.current_epoch)

            self.log_dict({f"nbhood_sizes/nb_size_{i}": spatial_edges.shape[1] / hidden_features.shape[0],
                            f"grav_facts/fact_{i}": grav_fact}, on_step=False, on_epoch=True)

            if self.hparams["concat_all_layers"]:
                all_hidden_features.append(hidden_features)
            
        return self.output_step(hidden_features, batch.batch, all_hidden_features)

    def get_layer_structure(self):
        """
        Construct a list of [input_size, output_size] for each layer (assuming nodes are already encoded).
        For a flat structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 64], [64, 64], [64, 64]]
        For a pyramid structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 32], [32, 16], [16, 8]]
        For an antipyramid structure, 3 layers, and a hidden size of 64, this would be:
        [[64, 128], [128, 256], [256, 512]]
        """

        if "layer_shape" not in self.hparams or self.hparams["layer_shape"]=="flat":
            self.layer_structure = [[self.hparams["hidden"]] * 2] * self.hparams["n_graph_iters"]
        elif self.hparams["layer_shape"] == "pyramid":
            self.layer_structure = [ [max(self.hparams["hidden"] // 2**i , 2), max(self.hparams["hidden"] // 2**(i+1), 2)] for i in range(self.hparams["n_graph_iters"]) ]
        elif self.hparams["layer_shape"] == "antipyramid":
            self.layer_structure = [ [max(self.hparams["hidden"] // 2**i , 2) , max(self.hparams["hidden"] // 2**(i-1), 2)] for i in range(self.hparams["n_graph_iters"], 0, -1) ]

    def get_output_structure(self):
        """
        Calculate the size of the final encoded layer that needs to be decoded.
        If we don't concat all layers, then it is simply the size of the final layer.
        If we do concat all layers, then it is the sum of all layer output sizes (the second entry in each layer shape pair).
        """

        if "concat_all_layers" in self.hparams and self.hparams["concat_all_layers"]:
            self.output_size = sum(layer[1] for layer in self.layer_structure)
        else:
            self.output_size = self.layer_structure[-1][1]

        if self.hparams["aggregation"] == "mean_sum":
            self.aggregation_factor = 3
        elif self.hparams["aggregation"] in ["mean", "sum"]:
            self.aggregation_factor = 1