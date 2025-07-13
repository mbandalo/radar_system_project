import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BayesGNN(nn.Module):
    """
    Bayesian Graph Neural Network that outputs Dirichlet parameters for multi-class scene distribution.
    (Note: This class is present but GNNFusion is used in train_scene_module.py)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # MLP to produce Dirichlet concentration parameters (alpha)
        self.fc_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus()  # ensure alpha > 0
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        alpha = self.fc_alpha(x) + 1e-3
        return alpha

class GNNFusion(nn.Module):
    """
    Deterministic GNN for scene classification (logits output).
    This model performs global pooling to produce a single output per graph/scene.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        logits = self.fc(x)
        return logits


class GNNPointSegmenter(nn.Module):
    """
    Graph Neural Network for per-point segmentation, designed to integrate scene-level context.
    Now with flexible number of hidden layers and optional Batch Normalization and Dropout.
    """
    def __init__(self, point_feature_dim: int, scene_context_dim: int, hidden_dim: int, num_point_classes: int, gnn_hidden_layers: list = None, gnn_dropout_rate: float = 0.0): # <--- DODANO: gnn_dropout_rate
        super().__init__()
        
        total_input_dim = point_feature_dim + scene_context_dim
        
        self.conv_layers = nn.ModuleList()
        current_in_dim = total_input_dim

        if gnn_hidden_layers is None or not gnn_hidden_layers:
            gnn_hidden_layers = [hidden_dim, hidden_dim] 

        for layer_idx, h_dim in enumerate(gnn_hidden_layers):
            self.conv_layers.append(GCNConv(current_in_dim, h_dim))
            self.conv_layers.append(nn.BatchNorm1d(h_dim))
            current_in_dim = h_dim
        
        self.dropout = nn.Dropout(gnn_dropout_rate)
        self.output_layer = nn.Linear(current_in_dim, num_point_classes)

    def forward(self, x, edge_index, scene_context=None, batch=None):
        if scene_context is not None:
            if batch is None:
                scene_context_broadcasted = scene_context.repeat(x.size(0), 1)
            else:
                scene_context_broadcasted = scene_context[batch]
            
            x = torch.cat([x, scene_context_broadcasted], dim=-1)

        for layer in self.conv_layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            x = F.relu(x)
            
            # Apply dropout after ReLU for all but the last GCNConv layer (before final linear layer)
            if layer != self.conv_layers[-1]:
                x = self.dropout(x)  

        return self.output_layer(x)

