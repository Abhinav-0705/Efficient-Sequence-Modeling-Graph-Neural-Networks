import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool


class RootOnlyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch, ptr=None):
        if ptr is not None:
            root_indices = ptr[:-1]
        else:
            diff = torch.diff(batch)
            root_indices = torch.cat([torch.tensor([0], device=batch.device), torch.where(diff != 0)[0] + 1])
        root_x = x[root_indices]
        return self.mlp(root_x)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, ptr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_p, max_p], dim=-1)
        return self.classifier(graph_repr)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, ptr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_p, max_p], dim=-1)
        return self.classifier(graph_repr)


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        head_dim = hidden_dim // heads
        self.conv1 = GATConv(in_dim, head_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, head_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, head_dim, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, ptr=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv3(x, edge_index))
        
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_p, max_p], dim=-1)
        return self.classifier(graph_repr)


class BiGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.td_conv1 = GCNConv(in_dim, hidden_dim)
        self.td_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.bu_conv1 = GCNConv(in_dim, hidden_dim)
        self.bu_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, ptr=None):
        rev_edge_index = edge_index[[1, 0]]
        
        td_x = F.relu(self.td_conv1(x, edge_index))
        td_x = self.dropout(td_x)
        td_x = F.relu(self.td_conv2(td_x, edge_index))
        
        bu_x = F.relu(self.bu_conv1(x, rev_edge_index))
        bu_x = self.dropout(bu_x)
        bu_x = F.relu(self.bu_conv2(bu_x, rev_edge_index))
        
        td_mean = global_mean_pool(td_x, batch)
        td_max = global_max_pool(td_x, batch)
        bu_mean = global_mean_pool(bu_x, batch)
        bu_max = global_max_pool(bu_x, batch)
        
        graph_repr = torch.cat([td_mean, td_max, bu_mean, bu_max], dim=-1)
        return self.classifier(graph_repr)


def build_gnn_model(model_name: str, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.5):
    name = model_name.lower().replace("-", "").replace("_", "")
    if name in ["rootonlymlp", "rootmlp", "mlp"]:
        return RootOnlyMLP(in_dim, hidden_dim, num_classes, dropout)
    elif name == "gcn":
        return GCN(in_dim, hidden_dim, num_classes, dropout)
    elif name in ["graphsage", "sage"]:
        return GraphSAGE(in_dim, hidden_dim, num_classes, dropout)
    elif name == "gat":
        return GAT(in_dim, hidden_dim, num_classes, heads=4, dropout=dropout)
    elif name in ["bigcn", "bidirectional_gcn"]:
        return BiGCN(in_dim, hidden_dim, num_classes, dropout)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

