import torch
import torch.nn as nn
import torch_geometric.nn as gnn

# class GNLayer(nn.Module):
#     def __init__(self, indim, hiddendim, outdim, edgedim=0):
#         super(GNLayer, self).__init__()
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * indim + edgedim, hiddendim),
#             nn.PReLU(),
#             nn.Linear(hiddendim, hiddendim),
#             nn.PReLU(),
#         )
#         self.node_mlp = nn.Sequential(
#             nn.Linear(indim + hiddendim, hiddendim),
#             nn.PReLU(),
#             nn.Linear(hiddendim, outdim),
#         )

#     def segment_sum(self, data, segment_ids, num_segments):
#       result_shape = (num_segments, data.size(1))
#       result = data.new_full(result_shape, 0)
#       segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
#       result.scatter_add_(0, segment_ids, data)
#       return result

#     def edge_model(self, source, target, edge_attr=None):
#         if edge_attr is None:
#             out = torch.cat([source, target], dim=1)
#         else:
#             out = torch.cat([source, target, edge_attr], dim=1)
#         out = self.edge_mlp(out)
#         return out

#     def node_model(self, x, edge_index, edge_attr):
#         row, col = edge_index
#         agg = self.segment_sum(edge_attr, row, num_segments=x.size(0))
#         agg = torch.cat([x, agg], dim=1)
#         out = self.node_mlp(agg)
#         return out

#     def forward(self, x, edge_index, edge_attr):
#         row, col = edge_index
#         edge_attr = self.edge_model(x[row], x[col], edge_attr)
#         x = self.node_model(x, edge_index, edge_attr)
#         return x


class GNLayer(gnn.MessagePassing):
    def __init__(self, indim, hiddendim, outdim, edgedim=0):
        super(GNLayer, self).__init__()
        self.in_channels = indim
        self.hidden_channels = hiddendim
        self.out_channels = outdim
        self.edge_dim = edgedim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * indim + edgedim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, hiddendim),
            nn.PReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(indim + hiddendim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, outdim),
        )

    def reset_parameters(self):
        super().reset_parameters()
        gnn.init.reset(self.edge_mlp)
        gnn.init.reset(self.node_mlp)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is None:
            out = torch.cat([x_i, x_j], dim=1)
        else:
            out = torch.cat([x_i, x_j, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out
