import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNLayer import GNLayer


class GNBlock(nn.Module):
    def __init__(self, indim, outdim, edgedim=0):
        super().__init__()
        self.conv = GNLayer(indim, outdim, outdim, edgedim)
        self.act = nn.PReLU()
        self.norm = nn.BatchNorm1d(outdim, track_running_stats=False)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.norm(x)
        return x


class GNN(nn.Module):
    def __init__(self, indim, hiddendim, outdim, edgedim=0):
      super().__init__()
      self.indim = indim
      self.hiddendim = hiddendim
      self.outdim = outdim
      self.edgedim = edgedim
      self.conv1 = GNBlock(indim, hiddendim, edgedim)
      self.conv2 = GNBlock(hiddendim, hiddendim, edgedim)
      self.conv3 = GNBlock(hiddendim, hiddendim, edgedim)
      self.conv4 = GNBlock(hiddendim, hiddendim, edgedim)
      self.pool = gnn.global_mean_pool
      self.head = gnn.MLP([hiddendim, hiddendim, outdim], norm=None)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if self.device != torch.device('cuda'):
        print('WARNING: GPU not available. Using CPU instead.')
      self.to(self.device)
      self.optimizer = None
      self.criterion = F.cross_entropy

    def forward(self, data):
      x = data.x
      x = self.conv1(x, data.edge_index, data.edge_attr)
      x = self.conv2(x, data.edge_index, data.edge_attr)
      x = self.conv3(x, data.edge_index, data.edge_attr)
      x = self.conv4(x, data.edge_index, data.edge_attr)
      x = self.pool(x, data.batch)
      x = self.head(x)
      return x

    def train_batch(self, loader):
      self.train()
      losses = []
      accs = []
      for data in loader:
        self.optimizer.zero_grad()
        data = data.to(self.device)
        out = self(data)
        loss = self.criterion(out, data.y)
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())
        acc = (out.argmax(dim=1) == data.y).sum() / data.y.size(0)
        accs.append(acc.item())
      return sum(losses)/len(losses), sum(accs)/len(accs)

    @torch.no_grad()
    def test_batch(self, loader):
      self.eval()
      losses = []
      accs = []
      for data in loader:
        data = data.to(self.device)
        out = self(data)
        loss = self.criterion(out, data.y)
        losses.append(loss.item())
        acc = (out.argmax(dim=1) == data.y).sum() / data.y.size(0)
        accs.append(acc.item())
      return sum(losses)/len(losses), sum(accs)/len(accs)

    @torch.no_grad()
    def predict_batch(self, loader):
      self.eval()
      y_preds = []
      y_trues = []
      for data in loader:
        data = data.to(self.device)
        out = self(data)
        y_preds.append(out.argmax(dim=1))
        y_trues.append(data.y)
      y_preds = torch.cat(y_preds).detach().cpu().numpy()
      y_trues = torch.cat(y_trues).detach().cpu().numpy()
      return y_preds, y_trues
