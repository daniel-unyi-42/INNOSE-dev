import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN import GNBlock, GNN
from torch_geometric.data import Data

class CriticGNN(GNN):
    def __init__(self, indim, hidden, outdim, edgedim=0):
        super(CriticGNN, self).__init__(indim, hidden, outdim, edgedim)

    def random_mask(self, num_nodes):
        probs = torch.rand((num_nodes, 1), device=self.device)
        mask = (probs > 0.5).detach().float()
        return mask

    def forward(self, data, mask=None):
        if mask is None:
            mask = self.random_mask(data.num_nodes)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = (x * mask).float()
        mask = mask.squeeze().bool()
        edge_mask = (mask[edge_index[0]]) & (mask[edge_index[1]])
        edge_index = edge_index[:, edge_mask]
        if edge_attr is not None:
            edge_attr = edge_attr[edge_mask]
        return super().forward(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch))

# for node level masks
class ActorGNN(nn.Module):
    def __init__(self, critic, baseline, lambda_):
        super(ActorGNN, self).__init__()
        self.indim = critic.indim
        self.hiddendim = critic.hiddendim
        self.edgedim = critic.edgedim
        self.conv1 = GNBlock(self.indim, self.hiddendim, self.edgedim)
        self.conv2 = GNBlock(self.hiddendim, self.hiddendim, self.edgedim)
        self.conv3 = GNBlock(self.hiddendim, self.hiddendim, self.edgedim)
        self.conv4 = GNBlock(self.hiddendim, 1, self.edgedim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != torch.device('cuda'):
            print('WARNING: GPU not available. Using CPU instead.')
        self.to(self.device)
        self.optimizer = None
        self.lambda_ = lambda_
        self.critic = critic
        self.baseline = baseline
    
    def loss(self, reward, y_true, y_pred):
        cross_entropy = -torch.sum(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8), dim=1)
        custom_actor_loss = reward * cross_entropy + self.lambda_ * torch.mean(y_pred, dim=1)
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = self.conv4(x, data.edge_index, data.edge_attr)
        return torch.sigmoid(x)

    def train_batch(self, loader):
        actor_losses = []
        mask_sizes = []
        critic_pos_accs = []
        critic_neg_accs = []
        baseline_accs = []
        for data in loader:
            data = data.to(self.device)
            self.eval()
            with torch.no_grad():
                probs = self(data)
                mask = torch.bernoulli(probs)
                mask_sizes.append(torch.sum(mask).item() / len(mask))
                critic_pos_out = self.critic(data, mask)
                pos_acc = (critic_pos_out.argmax(dim=1) == data.y).float().mean()
                critic_pos_accs.append(pos_acc.item())
                critic_neg_out = self.critic(data, 1.0 - mask)
                neg_acc = (critic_neg_out.argmax(dim=1) == data.y).float().mean()
                critic_neg_accs.append(neg_acc.item())
                baseline_out = self.baseline(data)
                baseline_acc = (baseline_out.argmax(dim=1) == data.y).float().mean()
                baseline_accs.append(baseline_acc.item())
                critic_pos_loss = self.critic.criterion(critic_pos_out, data.y, reduction='none')[data.batch]
                baseline_loss = self.baseline.criterion(baseline_out, data.y, reduction='none')[data.batch]
                reward = -(critic_pos_loss - baseline_loss)
            self.train()
            self.optimizer.zero_grad()
            probs = self(data)
            loss = self.loss(reward, mask, probs)
            loss.backward()
            self.optimizer.step()
            actor_losses.append(loss.item())
        return sum(actor_losses)/len(actor_losses),\
            sum(mask_sizes)/len(mask_sizes),\
                sum(critic_pos_accs)/len(critic_pos_accs),\
                    sum(critic_neg_accs)/len(critic_neg_accs),\
                        sum(baseline_accs)/len(baseline_accs)

    @torch.no_grad()
    def test_batch(self, loader):
        actor_losses = []
        mask_sizes = []
        critic_pos_accs = []
        critic_neg_accs = []
        baseline_accs = []
        self.eval()
        for data in loader:
            data = data.to(self.device)
            probs = self(data)
            mask = (probs > 0.5).float()
            critic_pos_out = self.critic(data, mask)
            pos_acc = (critic_pos_out.argmax(dim=1) == data.y).float().mean()
            critic_pos_accs.append(pos_acc.item())
            critic_neg_out = self.critic(data, 1.0 - mask)
            neg_acc = (critic_neg_out.argmax(dim=1) == data.y).float().mean()
            critic_neg_accs.append(neg_acc.item())
            mask_sizes.append(torch.sum(mask).item() / len(mask))
            baseline_out = self.baseline(data)
            baseline_acc = (baseline_out.argmax(dim=1) == data.y).float().mean()
            baseline_accs.append(baseline_acc.item())
            critic_pos_loss = self.critic.criterion(critic_pos_out, data.y, reduction='none')[data.batch]
            baseline_loss = self.baseline.criterion(baseline_out, data.y, reduction='none')[data.batch]
            reward = -(critic_pos_loss - baseline_loss)
            loss = self.loss(reward, mask, probs)
            actor_losses.append(loss.item())
        return sum(actor_losses)/len(actor_losses),\
            sum(mask_sizes)/len(mask_sizes),\
                sum(critic_pos_accs)/len(critic_pos_accs),\
                    sum(critic_neg_accs)/len(critic_neg_accs),\
                        sum(baseline_accs)/len(baseline_accs)
    
    @torch.no_grad()
    def predict_batch(self, loader):
        self.eval()
        y_probs = []
        y_masks = []
        critic_pos_preds = []
        critic_neg_preds = []
        baseline_preds = []
        y_trues = []
        for data in loader:
            data = data.to(self.device)
            probs = self(data)
            y_probs.append(probs)
            mask = (probs > 0.5).float()
            y_masks.append(mask)
            critic_pos_out = self.critic(data, mask)
            critic_pos_preds.append(critic_pos_out)
            critic_neg_out = self.critic(data, 1.0 - mask)
            critic_neg_preds.append(critic_neg_out)
            baseline_out = self.baseline(data)
            baseline_preds.append(baseline_out)
            y_trues.append(data.y)
        y_probs = torch.cat(y_probs).detach().cpu().numpy()
        y_masks = torch.cat(y_masks).detach().cpu().numpy()
        critic_pos_preds = torch.cat(critic_pos_preds).detach().cpu().numpy()
        critic_neg_preds = torch.cat(critic_neg_preds).detach().cpu().numpy()
        baseline_preds = torch.cat(baseline_preds).detach().cpu().numpy()
        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        return y_probs, y_masks, critic_pos_preds, critic_neg_preds, baseline_preds, y_trues
