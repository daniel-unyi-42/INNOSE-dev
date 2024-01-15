import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import BA2MotifDataset, GNNBenchmarkDataset, ModelNet
from SentiGraphDataset import SentiGraphDataset
from BenzeneDataset import BenzeneDataset
from Explainer import GNN, CriticGNN, ActorGNN
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(123)

# hyperparameters
hidden = 64
lr = 0.001
weight_decay = 0.01
epochs = 100
batch_size = 64
lambda_ = 0.001
accepted_mask_size = 0.5

train_split, val_split, test_split = (0.8, 0.1, 0.1)

train_baseline = False
train_critic = False
train_actor = True

dataset_name = 'MNIST'

####################################################################################################

data_path = f'/home/daniel/src/INNOSE/data/{dataset_name}'
log_path = f'/home/daniel/src/INNOSE/runs/{dataset_name}/lambda_{lambda_}_hidden_{hidden}_lr_{lr}_weight_decay_{weight_decay}_epochs_{epochs}_batch_size_{batch_size}'

data_path = os.path.expanduser(data_path)
log_path = os.path.expanduser(log_path)

train = train_baseline or train_critic or train_actor

if train:
    writer = SummaryWriter(log_path)

assert train_split + val_split + test_split == 1.0

class Transform():
    def __call__(self, data):
        # if data.face is not None:
        #     data = T.FaceToEdge()(data)
        #     data = T.NormalizeScale()(data)
        #     data.x = data.pos.copy()
        if data.edge_attr is not None:
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
        data.y = data.y.long()
        return data

dataset = GNNBenchmarkDataset(data_path, dataset_name, transform=Transform())
# dataset = ModelNet(data_path, transform=Transform())
# dataset = BenzeneDataset(data_path)
dataset = dataset.shuffle()[:7000]

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [train_split, val_split, test_split])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

baseline = GNN(dataset.num_features, hidden, dataset.num_classes, dataset.num_edge_features)
baseline.optimizer = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=weight_decay)
print(baseline)

critic = CriticGNN(dataset.num_features, hidden, dataset.num_classes, dataset.num_edge_features)
critic.optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
print(critic)

actor = ActorGNN(critic, baseline, lambda_)
actor.optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
print(actor)

if train_baseline:
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = baseline.train_batch(train_loader)
        val_loss, val_acc = baseline.test_batch(val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        writer.add_scalar('BASELINE/train_loss', train_loss, epoch)
        writer.add_scalar('BASELINE/train_acc', train_acc, epoch)
        writer.add_scalar('BASELINE/val_loss', val_loss, epoch)
        writer.add_scalar('BASELINE/val_acc', val_acc, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt')))
val_loss, val_acc = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')
test_loss, test_acc = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test ACC: {test_acc:.4f}')

# Regression plot

y_preds, y_trues = baseline.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': y_preds, 'y_true': y_trues})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))
plt.clf()


if train_critic:
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = critic.train_batch(train_loader)
        val_loss, val_acc = critic.test_batch(val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(critic.state_dict(), os.path.join(log_path, 'critic.pt'))
        writer.add_scalar('CRITIC/train_loss', train_loss, epoch)
        writer.add_scalar('CRITIC/train_acc', train_acc, epoch)
        writer.add_scalar('CRITIC/val_loss', val_loss, epoch)
        writer.add_scalar('CRITIC/val_acc', val_acc, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

critic.load_state_dict(torch.load(os.path.join(log_path, 'critic.pt')))
val_loss, val_acc = critic.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')
test_loss, test_acc = critic.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test ACC: {test_acc:.4f}')

# Regression plot

y_preds, y_trues = critic.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': y_preds, 'y_true': y_trues})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))
plt.clf()


if train_actor:
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_mask_size, train_critic_pos_acc, train_critic_neg_acc, train_baseline_acc = actor.train_batch(train_loader)
        val_loss, val_mask_size, val_critic_pos_acc, val_critic_neg_acc, val_baseline_acc = actor.test_batch(val_loader)
        if val_critic_pos_acc > best_val_acc and val_mask_size < accepted_mask_size:
            best_val_acc = val_critic_pos_acc
            torch.save(actor.state_dict(), os.path.join(log_path, 'actor.pt'))
        writer.add_scalar('ACTOR/train_loss', train_loss, epoch)
        writer.add_scalar('ACTOR/train_mask_size', train_mask_size, epoch)
        writer.add_scalar('ACTOR/train_critic_pos_acc', train_critic_pos_acc, epoch)
        writer.add_scalar('ACTOR/train_critic_neg_acc', train_critic_neg_acc, epoch)
        writer.add_scalar('ACTOR/train_baseline_acc', train_baseline_acc, epoch)
        writer.add_scalar('ACTOR/val_loss', val_loss, epoch)
        writer.add_scalar('ACTOR/val_mask_size', val_mask_size, epoch)
        writer.add_scalar('ACTOR/val_critic_pos_acc', val_critic_pos_acc, epoch)
        writer.add_scalar('ACTOR/val_critic_neg_acc', val_critic_neg_acc, epoch)
        writer.add_scalar('ACTOR/val_baseline_acc', val_baseline_acc, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train mask size: {train_mask_size:.4f},\
              Train critic pos ACC: {train_critic_pos_acc:.4f}, Train critic neg ACC: {train_critic_neg_acc:.4f}, Train baseline ACC: {train_baseline_acc:.4f},\
                Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f},\
                    Val critic pos ACC: {val_critic_pos_acc:.4f}, Val critic neg ACC: {val_critic_neg_acc:.4f}, Val baseline ACC: {val_baseline_acc:.4f}')

actor.load_state_dict(torch.load(os.path.join(log_path, 'actor.pt')))
val_loss, val_mask_size, val_critic_pos_acc, val_critic_neg_acc, val_baseline_acc = actor.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f}, Val critic pos ACC: {val_critic_pos_acc:.4f}, Val critic neg ACC: {val_critic_neg_acc:.4f}, Val baseline ACC: {val_baseline_acc:.4f}')
test_loss, test_mask_size, test_critic_pos_acc, test_critic_neg_acc, test_baseline_acc = actor.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test mask size: {test_mask_size:.4f}, Test critic pos ACC: {test_critic_pos_acc:.4f}, Test critic neg ACC: {test_critic_neg_acc:.4f}, Test baseline ACC: {test_baseline_acc:.4f}')

if train:
    writer.close()

# predict
for img, data in enumerate(test_set):
    y_prob, y_mask, _, _, _, y_true = actor.predict_batch([data])
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G, with_labels=False, node_color=y_mask, node_size=100)
    plt.savefig(f'graph_{y_true.item()}_{img}.png')
    plt.clf()




y_prob, y_mask, critic_pos_pred, critic_neg_pred, baseline_pred, y_true = actor.predict_batch(test_loader)
import numpy as np
baseline_acc = (baseline_pred.argmax(axis=1) == y_true).astype(float)
pos_acc = (critic_pos_pred.argmax(axis=1) == y_true).astype(float)
neg_acc = (critic_neg_pred.argmax(axis=1) == y_true).astype(float)
print('Fidelity+ ', np.mean(baseline_acc - neg_acc))
print('Fidelity- ', np.mean(baseline_acc - pos_acc))
# print('Fidelity+ ', np.mean(baseline_pred[:, 1] - critic_neg_pred[:, 1]))
# print('Fidelity- ', np.mean(baseline_pred[:, 1] - critic_pos_pred[:, 1]))
