import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, is_undirected, to_undirected
from torch_geometric.datasets import BA2MotifDataset, GNNBenchmarkDataset, TUDataset
from BAMultiShapesDataset import BAMultiShapesDataset
from AlkaneCarbonylDataset import AlkaneCarbonylDataset
from BenzeneDataset import BenzeneDataset
from FluorideCarbonylDataset import FluorideCarbonylDataset
from SentiGraphDataset import SentiGraphDataset
from Explainer import GNN, CriticGNN, ActorGNN
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(123)

print_results = True

# hyperparameters
hidden = 256
lr = 0.001
epochs = 100
batch_size = 256
lambda_ = 0.1
# sparsity = 0.1

train_split, val_split, test_split = (0.8, 0.1, 0.1)

train_baseline = True
train_critic = True
train_actor = True

dataset_name = 'Graph-SST2'

####################################################################################################

data_path = f'/home/daniel/src/INNOSE/data/{dataset_name}'
log_path = f'/home/daniel/src/INNOSE/runs/{dataset_name}'

data_path = os.path.expanduser(data_path)
log_path = os.path.expanduser(log_path)

train = train_baseline or train_critic or train_actor

if train:
    writer = SummaryWriter(log_path)

assert train_split + val_split + test_split == 1.0

class Transform():
    def __call__(self, data):
        if not hasattr(data, 'true'):
            data.true = torch.zeros(data.num_nodes)
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR10':
            data = T.ToUndirected()(data)
            data.x = torch.cat([data.x, data.pos], dim=1)
            data.edge_attr = data.edge_attr.unsqueeze(-1)
        elif dataset_name == 'BA2Motif':
            data.true[-5:] = 1.0
        elif dataset_name == 'BAMultiShapes':
            def subgraph_matching(G, g):
                gm = nx.algorithms.isomorphism.GraphMatcher(G, g)
                return list(gm.mapping.keys()) if gm.subgraph_is_isomorphic() else []
            G1 = to_networkx(data, to_undirected=True)
            g2 = nx.generators.small.house_graph()
            g3 = nx.generators.lattice.grid_2d_graph(3, 3)
            g4 = nx.generators.classic.wheel_graph(6)
            gm2 = subgraph_matching(G1, g2)
            gm3 = subgraph_matching(G1, g3)
            gm4 = subgraph_matching(G1, g4)
            data.true[gm2 + gm3 + gm4] = 1.0
        elif dataset_name == 'AlkaneCarbonyl' or dataset_name == 'Benzene' or dataset_name == 'FluorideCarbonyl':
            data.true = torch.any(data.true, dim=1).float()
        elif dataset_name == 'Graph-SST2' or dataset_name == 'Graph-Twitter':
            data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
        assert is_undirected(data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
        return data

# dataset = BA2MotifDataset(data_path, pre_transform=Transform())
# dataset = BAMultiShapesDataset(data_path, pre_transform=Transform())
# dataset = AlkaneCarbonylDataset(data_path, pre_transform=Transform())
# dataset = BenzeneDataset(data_path, pre_transform=Transform())
# dataset = FluorideCarbonylDataset(data_path, pre_transform=Transform())
# dataset = TUDataset(data_path, dataset_name, pre_transform=Transform())
# dataset = GNNBenchmarkDataset(data_path, dataset_name, pre_transform=Transform())
dataset = SentiGraphDataset(data_path, dataset_name, pre_transform=Transform())

dataset = dataset.shuffle()[:3500]
print('Length of dataset: ', len(dataset))

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                            [train_split, val_split, test_split])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

if hasattr(dataset, 'num_node_features'):
    num_features = dataset.num_node_features
else:
    num_features = dataset[0].x.shape[1]

if hasattr(dataset, 'num_classes'):
    num_classes = dataset.num_classes
else:
    num_classes = dataset[0].y.max().item() + 1

baseline = GNN(num_features, hidden, num_classes, dataset.num_edge_features)
baseline.optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
#print(baseline)

if train_baseline:
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = baseline.train_batch(train_loader)
        val_loss, val_acc = baseline.test_batch(val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('Saving baseline...')
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        writer.add_scalar('BASELINE/train_loss', train_loss, epoch)
        writer.add_scalar('BASELINE/train_acc', train_acc, epoch)
        writer.add_scalar('BASELINE/val_loss', val_loss, epoch)
        writer.add_scalar('BASELINE/val_acc', val_acc, epoch)
        if print_results:
            print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt')))
val_loss, val_acc = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')
test_loss, test_acc = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test ACC: {test_acc:.4f}')

# # Regression plot

# y_preds, y_trues = baseline.predict_batch(test_loader)

# plot_df = pd.DataFrame({'y_pred': y_preds, 'y_true': y_trues})
# figure = plt.figure(figsize=(4, 4))
# ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
# ax.set(xlabel='prediction', ylabel='ground truth')
# plt.savefig(os.path.join(log_path, 'regression.png'))
# plt.clf()


for sparsity in [0.1]:

    print(f'Running for sparsity {sparsity}')
    log_path_with_sparsity = os.path.join(log_path, str(sparsity))
    os.makedirs(log_path_with_sparsity, exist_ok=True)
    if train:
        writer.close()
        writer = SummaryWriter(log_path_with_sparsity)

    critic = CriticGNN(num_features, hidden, num_classes, dataset.num_edge_features, sparsity)
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    #print(critic)

    actor = ActorGNN(critic, baseline, lambda_)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    #print(actor)

    if train_critic:
        best_val_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = critic.train_batch(train_loader)
            val_loss, val_acc = critic.test_batch(val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print('Saving critic...')
                torch.save(critic.state_dict(), os.path.join(log_path_with_sparsity, 'critic.pt'))
            writer.add_scalar('CRITIC/train_loss', train_loss, epoch)
            writer.add_scalar('CRITIC/train_acc', train_acc, epoch)
            writer.add_scalar('CRITIC/val_loss', val_loss, epoch)
            writer.add_scalar('CRITIC/val_acc', val_acc, epoch)
            if print_results:
                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')

    critic.load_state_dict(torch.load(os.path.join(log_path_with_sparsity, 'critic.pt')))
    val_loss, val_acc = critic.test_batch(val_loader)
    print(f'Val loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}')
    test_loss, test_acc = critic.test_batch(test_loader)
    print(f'Test loss: {test_loss:.4f}, Test ACC: {test_acc:.4f}')

    # # Regression plot

    # y_preds, y_trues = critic.predict_batch(test_loader)

    # plot_df = pd.DataFrame({'y_pred': y_preds, 'y_true': y_trues})
    # figure = plt.figure(figsize=(4, 4))
    # ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
    # ax.set(xlabel='prediction', ylabel='ground truth')
    # plt.savefig(os.path.join(log_path, 'regression.png'))
    # plt.clf()


    if train_actor:
        primary = 0
        secondary = 1
        for epoch in range(epochs*2):
            train_loss, train_mask_size, train_critic_pos_acc, train_critic_neg_acc, train_baseline_acc = actor.train_batch(train_loader)
            val_loss, val_mask_size, val_critic_pos_acc, val_critic_neg_acc, val_baseline_acc = actor.test_batch(val_loader)
            if (val_critic_pos_acc - primary > 0.015 and val_mask_size < sparsity + 0.015) or (abs(val_critic_pos_acc - primary) < 0.015 and val_critic_neg_acc < secondary and val_mask_size < sparsity + 0.015):
                primary = val_critic_pos_acc
                print('Saving actor...')
                torch.save(actor.state_dict(), os.path.join(log_path_with_sparsity, f'actor.pt'))
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
            if print_results:
                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train mask size: {train_mask_size:.4f},\
                    Train critic pos ACC: {train_critic_pos_acc:.4f}, Train critic neg ACC: {train_critic_neg_acc:.4f}, Train baseline ACC: {train_baseline_acc:.4f},\
                        Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f},\
                            Val critic pos ACC: {val_critic_pos_acc:.4f}, Val critic neg ACC: {val_critic_neg_acc:.4f}, Val baseline ACC: {val_baseline_acc:.4f}')

    actor.load_state_dict(torch.load(os.path.join(log_path_with_sparsity, f'actor.pt')))
    val_loss, val_mask_size, val_critic_pos_acc, val_critic_neg_acc, val_baseline_acc = actor.test_batch(val_loader)
    print(f'Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f}, Val critic pos ACC: {val_critic_pos_acc:.4f}, Val critic neg ACC: {val_critic_neg_acc:.4f}, Val baseline ACC: {val_baseline_acc:.4f}')
    test_loss, test_mask_size, test_critic_pos_acc, test_critic_neg_acc, test_baseline_acc = actor.test_batch(test_loader)
    print(f'Test loss: {test_loss:.4f}, Test mask size: {test_mask_size:.4f}, Test critic pos ACC: {test_critic_pos_acc:.4f}, Test critic neg ACC: {test_critic_neg_acc:.4f}, Test baseline ACC: {test_baseline_acc:.4f}')


    # predict
    _, _, critic_pos_preds, _, _, _, _ = actor.predict_batch(test_loader)
    for img, data in enumerate(test_set):
        y_prob, y_mask, critic_pos_pred, _, _, y_true, _ = actor.predict_batch([data])
        G = to_networkx(data, to_undirected=True)
        if dataset_name == 'MNIST':
            nx.draw_networkx(G, pos=data.pos.cpu().numpy(), with_labels=False, node_color=data.x[:, 0].cpu().numpy(), node_size=y_mask[:, 0]*100+10)
        elif dataset_name == 'CIFAR10':
            nx.draw_networkx(G, pos=data.pos.cpu().numpy(), with_labels=False, node_color=data.x[:, :3].cpu().numpy(), node_size=y_mask[:, 0]*100+10)
        elif dataset_name == 'Mutagenicity':
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            node_labels = {i:node_dict[atom] for i, atom in enumerate(data.x.argmax(dim=1).cpu().numpy())}
            nx.draw_networkx(G, labels=node_labels, node_color=y_mask)
        elif dataset_name == 'AlkaneCarbonyl' or dataset_name == 'Benzene' or dataset_name == 'FluorideCarbonyl':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'F', 5: 'P', 6: 'Cl', 7: 'Br', 8: 'Na', 9: 'Ca', 10: 'I', 11: 'B', 12: 'H', 13: '*'}
            #edge_dict = {0: 'red', 1: 'green', 2: 'blue', 3: 'black', 4: 'grey'}
            node_labels = {i:node_dict[atom]+'_'+str(data.true.cpu().numpy()[i].item()) for i, atom in enumerate(data.x.argmax(dim=1).cpu().numpy())}
            node_labels_other = {i:label for i, label in enumerate(data.true.cpu().numpy())}
            #edge_colors = [edge_dict[bond] for bond in data.edge_attr.argmax(dim=1).cpu().numpy()]
            nx.draw_networkx(G, labels=node_labels, node_color=y_mask, edge_color='black')
        else:
            node_labels = {i:label for i, label in enumerate(data.true.cpu().numpy())}
            nx.draw_networkx(G, labels=node_labels, node_color=y_mask, node_size=100)
        plt.savefig(f'{log_path_with_sparsity}/graph_{y_true.item()}_{critic_pos_preds[img].argmax().item()}_{img}.png')
        plt.clf()




    y_prob, y_mask, critic_pos_pred, critic_neg_pred, baseline_pred, y_true, explanation = actor.predict_batch(test_loader)
    from sklearn.metrics import confusion_matrix
    import numpy as np
    np.set_printoptions(suppress=True)
    baseline_acc = (baseline_pred.argmax(axis=1) == y_true).astype(float)
    pos_acc = (critic_pos_pred.argmax(axis=1) == y_true).astype(float)
    neg_acc = (critic_neg_pred.argmax(axis=1) == y_true).astype(float)
    print('Fidelity+ ', np.mean(baseline_acc - neg_acc))
    print('Fidelity- ', np.mean(baseline_acc - pos_acc))

    print('Fidelity+ ', np.mean(baseline_pred[:, 1] - critic_neg_pred[:, 1]))
    print('Fidelity- ', np.mean(baseline_pred[:, 1] - critic_pos_pred[:, 1]))

    print(confusion_matrix(explanation, y_mask))


if train:
    writer.close()
