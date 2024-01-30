from networkx.generators import random_graphs, lattice, small, classic
import networkx as nx
import pickle as pkl
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url



def merge_graphs(g1, g2, nb_random_edges=1):
    mapping = dict()
    max_node = max(g1.nodes())

    i = 1
    for n in g2.nodes():
        mapping[n] = max_node + i
        i = i + 1
    g2 = nx.relabel_nodes(g2,mapping)

    g12 = nx.union(g1,g2)
    for i in range(nb_random_edges):
        e1 = list(g1.nodes())[np.random.randint(0,len(g1.nodes()))]
        e2 = list(g2.nodes())[np.random.randint(0,len(g2.nodes()))]
        g12.add_edge(e1,e2)        
    return g12

def generate_class1(nb_random_edges, nb_node_ba=40):
    r = np.random.randint(3)
    
    if r == 0: # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-9, 1)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
    elif r == 1: # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-5, 1)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = small.house_graph()
        g123 = merge_graphs(g12,g3,nb_random_edges)
    elif r == 2: # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5-9, 1)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
    return g123

def generate_class0(nb_random_edges, nb_node_ba=40):
    r = np.random.randint(10)
    
    if r > 3:
        g12 = random_graphs.barabasi_albert_graph(nb_node_ba, 1) 
    if r == 0: # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6, 1)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)      
    if r == 1: # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5, 1)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)      
    if r == 2: # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9, 1)
        g2 = lattice.grid_2d_graph(3, 3)
        g12 = merge_graphs(g1,g2,nb_random_edges)            
    if r == 3: # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9-5-6, 1)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
        g4 =  classic.wheel_graph(6)
        g12 = merge_graphs(g123,g4,nb_random_edges)
    return g12

def generate(num_samples):
    assert num_samples % 2 == 0
    adjs = []
    labels = []
    feats = []
    nb_node_ba = 40

    for _ in range(int(num_samples/2)):
        g = generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba)
        adjs.append(nx.adjacency_matrix(g).A)
        labels.append(0)
        feats.append(list(np.ones((len(g.nodes()),10))/10))

    for _ in range(int(num_samples/2)):
        g = generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba)
        adjs.append(nx.adjacency_matrix(g).A)
        labels.append(1)
        feats.append(list(np.ones((len(g.nodes()), 10))/10))
    return adjs,feats,labels 

def save(data):
    f = open('BAMultiShapes2.pkl','wb')
    pkl.dump(data,f)
    f.close()



class BAMultiShapesDataset(InMemoryDataset):
    url = 'https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/datasets/BAMultiShapes'
    filename = 'BAMultiShapes.pkl'

    def __init__(self, root, transform=None, pre_transform=None):
        super(BAMultiShapesDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return ['bamultishapes.pt']

    def download(self):
        download_url(f'{self.url}/{self.filename}', self.raw_dir)
        # # generate dataset:
        # data = generate(1000)
        # save(data)

    def process(self):
        data_list = []
        f = open(f'{self.raw_dir}/{self.raw_file_names}', 'rb')
        adjs, feats, labels = pkl.load(f)
        f.close()
        for adj, feat, label in zip(adjs, feats, labels):
            adj = np.array(adj)
            edge_index = []
            for i in range(len(adj)):
                for j in range(len(adj)):
                    if adj[i][j] == 1:
                        edge_index.append([i, j])
            data = Data(
                x=torch.tensor(np.array(feat), dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y=torch.tensor(label, dtype=torch.long)
            )
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
