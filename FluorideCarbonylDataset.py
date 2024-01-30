import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url

class FluorideCarbonylDataset(InMemoryDataset):
    url = 'https://github.com/mims-harvard/GraphXAI/raw/main/graphxai/datasets/real_world/fluoride_carbonyl'
    filename = 'fluoride_carbonyl.npz'

    def __init__(self, root, transform=None, pre_transform=None):
        super(FluorideCarbonylDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return ['fluoride_carbonyl.pt']

    def download(self):
        download_url(f'{self.url}/{self.filename}', self.raw_dir)

    def process(self):
        data_list = []
        data = np.load(f'{self.raw_dir}/{self.raw_file_names}', allow_pickle=True)
        attr, X, y = data['attr'], data['X'], data['y']
        ylist = [y[i][0] for i in range(y.shape[0])]
        X = X[0]
        for i in range(len(X)):
            x = torch.from_numpy(X[i]['nodes'])
            edge_attr = torch.from_numpy(X[i]['edges'])
            y = torch.tensor([ylist[i]]).long()
            e1 = torch.from_numpy(X[i]['receivers']).long()
            e2 = torch.from_numpy(X[i]['senders']).long()
            edge_index = torch.stack([e1, e2])
            true = torch.from_numpy(attr[i][0]['nodes']).float()
            data = Data(
                x = x,
                y = y,
                edge_attr = edge_attr,
                edge_index = edge_index,
                true = true,
            )
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
