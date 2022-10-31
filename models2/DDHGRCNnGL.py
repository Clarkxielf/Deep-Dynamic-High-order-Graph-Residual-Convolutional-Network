import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm, GATConv, GraphConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
import numpy as np
from models2.graph_clf import GraphClf


class DDHGRCNnGL(torch.nn.Module):
    def __init__(self, feature, out_channel, pooltype):
        super(DDHGRCNnGL, self).__init__()

        self.pool1, self.pool2 = self.poollayer(pooltype)

        self.GConv1 = GraphConv(feature,1024)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = GraphConv(1024,1024)
        self.bn2 = BatchNorm(1024)

        self.p1 = nn.MaxPool1d(2,2)
        self.BN1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.p2 = nn.MaxPool1d(2,2)
        self.BN2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.conv2d1 = nn.Conv2d(1, 128, 1, 1)
        self.relu = nn.ReLU()
        self.bn2d1 = nn.BatchNorm2d(128)
        self.conv2d2 = nn.Conv2d(128, 64, 3, 2)
        self.relu = nn.ReLU()
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv2d3 = nn.Conv2d(64, 1, 1, 1)
        self.relu = nn.ReLU()

        self.conv2d4 = nn.Conv2d(1, 128, 1, 1)
        self.relu = nn.ReLU()
        self.bn2d3 = nn.BatchNorm2d(128)
        self.conv2d5 = nn.Conv2d(128, 64, 3, 2)
        self.relu = nn.ReLU()
        self.bn2d4 = nn.BatchNorm2d(64)
        self.conv2d6 = nn.Conv2d(64, 1, 1, 1)
        self.relu = nn.ReLU()

        self.downsample1 = nn.MaxPool2d(3, 2)
        self.downsample2 = nn.MaxPool2d(3, 2)

        self.fc = nn.Sequential(nn.Linear(9, 1024), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Sequential(nn.Linear(1024, out_channel))

    def forward(self, data, pooltype):
        x, edge_index, batch= data.x, data.edge_index, data.batch

        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, batch = self.poolresult(self.pool1, pooltype, x, edge_index, batch)
        x1 = global_mean_pool(x, batch)

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, batch = self.poolresult(self.pool2, pooltype, x, edge_index, batch)
        x2 = global_mean_pool(x, batch)

        x = x1 + x2

        x = self.p1(x)
        x = self.BN1(x)
        x = self.relu(x)

        x = self.p2(x)
        x = self.BN2(x)
        x = self.relu(x)

        x = x.unsqueeze(1)
        x = torch.reshape(x, ((64,1,16,16)))
        cut = x
        x = self.conv2d1(x)
        x = self.relu(x)
        x = self.bn2d1(x)
        x = self.conv2d2(x)
        x = self.relu(x)
        x = self.bn2d2(x)
        x = self.conv2d3(x)
        x = self.relu(x)
        cut = self.downsample1(cut)
        x = x + cut

        cut = x
        x = self.conv2d4(x)
        x = self.relu(x)
        x = self.bn2d3(x)
        x = self.conv2d5(x)
        x = self.relu(x)
        x = self.bn2d4(x)
        x = self.conv2d6(x)
        x = self.relu(x)
        cut = self.downsample2(cut)
        x = x + cut

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

    def poollayer(self, pooltype):

        self.pooltype = pooltype

        if self.pooltype == 'TopKPool':
            self.pool1 = TopKPooling(1024)
            self.pool2 = TopKPooling(1024)
        elif self.pooltype == 'EdgePool':
            self.pool1 = EdgePooling(1024)
            self.pool2 = EdgePooling(1024)
        elif self.pooltype == 'ASAPool':
            self.pool1 = ASAPooling(1024)
            self.pool2 = ASAPooling(1024)
        elif self.pooltype == 'SAGPool':
            self.pool1 = SAGPooling(1024)
            self.pool2 = SAGPooling(1024)
        else:
            print('Such graph pool method is not implemented!!')

        return self.pool1, self.pool2

    def poolresult(self,pool,pooltype,x,edge_index,batch):

        self.pool = pool

        if pooltype == 'TopKPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'EdgePool':
            x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'ASAPool':
            x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'SAGPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        else:
            print('Such graph pool method is not implemented!!')

        return x, edge_index, batch

