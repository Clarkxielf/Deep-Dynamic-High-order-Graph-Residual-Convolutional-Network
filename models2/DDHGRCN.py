import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm, GATConv, GraphConv # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool
import numpy as np
from models2.graph_clf import GraphClf
# HoGCN套用图学习与卷积
config = {
'model_name': 'GraphClf',
'graph_skip_conn': 0.3,
'sparsity_ratio': 0.3,
'scalable_run': False,
'num_anchors': 200,
'hidden_size': 512,
'use_bert': 'False',
'dropout': 0.3,
'gl_dropout': 0.3,
'bignn': False,
'graph_module': 'gcn',
'graph_type': 'dynamic',
'graph_learn': True,
'graph_metric_type': 'weighted_cosine',
'graph_include_self': False,
'update_adj_ratio': 0.7,
'graph_learn_regularization': True,
'smoothness_ratio': 0.1,
'degree_ratio': 0.1,
'graph_learn_ratio': 0,
'input_graph_knn_size': 20,
'graph_learn_hidden_size': 'null',
'graph_learn_epsilon': 0.75,
'graph_learn_topk': None,
'graph_learn_num_pers': 1,
'graph_hops': 2,
'gat_nhead': 8,
'gat_alpha': 0.2,
'optimizer': 'adam',
'learning_rate': 0.01,
'weight_decay': 0.0005,
'lr_patience': 2,
'lr_reduce_factor': 0.5,
'grad_clipping': None,
'grad_accumulated_steps': 1,
'eary_stop_metric': 'nloss',
'pretrain_epoch': 50,
'max_iter': 10,
'eps_adj': 1e-3,
'rl_ratio': 0,
'rl_ratio_power': 1,
'rl_start_epoch': 1,
'max_rl_ratio': 0.99,
'rl_reward_metric': 'acc',
'rl_wmd_ratio': 0,
'random_seed': 1234,
'shuffle': True,
'max_epochs': 10000,
'patience': 100,
'verbose': 20,
'print_every_epochs': 500,
'out_predictions': False,
'out_raw_learned_adj_path': 'wine_idgl_anchor_raw_learned_adj.npy',
'save_params': True,
'logging': True,
'no_cuda': False,
'device': 'cuda:0',
'cuda_id': 0,
'num_feat': 512,
'num_class': 5
}

class DDHGRCN(torch.nn.Module):
    def __init__(self, feature, out_channel, pooltype):
        super(DDHGRCN, self).__init__()

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

    def forward(self, node, edge_index, batch, pooltype):

        x = self.GConv1(node, edge_index)
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




    def graph_learner(self, feature):
        self.learn = GraphClf(config)
        x, edge_index, batch, edge_attr = feature.x, feature.edge_index, feature.batch, feature.edge_attr

        init_adj = index_to_adj(edge_attr, edge_index)

        features = F.dropout(x, self.learn.config.get('feat_adj_dropout', 0), training=self.learn.training)
        init_node_vec = features
        init_node_vec = init_node_vec.to('cuda:0')
        self.learn.graph_learner.to('cuda:0')

        cur_raw_adj, cur_adj = self.learn.learn_graph(self.learn.graph_learner, init_node_vec, config['graph_skip_conn'], graph_include_self=False, init_adj=init_adj)
        cur_adj = F.dropout(cur_adj, self.learn.config.get('feat_adj_dropout', 0), training=self.learn.training)
        cur_adj.to('cuda:0')
        self.learn.encoder.graph_encoders[0].to('cuda:0')
        self.learn.encoder.graph_encoders[-1].to('cuda:0')

        node_vec = torch.relu(self.learn.encoder.graph_encoders[0](init_node_vec, cur_adj))
        node_vec = F.dropout(node_vec, self.learn.dropout, training=self.learn.training)

        node_vec.to('cuda:0')

        output = self.learn.encoder.graph_encoders[-1](node_vec, cur_adj)
        output = F.log_softmax(output, dim=-1)

        edge_index = adj_to_index(cur_adj)

        return output, init_node_vec, edge_index




def index_to_adj(edge_attr, edge_index):
    edge_attr = edge_attr.to('cpu')
    edge_index = edge_index.to('cpu')
    edge_attr = np.array(edge_attr)
    edge_index = np.array(edge_index)
    edge_attr = edge_attr.tolist()
    edge_index = edge_index.tolist()

    adj = torch.zeros([640, 640], dtype=torch.float32, device='cuda:0')

    for i in range(3200):
        a = edge_index[0][i]
        b = edge_index[1][i]
        c = edge_attr[i]
        adj[a][b] = c

    adj.to('cuda:0')

    return adj

def adj_to_index(adj):
    adj = adj.to('cpu')
    adj = adj.detach().numpy()
    edge_attr = list()
    hang2 = list()
    hang1 = list()
    for i in range(640):
        for j in range(640):
            if adj[i][j] > 0.1:
                edge_attr.append(adj[i][j])
                hang1.append(i)
                hang2.append(j)
            else:
                j += 1


    hang1 = torch.tensor(hang1, device='cuda:0')
    hang2 = torch.tensor(hang2, device='cuda:0')

    hang1 = torch.unsqueeze(hang1, dim=0)
    hang2 = torch.unsqueeze(hang2, dim=0)

    edge_index = torch.cat((hang1, hang2), 0)

    return edge_index