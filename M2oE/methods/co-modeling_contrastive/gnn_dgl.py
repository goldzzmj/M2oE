import dgl
import torch

from conv_dgl import GNN_SpMoE_node
from dgl.nn.pytorch.glob import mean_nodes
import torch.nn.functional as F
import torch.nn as nn

class GMoE(torch.nn.Module):

    def __init__(self, args, num_tasks, num_layer = 5, emb_dim = 300,
                    gnn_type = 'gcn', virtual_node = True, moe=True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    num_experts=3, k=1, coef=1e-2, hop=1, num_experts_1hop=None):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GMoE, self).__init__()

        self.args = args

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_SpMoE_node(args, num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, num_experts=num_experts, k=k, coef=coef, num_experts_1hop=num_experts_1hop)

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        hidden_layer = [args.hidden,args.hidden,args.hidden,args.hidden]
        self.e1 = nn.Linear(args.d_graph, hidden_layer[0])
        self.e2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.e3 = nn.Linear(hidden_layer[1], hidden_layer[2])
        self.e4 = nn.Linear(hidden_layer[2], hidden_layer[3])
        self.sigmoid = torch.nn.Sigmoid()
        self.predictor = nn.Linear(args.hidden, args.output_layer)

        self.mlp = nn.Linear(256,256)

    def forward(self, g):
        with g.local_scope():
            x = g.ndata['feat'] # (9325,)

            h_node = self.gnn_node(g, x) # (num_nodes,256)

            g.ndata['h'] = h_node

            # subgraph_ids = g.ndata['subgraph_id']
            # unique_ids = torch.unique(subgraph_ids)
            # sub_feature = []
            # batch_nodes_mask = []
            # for subgraph_id in unique_ids:
            #     mask = (subgraph_ids == subgraph_id)
            #     subgraph_features = g.ndata['h'][mask]
            #     sub_feature.append(subgraph_features)
            #     # get non-padding node mask
            #     pad_nodes = g.ndata['node_label'][mask].eq(0)
            #     batch_nodes_mask.append(pad_nodes)
            # batch_feature = torch.stack(sub_feature)
            # mask_gra = torch.stack(batch_nodes_mask)
            # updated_features = self.mlp(batch_feature)
            # g.ndata['h'] = torch.reshape(updated_features,(-1,256))


            # graph mean all nodes
            h_node = dgl.mean_nodes(g, 'h')  # (batchsize,256)

            # graph sum all nodes
            # h_node = dgl.sum_nodes(g,'h') # (batchsize,256)

        hidden = F.relu(self.e1(h_node))  # (512,64)
        # Predictor
        h_2 = F.relu(self.e2(hidden))  # (512,64)
        h_3 = F.relu(self.e3(h_2))  # (512,64)
        h_4 = F.relu(self.e4(h_3))  # (512,64)
        y = self.predictor(h_4)  # (512,1)

        if self.args.task_type == 'Classification':
            return F.log_softmax(y, dim=1), hidden
        elif self.args.task_type == 'Regression':
            return y, hidden

if __name__ == '__main__':
    # GMoE(num_tasks = 10)
    print()