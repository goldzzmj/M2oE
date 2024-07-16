import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from utils_graph import *
from model import *
from M2oE import *
from loss import info_nce_loss, get_kl_loss, get_graph_seq_kl_loss
import argparse
import warnings
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

from gnn_dgl import GMoE
from conv_dgl import GNN_SpMoE_node

from tqdm import tqdm


parser = argparse.ArgumentParser()

# Model & Training Settings
parser.add_argument('--dataset', type=str, default='AMP',
                    choices=['AMP', 'AP', 'PepDB', 'RT'])
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--seed', type=int, default=5,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seq_lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Initial learning rate.')

# Sequential Model Parameters
parser.add_argument('--src_vocab_size_seq', type=int, default=21,
                    help='20 natural amino acids plus padding token')
parser.add_argument('--model_seq', type=str, default='Transformer')
parser.add_argument('--d_model', type=int, default=64,
                    help='Output dim of self-attention layers')
parser.add_argument('--fc_layers', type=int, default=4,
                    help='Predictor MLP layers')
parser.add_argument('--d_k', type=int, default=64,
                    help='Hidden dim of K and Q')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Num of self-attention layers')
parser.add_argument('--n_heads', type=int, default=3,
                    help='Num of head for multi-head attention')

# Graphical Model Parameters
parser.add_argument('--model_graph', type=str, default='GraphSAGE')
parser.add_argument('--GraphSAGE_aggregator', type=str, default='lstm',
                    help='Aggregation function of GraphSAGE')
parser.add_argument('--d_graph', type=int, default=256)
parser.add_argument('--src_vocab_size_graph', type=int, default=15,
                    help='15 types of beads')
parser.add_argument('--graph_layer', type=int, default=1)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--drop_ratio', type=float, default=0.2)
parser.add_argument('--num_experts', '-n', type=int, default=5,
                    help='total number of experts in GCN-MoE')
parser.add_argument('-k', type=int, default=2,
                    help='selected number of experts in GCN-MoE')
parser.add_argument('--gnn_depth', type=int, default=1,
                    help='Each GNN layer depth')

# Decoder model Parameters
parser.add_argument('--decoder_depth', type=int, default=6,
                    help='Decoder layer depth')

# InfoNCE Params
parser.add_argument('--n_views', type=int, default=2,
                    help='Num of positive pairs')
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--nce_weight', type=float, default=1e-4)
parser.add_argument('--kl_weight', type=float, default=1e-4)
parser.add_argument('--seq_graph_weight', type=float, default=0.5)
parser.add_argument('--use_infonce_loss', type=str, default=False, help='Contrastive loss')

# Use VAE structure
parser.add_argument('--model_mode', type=str, default='moe', help='chose seqenece model:"vae"、"moe、transformer"')

# plot val acc
parser.add_argument('--plot_pic', type=str, default=True)

args = parser.parse_args()

# The type of task
if args.dataset in ['AP','RT']:
    args.task_type = 'Regression'
elif args.dataset in ['AMP','PepDB']:
    args.task_type = 'Classification'
else: 
    warnings.warn('Dataset with undefined task')

# The maximum length of the peptides from each dataset 
if args.dataset in ['AP']:
    args.src_len = 10
elif args.dataset == ['RT','AMP','PepDB']:
    args.src_len = 50
else: 
    args.src_len = 100

# Set the default device 
args.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
# Set model saving/loading path
if not os.path.exists('./saved_models'):
    os.mkdir('/root/autodl-tmp/RepCon2/RepCon-main/methods/co-modeling_contrastive/saved_models')
args.model_path='/root/autodl-tmp/RepCon2/RepCon-main/methods/co-modeling_contrastive/saved_models'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  

def main():

    # Read raw peptide dataset in FASTA form
    df_seq_train = pd.read_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/{}/train.csv'.format(args.dataset))
    df_seq_valid = pd.read_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/{}/valid.csv'.format(args.dataset))
    df_seq_test = pd.read_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/{}/test.csv'.format(args.dataset))

    # Convert dataset from amino acid sequences to molecular graphs
    if not os.path.exists(r'/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}'.format(args.dataset)):
        os.mkdir(r'/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}'.format(args.dataset))
    if not os.path.exists(r'/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}/test.bin'.format(args.dataset)):
        make_dgl_dataset(args)

    # Build data in DGL graph (coarse-grained molecular graphs)
    train_dataset_graph, train_label = dgl.load_graphs('/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}/train.bin'.format(args.dataset))
    valid_dataset_graph, valid_label = dgl.load_graphs('/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}/valid.bin'.format(args.dataset))
    test_dataset_graph, test_label = dgl.load_graphs('/root/autodl-tmp/RepCon2/RepCon-main/graph_dataset/{}/test.bin'.format(args.dataset))

    # 获取最大的节点数量
    max_train_nodes = max(graph.number_of_nodes() for graph in train_dataset_graph)
    max_valid_nodes = max(graph.number_of_nodes() for graph in valid_dataset_graph)
    max_test_nodes = max(graph.number_of_nodes() for graph in test_dataset_graph)

    new_train_dataset_graph = pad_graphs(train_dataset_graph, max_train_nodes)
    new_valid_dataset_graph = pad_graphs(valid_dataset_graph, max_valid_nodes)
    new_test_dataset_graph = pad_graphs(test_dataset_graph, max_test_nodes)

    # # 添加子图ID特征
    # for i, subgraph in enumerate(train_dataset_graph):
    #     subgraph.ndata['subgraph_id'] = torch.tensor([i] * subgraph.number_of_nodes())

    for i, graph in enumerate(new_train_dataset_graph):
        # 假设每个子图的所有节点都有一个唯一的子图ID
        graph.ndata['subgraph_id'] = torch.full((graph.number_of_nodes(),), i, dtype=torch.long)

    for i, graph in enumerate(new_valid_dataset_graph):
        # 假设每个子图的所有节点都有一个唯一的子图ID
        graph.ndata['subgraph_id'] = torch.full((graph.number_of_nodes(),), i, dtype=torch.long)

    for i, graph in enumerate(new_test_dataset_graph):
        # 假设每个子图的所有节点都有一个唯一的子图ID
        graph.ndata['subgraph_id'] = torch.full((graph.number_of_nodes(),), i, dtype=torch.long)

    # from torch_geometric.data import Data
    #
    # u, v = train_dataset_graph[0].edges()
    # edge_index = torch.stack([u, v], dim=0)
    # x = train_dataset_graph[0].ndata['feat']
    #
    # y = train_label['labels'][0]
    # data = Data(x=x, edge_index=edge_index, y=y)

    # Process the labels
    if args.task_type == 'Classification':
        train_label = train_label['labels'].long()
        valid_label = valid_label['labels'].long()
        test_label = test_label['labels'].long()
        args.output_layer = int(train_label.max()) - int(train_label.min()) + 1
    elif args.task_type == 'Regression':
        train_label = train_label['labels'].float()
        valid_label = valid_label['labels'].float()
        test_label = test_label['labels'].float()
        args.output_layer = 1
        args.label_max = train_label.max().item()
        args.label_min = train_label.min().item()
        # Normalize the regression labels by min-max
        train_label = (train_label - args.label_min) / (args.label_max - args.label_min)
        valid_label = (valid_label - args.label_min) / (args.label_max - args.label_min)
        test_label = (test_label - args.label_min) / (args.label_max - args.label_min)

    # Convert sequential peptide data from pandas dataframe to torch tensor
    train_feat = np.array(df_seq_train["Feature"])
    valid_feat = np.array(df_seq_valid["Feature"])
    test_feat = np.array(df_seq_test["Feature"])

    train_dataset_seq, train_dataset_pad_len = make_seq_data_TDVAE(train_feat, args.src_len)  # (54159,10)
    valid_dataset_seq, valid_dataset_pad_len = make_seq_data_TDVAE(valid_feat, args.src_len)  # (4000,10)
    test_dataset_seq, test_dataset_pad_len = make_seq_data_TDVAE(test_feat, args.src_len)  # (4000,10)

    train_dataset_seq, train_dataset_pad_len = train_dataset_seq.to(device), train_dataset_pad_len
    valid_dataset_seq, valid_dataset_pad_len = valid_dataset_seq.to(device), valid_dataset_pad_len
    test_dataset_seq, test_dataset_pad_len = test_dataset_seq.to(device), test_dataset_pad_len

    # Build DataLoaders
    train_dataset = MyDataSet_TDVAE(new_train_dataset_graph, train_dataset_seq, train_label, train_dataset_pad_len)
    train_loader = Data.DataLoader(train_dataset, args.batch_size, True,
                                   collate_fn=collate)
    valid_dataset = MyDataSet_TDVAE(new_valid_dataset_graph, valid_dataset_seq, valid_label, valid_dataset_pad_len)
    valid_loader = Data.DataLoader(valid_dataset, args.batch_size, False,
                                   collate_fn=collate)
    test_dataset = MyDataSet_TDVAE(new_test_dataset_graph, test_dataset_seq, test_label, test_dataset_pad_len)
    test_loader = Data.DataLoader(test_dataset, args.batch_size, False,
                                  collate_fn=collate)

    # ---------------- Train Phase ----------------#

    if args.mode == 'train':

        # Initialize the models and the optimizers
        M2oE_model = M2oE(args, args.src_vocab_size_seq, args.d_model, args.d_k,
                          args.d_model, args.src_len, max_train_nodes, args.task_type, args.GraphSAGE_aggregator, args.num_experts,
                          args.n_heads, args.d_model, args.n_layers, args.gnn_depth).to(args.device)


        optimizer = optim.Adam(M2oE_model.parameters(), lr=args.lr)
        # Graph_optimizer = optim.AdamW(M2oE_model.parameters(), lr=args.graph_lr)
        # Graph_optimizer = optim.SGD(M2oE_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        # Graph_scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        # Graph_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

        # Initialize MSE loss and CE loss for Infonce
        loss_mse = torch.nn.MSELoss().to(args.device)
        loss_infonce = torch.nn.CrossEntropyLoss().to(args.device)

        # Initialize the validation performance
        valid_mse_saved = 1e5
        valid_acc_saved = 0

        # Plot validation Mse dict
        valid_mse_dict = {}
        valid_acc_dict = {}

        # Train models by epoches of training data
        for epoch in tqdm(range(args.epochs)):
            M2oE_model.train()
            # Extract a batch of data
            for graphs, sequences, labels, pad_len in train_loader:
                graphs = graphs.to(args.device)
                sequences = sequences.to(args.device)
                labels = labels.to(args.device)

                # subgraph_ids2 = graphs.ndata['subgraph_id']
                # seen = set()
                # result = [x.item() for x in subgraph_ids2 if not (x.item() in seen or seen.add(x.item()))]
                # print(result)
                #
                # for subgraph_id in result:
                #     subgraph_nodes = (graphs.ndata['subgraph_id'] == subgraph_id).nonzero().squeeze(1)
                #     subgraph_features = graphs.ndata['feat'][subgraph_nodes]

                # Forward Models
                predict, loss_moe = M2oE_model(sequences, graphs, pad_len)

                # Supervised loss
                if args.task_type == 'Classification':
                    loss_Supervised = F.nll_loss(predict, labels)
                elif args.task_type == 'Regression':
                    loss_Supervised = loss_mse(predict, labels)

                # # Unsupervised InfoNCE loss
                # if args.use_infonce_loss:
                #     hid_pairs = torch.cat([seq_hid, graph_hid], 0)  # (1024,64)
                #     logits, cont_labels = info_nce_loss(args, hid_pairs)  # logits:(1024,1023), cont_labels:(1024,)
                #     l_infonce = args.nce_weight * loss_infonce(logits, cont_labels)

                # loss = loss_Supervised + loss_moe / args.batch_size
                # loss = loss_Supervised + loss_moe
                loss = loss_Supervised
                # print(f'loss:{loss}')

                # Update model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    M2oE_model.eval()

                    # Derive prediction results for all samples in validation set
                    predicts = []
                    for graphs, sequences, labels, pad_len in valid_loader:
                        outputs, _ = M2oE_model(sequences,graphs,pad_len)
                        # predicts = predicts + outputs.cpu().detach().numpy().tolist()
                        predicts = predicts + outputs.cpu().detach().numpy().tolist()
                    predicts = torch.tensor(predicts)

                    # Print Acc. or MSE on validation set
                    if args.task_type == 'Classification':
                        valid_acc = accuracy(predicts, valid_label).item()
                        if valid_acc_saved < valid_acc:
                            valid_acc_saved = valid_acc
                            valid_acc_dict[int(epoch)] = float(valid_acc)
                            print('Epoch:', epoch + 1)
                            print('Valid Performance:', valid_acc)
                        torch.save(M2oE_model.state_dict(),
                                   args.model_path + '{}_Cont_M2oE.pt'.format(args.dataset))

                    elif args.task_type == 'Regression':
                        # print('predicts shape:',predicts.size())
                        # print('valid_label shape:',valid_label.size())
                        valid_mse = loss_mse(predicts * (args.label_max - args.label_min),
                                             valid_label * (args.label_max - args.label_min)).item()
                        if valid_mse_saved > valid_mse:
                            valid_mse_saved = valid_mse
                            valid_mse_dict[int(epoch)] = float(valid_mse)
                            print('Epoch:', epoch + 1)
                            print('Valid Performance:', valid_mse)
                            torch.save(M2oE_model.state_dict(),
                                       args.model_path + '{}_Cont_M2oE.pt'.format(args.dataset))

        if args.plot_pic:
            if args.task_type == 'Classification':
                x_values = list(valid_acc_dict.keys())
                y_values = list(valid_acc_dict.values())
                print(valid_acc_dict)
            elif args.task_type == 'Regression':
                # 将字典的键和值分别提取出来作为X轴和Y轴的数据
                x_values = list(valid_mse_dict.keys())
                y_values = list(valid_mse_dict.values())
                print(valid_mse_dict)

            # 创建折线图
            plt.figure(figsize=(10, 5))  # 设置图像大小
            plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')  # 绘制数据点及折线
            plt.title('Epoch vs Val-Mse')  # 设置图标题
            plt.xlabel('Epoch')  # 设置x轴标签
            plt.ylabel('Acc')  # 设置y轴标签
            plt.grid(True)  # 显示网格
            plt.show()

    # ---------------- Test Phase----------------#
    M2oE_model_load = M2oE(args, args.src_vocab_size_seq, args.d_model, args.d_k,
                      args.d_model, args.src_len, max_train_nodes, args.task_type, args.GraphSAGE_aggregator, args.num_experts,
                      args.n_heads, args.d_model, args.n_layers, args.gnn_depth).to(args.device)
    M2oE_model_load.load_state_dict(torch.load(args.model_path + '{}_Cont_M2oE.pt'.format(args.dataset)))

    # M2oE_model_load.alpha.data = torch.tensor(0.8133)

    print('M2oE_model_alpha value:',M2oE_model_load.alpha.data.item())

    # Seq_model_load.eval()
    M2oE_model_load.eval()

    with torch.no_grad():
        predicts = []
        for graphs, sequences, labels, pad_len in test_loader:
            outputs, _ = M2oE_model_load(sequences, graphs, pad_len)
            predicts = predicts + outputs.cpu().detach().numpy().tolist()
        predicts = torch.tensor(predicts)
   
    df_test = pd.read_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/{}/test.csv'.format(args.dataset))

    if args.task_type == 'Classification':
        predict_tensor = predicts.max(1)[1].type_as(test_label)
        predict = predict_tensor.cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        test_acc = accuracy(predicts,test_label).item()
        df_test_save['Acc'] = test_acc
        print(f'Acc:{test_acc}')

        if args.output_layer >= 2:
            from sklearn.metrics import precision_score, recall_score, f1_score
            Precision = precision_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            Recall = recall_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            F1_score = f1_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['Precision'] = precision_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['Recall'] = recall_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')

        print(f'Acc:{test_acc}')
        print(f'Precision:{Precision}')
        print(f'Recall:{Recall}')
        print(f'F1_score:{F1_score}')

        # if not os.path.exists('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling contrastive'.format(args.dataset)):
        #     os.mkdir('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling_contrastive'.format(args.dataset))
        # df_test_save.to_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling_contrastive\contrastive_seqlr{}_gralr{}_d{}_seed{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.seed))

    if args.task_type == 'Regression':
        predict = predicts.squeeze(1).cpu().detach().numpy().tolist()
        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        error = []
        for i in range(len(labels)):
            error.append((labels[i] - predict[i]) * (args.label_max - args.label_min))
        absError = []
        squaredError = []
        for val in error:
            absError.append(abs(val))
            squaredError.append(val * val)
        MSE = sum(squaredError) / len(squaredError)
        MAE = sum(absError) / len(absError)

        from sklearn.metrics import r2_score
        R2 = r2_score(test_label.cpu(), predicts.cpu().detach())

        df_test_save['MSE'] = squaredError
        df_test_save['MAE'] = absError
        df_test_save['MSE_ave'] = MSE
        df_test_save['MAE_ave'] = MAE
        df_test_save['R2'] = R2

        print('==================test metric=======================')
        # print(f'MSE:{squaredError}')
        # print(f'MAE:{absError}')
        print(f'MSE_ave:{MSE}')
        print(f'MAE_ave:{MAE}')
        print(f'R2:{R2}')
        print('====================================================')
        # if not os.path.exists(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset)):
        #     os.mkdir(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset))
        # df_test_save.to_csv(r'E:\RepCon-main\results\{}\co-modeling contrastive\contrastive_seqlr{}_gralr{}_d{}_nce{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.nce_weight))

if __name__ == '__main__':
    main()

