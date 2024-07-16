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
from loss import info_nce_loss, get_kl_loss, get_graph_seq_kl_loss
import argparse
import warnings
import os
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR
import matplotlib.pyplot as plt

from gnn_dgl import GMoE

from tqdm import tqdm

parser = argparse.ArgumentParser()

# Model & Training Settings
parser.add_argument('--dataset', type=str, default='AMP',
                    choices=['AMP','AP','PepDB','RT'])
parser.add_argument('--mode', type=str, default='train',
                    choices=['train','test'])
parser.add_argument('--seed', type=int, default=5, 
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--seq_lr', type=float, default=1e-4,
                    help='Initial learning rate.')
parser.add_argument('--graph_lr', type=float, default=5e-4,
                    help='Initial learning rate.')

# Sequential Model Parameters
parser.add_argument('--src_vocab_size_seq', type=int, default=21,
                    help='20 natural amino acids plus padding token')
parser.add_argument('--model_seq', type=str, default='Transformer')
parser.add_argument('--d_model', type=int, default=64,
                    help='Output dim of self-attention layers')
parser.add_argument('--fc_layers', type=int, default=4,
                    help='Predictor MLP layers')
parser.add_argument('--d_ff', type=int, default=2048, 
                    help='Hidden dim of FFN')
parser.add_argument('--d_k', type=int, default=64, 
                    help='Hidden dim of K and Q')
parser.add_argument('--d_v', type=int, default=64, 
                    help='Hidden dim of V')
parser.add_argument('--n_layers', type=int, default=6, 
                    help='Num of self-attention layers')
parser.add_argument('--n_heads', type=int, default=8, 
                    help='Num of head for multi-head attention')

# Graphical Model Parameters
parser.add_argument('--model_graph', type=str, default='GraphSAGE')
parser.add_argument('--GraphSAGE_aggregator', type=str, default='lstm',
                    help='Aggregation function of GraphSAGE')
parser.add_argument('--conv_layers', type=int, default=2)
parser.add_argument('--d_graph', type=int, default=256)
parser.add_argument('--src_vocab_size_graph', type=int, default=15,
                    help='15 types of beads') 
parser.add_argument('--graph_layer', type=int, default=2)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--drop_ratio', type=float, default=0.5)
parser.add_argument('--num_experts', '-n', type=int, default=5,
                    help='total number of experts in GCN-MoE')
parser.add_argument('--num_experts_1hop', '--n1', type=int, default=5,
                    help='number of hop-1 experts in GCN-MoE. Only used when --hop>1.')
parser.add_argument('-k', type=int, default=2,
                    help='selected number of experts in GCN-MoE')
parser.add_argument('--coef', type=float, default=1,
                    help='loss coefficient for load balancing loss in sparse MoE training')



# InfoNCE Params
parser.add_argument('--n_views', type=int, default=2, 
                    help='Num of positive pairs')
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--nce_weight', type=float, default=1e-4)
parser.add_argument('--kl_weight', type=float, default=1e-4)
parser.add_argument('--seq_graph_weight', type=float, default=0.5)
parser.add_argument('--use_infonce_loss', type=str, default=False, help='Contrastive loss')

# Use VAE structure
parser.add_argument('--model_mode', type=str, default='moe',help='chose seqenece model:"vae"、"moe、transformer"')

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
    
    # Process the labels
    if args.task_type == 'Classification':
        train_label = train_label['labels'].long()
        valid_label = valid_label['labels'].long()
        test_label = test_label['labels'].long()
        args.output_layer = int(train_label.max())-int(train_label.min())+1
    elif args.task_type == 'Regression':
        train_label = train_label['labels'].float()
        valid_label = valid_label['labels'].float()
        test_label = test_label['labels'].float()
        args.output_layer = 1
        args.label_max = train_label.max().item()
        args.label_min = train_label.min().item()
        # Normalize the regression labels by min-max
        train_label = (train_label - args.label_min) / (args.label_max-args.label_min)
        valid_label = (valid_label - args.label_min) / (args.label_max-args.label_min)
        test_label = (test_label - args.label_min) / (args.label_max-args.label_min)
    
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
    train_dataset = MyDataSet_TDVAE(train_dataset_graph, train_dataset_seq, train_label, train_dataset_pad_len)
    train_loader = Data.DataLoader(train_dataset, args.batch_size, True,
                                   collate_fn=collate)
    valid_dataset = MyDataSet_TDVAE(valid_dataset_graph, valid_dataset_seq, valid_label, valid_dataset_pad_len)
    valid_loader = Data.DataLoader(valid_dataset, args.batch_size, False,
                                   collate_fn=collate)
    test_dataset = MyDataSet_TDVAE(test_dataset_graph, test_dataset_seq, test_label, test_dataset_pad_len)
    test_loader = Data.DataLoader(test_dataset, args.batch_size, False,
                                  collate_fn=collate)
    
    #---------------- Train Phase ----------------#

    if args.mode == 'train':

        # Initialize the models and the optimizers
        if args.model_mode == 'vae':
            Seq_model = vTransformer(args).to(args.device)
        elif args.model_mode == 'moe':
            Seq_model = SwitchTransformer(
                num_tokens=args.src_vocab_size_seq, dim=args.d_model,
                heads=args.n_heads, dim_head=args.d_v,task_type = args.task_type
            ).to(args.device)
        elif args.model_mode == 'transformer':
            Seq_model = Transformer(args).to(args.device)

        Seq_optimizer = optim.Adam(Seq_model.parameters(), lr=args.seq_lr)
        Seq_scheduler = CosineAnnealingLR(Seq_optimizer, T_max=10, eta_min=1e-5)
        # Seq_scheduler = StepLR(Seq_optimizer, step_size=10, gamma=0.8)
        if args.model_mode == 'vae' :
            Graph_model = VGAEModel(args).to(args.device)
        elif args.model_mode == 'moe':
            # Graph_model = GNNs_MoE(args).to(args.device)
            Graph_model = GMoE(args=args, gnn_type='sage', num_tasks=args.output_layer, num_layer=args.graph_layer, emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio, virtual_node=False,
                        moe='sparse', num_experts=args.num_experts, k=args.k, coef=args.coef,
                        num_experts_1hop=args.num_experts_1hop).to(device)

        elif args.model_mode == 'transformer':
            Graph_model = GNNs(args).to(args.device)
        Graph_optimizer = optim.Adam(Graph_model.parameters(), lr=args.graph_lr)
        Graph_scheduler = CosineAnnealingLR(Graph_optimizer, T_max=10, eta_min=1e-5)

        # Initialize MSE loss and CE loss for Infonce
        loss_mse = torch.nn.MSELoss().to(args.device)
        loss_infonce = torch.nn.CrossEntropyLoss().to(args.device)

        # Initialize the validation performance
        valid_mse_saved = 1e5
        valid_graph_mse_saved = 1e5
        valid_acc_saved = 0

        # Plot validation Mse dict
        valid_mse_dict = {}
        valid_acc_dict = {}

        # Train models by epoches of training data
        for epoch in tqdm(range(args.epochs)):
            # Seq_model.train()
            Graph_model.train()
            # Extract a batch of data
            for graphs,sequences,labels,pad_len in train_loader:
                graphs = graphs.to(args.device)
                sequences = sequences.to(args.device)
                labels = labels.to(args.device)
                
                # Forward Models
                if args.model_mode == 'vae' :
                    seq_outputs, seq_hid, mean1, log_std1 = Seq_model(sequences)
                    # graph_outputs, graph_hid, mean2, log_std2 = Graph_model(graphs)
                elif args.model_mode == 'moe':
                    # seq_outputs, seq_hid, aux_loss = Seq_model(sequences)
                    # graph_outputs, graph_hid = Graph_model(graphs)
                    graph_outputs, _ = Graph_model(graphs) # (batchsize,1)
                    load_balance_loss = Graph_model.gnn_node.load_balance_loss
                else:
                    # seq_outputs:[512,1] \ seq_hid:[512,64]
                    seq_outputs, seq_hid = Seq_model(sequences)
                    # graph_outputs, graph_hid = Graph_model(graphs)

                # Supervised loss
                if args.task_type == 'Classification':
                    # loss_seq = F.nll_loss(seq_outputs, labels)
                    loss_graph = F.nll_loss(graph_outputs, labels)
                elif args.task_type == 'Regression':
                    loss_seq = loss_mse(seq_outputs, labels)
                    # loss_graph = loss_mse(graph_outputs, labels)

                # Unsupervised InfoNCE loss
                if args.use_infonce_loss:
                    hid_pairs = torch.cat([seq_hid, graph_hid], 0) # (1024,64)
                    logits, cont_labels = info_nce_loss(args,hid_pairs) # logits:(1024,1023), cont_labels:(1024,)
                    l_infonce = args.nce_weight*loss_infonce(logits, cont_labels)

                if args.model_mode == 'vae':
                    Seq_kl_loss = get_kl_loss(seq_outputs, mean1, log_std1)
                    # Graph_kl_loss = get_kl_loss(graph_outputs, mean2, log_std2)

                    # Overall training loss
                    # loss = loss_seq + loss_graph - args.kl_weight * \
                    #        (args.seq_graph_weight * Seq_kl_loss + (1 - args.seq_graph_weight) * Graph_kl_loss)

                    loss = loss_seq - args.kl_weight * Seq_kl_loss

                elif args.model_mode == 'moe':
                    # loss = loss_seq + loss_graph + l_infonce
                    # loss = loss_seq + loss_graph + l_infonce + aux_loss
                    # loss = loss_seq + loss_graph + aux_loss + load_balance_loss # seq + graph
                    # loss = loss_seq + loss_graph + aux_loss + load_balance_loss + l_infonce
                    loss = loss_graph + load_balance_loss # only graph
                    # loss = loss_graph
                    # loss = loss_seq + aux_loss # only seq
                elif args.model_mode == 'transformer':
                    # loss = loss_seq + loss_graph + l_infonce
                    loss = loss_seq
                    # loss = loss_graph


                # Update model parameters
                # Seq_optimizer.zero_grad()
                Graph_optimizer.zero_grad()
                loss.backward()
                # Seq_optimizer.step()
                Graph_optimizer.step()
                # Seq_scheduler.step()
                Graph_scheduler.step()

            # Validation
            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    # Seq_model.eval()
                    Graph_model.eval()

                    # Derive prediction results for all samples in validation set
                    predicts = []
                    gra_predicts = []
                    for graphs,sequences,labels,pad_len in valid_loader:
                        if args.model_mode == 'vae':
                            outputs, _, _, _ = Seq_model(sequences)
                            gra_outputs, _, _, _ = Graph_model(graphs)
                        elif args.model_mode == 'moe':
                            # outputs, _, _ = Seq_model(sequences)
                            gra_outputs, _ = Graph_model(graphs)
                        elif args.model_mode == 'transformer':
                            outputs, _, = Seq_model(sequences)
                            # gra_outputs, _ = Graph_model(graphs)
                        # predicts = predicts + outputs.cpu().detach().numpy().tolist()
                        gra_predicts = gra_predicts + gra_outputs.cpu().detach().numpy().tolist()
                    # predicts = torch.tensor(predicts)
                    gra_predicts = torch.tensor(gra_predicts)
                    
                    # Print Acc. or MSE on validation set
                    if args.task_type == 'Classification':
                        valid_acc = accuracy(gra_predicts, valid_label).item()
                        if valid_acc_saved < valid_acc:
                            valid_acc_saved = valid_acc
                            valid_acc_dict[int(epoch)] = float(valid_acc)
                            print('Epoch:',epoch+1)
                            print('Valid Performance:',valid_acc)
                        if args.model_mode == 'vae':
                            torch.save(Seq_model.state_dict(), args.model_path + '{}_Cont_Seq_VAE.pt'.format(args.dataset))
                        elif args.model_mode == 'moe':
                            # torch.save(Seq_model.state_dict(), args.model_path + '{}_Cont_Seq_MoE.pt'.format(args.dataset))
                            torch.save(Graph_model.state_dict(),args.model_path + '{}_Cont_Graph_MoE.pt'.format(args.dataset))
                        elif args.model_mode == 'transformer':
                            torch.save(Seq_model.state_dict(),args.model_path+'{}_Cont_Seq.pt'.format(args.dataset))

                    elif args.task_type == 'Regression':
                        valid_Seq_mse = loss_mse(predicts*(args.label_max-args.label_min),valid_label*(args.label_max-args.label_min)).item()
                        # valid_graph_mse = loss_mse(gra_predicts*(args.label_max-args.label_min),valid_label*(args.label_max-args.label_min)).item()
                        if valid_mse_saved > valid_Seq_mse:
                            valid_mse_saved = valid_Seq_mse
                            valid_mse_dict[int(epoch)] = float(valid_Seq_mse)
                            print('Epoch:',epoch+1)
                            print('Valid Sequence Performance:',valid_Seq_mse)
                            # print('Valid Graph Performance:',valid_graph_mse)
                            if args.model_mode == 'vae':
                                torch.save(Seq_model.state_dict(),
                                           args.model_path + '{}_Cont_Seq_VAE.pt'.format(args.dataset))
                            elif args.model_mode == 'moe':
                                # # save seq
                                # torch.save(Seq_model.state_dict(),
                                #            args.model_path + '{}_Cont_Seq_MoE.pt'.format(args.dataset))
                                # save graph
                                torch.save(Graph_model.state_dict(),
                                           args.model_path + '\{}_Cont_Graph_MoE.pt'.format(args.dataset))
                            elif args.model_mode == 'transformer':
                                torch.save(Seq_model.state_dict(),args.model_path + '{}_Cont_Seq.pt'.format(args.dataset))
                                # torch.save(Graph_model.state_dict(),                                    args.model_path + '{}_Cont_Graph.pt'.format(args.dataset))


                        # if valid_graph_mse_saved > valid_graph_mse:
                        #     valid_graph_mse_saved = valid_graph_mse
                        #     valid_mse_dict[int(epoch)] = float(valid_graph_mse)
                        #     print('Epoch:',epoch+1)
                        #     print('Valid Sequence Performance:',valid_Seq_mse)
                        #     print('Valid Graph Performance:',valid_graph_mse)
                        #     if args.model_mode == 'vae':
                        #         torch.save(Seq_model.state_dict(),
                        #                    args.model_path + '{}_Cont_Seq_VAE.pt'.format(args.dataset))
                        #     elif args.model_mode == 'moe':
                        #         # # # save seq
                        #         # torch.save(Seq_model.state_dict(),
                        #         #            args.model_path + '{}_Cont_Seq_MoE.pt'.format(args.dataset))
                        #         # save graph
                        #         torch.save(Graph_model.state_dict(),
                        #                    args.model_path + '{}_Cont_Graph_MoE.pt'.format(args.dataset))
                        #     elif args.model_mode == 'transformer':
                        #         # torch.save(Seq_model.state_dict(),
                        #         #            args.model_path + '{}_Cont_Seq.pt'.format(args.dataset))
                        #         torch.save(Graph_model.state_dict(),args.model_path + '{}_Cont_Graph.pt'.format(args.dataset))

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

    #---------------- Test Phase----------------#
    if args.model_mode == 'vae':
        Seq_model_load = vTransformer(args).to(args.device)
        Seq_model_load.load_state_dict(torch.load(args.model_path + '{}_Cont_Seq_VAE.pt'.format(args.dataset)))
    elif args.model_mode == 'moe':
        # # load Seq model
        # Seq_model_load = SwitchTransformer(num_tokens=args.src_vocab_size_seq,
        #                                    dim=args.d_model,
        #         heads=args.n_heads, dim_head=args.d_v, task_type=args.task_type).to(args.device)
        # Seq_model_load.load_state_dict(torch.load(args.model_path + '{}_Cont_Seq_MoE.pt'.format(args.dataset)))

        # load Graph model
        Graph_model_load = GNNs_MoE(args).to(args.device)
        Graph_model_load = GMoE(args, gnn_type='sage', num_tasks=args.output_layer, num_layer=args.graph_layer, emb_dim=args.emb_dim,drop_ratio=args.drop_ratio, virtual_node=False,moe='sparse', num_experts=args.num_experts, k=args.k, coef=args.coef,num_experts_1hop=args.num_experts_1hop).to(device)
        Graph_model_load.load_state_dict(torch.load(args.model_path + '{}_Cont_Graph_MoE.pt'.format(args.dataset)))

    elif args.model_mode == 'transformer':
        Seq_model_load = Transformer(args).to(args.device)
        Seq_model_load.load_state_dict(torch.load(args.model_path+'{}_Cont_Seq.pt'.format(args.dataset)))
        # Graph_model_load = GNNs(args).to(args.device)
        # Graph_model_load.load_state_dict(torch.load(args.model_path + '{}_Cont_Graph.pt'.format(args.dataset)))

    # Seq_model_load.eval()
    Graph_model_load.eval()
    
    with torch.no_grad():
        Seq_predicts = []
        graph_predicts = []
        for graphs,sequences,labels,pad_len in test_loader:
            if args.model_mode == 'vae':
                outputs,_,_,_ = Seq_model_load(sequences)
            elif args.model_mode == 'moe':
                # Seq model inference
                # Seq_outputs, _, _ = Seq_model_load(sequences)

                # Graph model inference
                Graph_outputs, _ = Graph_model_load(graphs)
            elif args.model_mode == 'transformer':
                Seq_outputs,_ = Seq_model_load(sequences)
                # outputs, _ = Graph_model_load(graphs)
            # Seq_predicts = Seq_predicts + Seq_outputs.cpu().detach().numpy().tolist()
            graph_predicts = graph_predicts + Graph_outputs.cpu().detach().numpy().tolist()
        # Seq_predicts = torch.tensor(Seq_predicts)
        graph_predicts = torch.tensor(graph_predicts)

    df_test = pd.read_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/{}/test.csv'.format(args.dataset))

    if args.task_type == 'Classification':
        # predict_tensor = Seq_predicts.max(1)[1].type_as(test_label)
        # predict = predict_tensor.cpu().detach().numpy().tolist()

        predict_tensor = graph_predicts.max(1)[1].type_as(test_label)
        predict = predict_tensor.cpu().detach().numpy().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        test_acc = accuracy(graph_predicts,test_label).item()
        df_test_save['Acc'] = test_acc

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

        # if not os.path.exists('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling_contrastive'.format(args.dataset)):
        #     os.mkdir('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling_contrastive'.format(args.dataset))
        # df_test_save.to_csv('/root/autodl-tmp/RepCon2/RepCon-main/seq_dataset/results/{}/co-modeling_contrastive\contrastive_seqlr{}_gralr{}_d{}_seed{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.seed))

    if args.task_type == 'Regression':
        Seq_predict = Seq_predicts.squeeze(1).cpu().detach().numpy().tolist()
        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['Seq_predict'] = Seq_predict
        df_test_save['label'] = labels
        error = []
        for i in range(len(labels)):
            error.append((labels[i]-Seq_predict[i])*(args.label_max-args.label_min))
        absError = []
        squaredError = []
        for val in error:
            absError.append(abs(val))
            squaredError.append(val*val)
        MSE = sum(squaredError)/len(squaredError)
        MAE = sum(absError)/len(absError)

        from sklearn.metrics import r2_score
        R2 = r2_score(test_label.cpu(),Seq_predicts.cpu().detach())
        
        df_test_save['MSE'] = squaredError
        df_test_save['MAE'] = absError
        df_test_save['MSE_ave'] = MSE
        df_test_save['MAE_ave'] = MAE
        df_test_save['R2'] = R2

        print('==================test metric=======================')
        # print(f'MSE:{squaredError}')
        # print(f'MAE:{absError}')
        print('Sequence model predict')
        print(f'MSE_ave:{MSE}')
        print(f'MAE_ave:{MAE}')
        print(f'R2:{R2}')
        print('====================================================')

        # # -----------graph--------------------------------

        # graph_predict = graph_predicts.squeeze(1).cpu().detach().numpy().tolist()
        # df_test_save = pd.DataFrame()
        # labels = test_label.squeeze(1).tolist()
        # df_test_save['feature'] = df_test['Feature']
        # df_test_save['graph_predict'] = graph_predict
        # df_test_save['label'] = labels
        # error = []
        # for i in range(len(labels)):
        #     error.append((labels[i]-graph_predict[i])*(args.label_max-args.label_min))
        # absError = []
        # squaredError = []
        # for val in error:
        #     absError.append(abs(val))
        #     squaredError.append(val*val)
        # MSE = sum(squaredError)/len(squaredError)
        # MAE = sum(absError)/len(absError)

        # from sklearn.metrics import r2_score
        # R2 = r2_score(test_label.cpu(),graph_predicts.cpu().detach())
        
        # df_test_save['MSE'] = squaredError
        # df_test_save['MAE'] = absError
        # df_test_save['MSE_ave'] = MSE
        # df_test_save['MAE_ave'] = MAE
        # df_test_save['R2'] = R2

        # print('==================test metric=======================')
        # # print(f'MSE:{squaredError}')
        # # print(f'MAE:{absError}')
        # print('Graph model predict')
        # print(f'MSE_ave:{MSE}')
        # print(f'MAE_ave:{MAE}')
        # print(f'R2:{R2}')
        # print('====================================================')

        # if not os.path.exists(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset)):
        #     os.mkdir(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset))
        # df_test_save.to_csv(r'E:\RepCon-main\results\{}\co-modeling contrastive\contrastive_seqlr{}_gralr{}_d{}_nce{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.nce_weight))

if __name__ == '__main__':
    main()

