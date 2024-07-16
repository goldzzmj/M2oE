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
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from tqdm import tqdm

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()

# Model & Training Settings
parser.add_argument('--dataset', type=str, default='AP',
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

# InfoNCE Params
parser.add_argument('--n_views', type=int, default=2, 
                    help='Num of positive pairs')
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--nce_weight', type=float, default=1e-4)
parser.add_argument('--kl_weight', type=float, default=1e-4)
parser.add_argument('--seq_graph_weight', type=float, default=0.5)

# Use VAE structure
parser.add_argument('--use_vae', type=str, default=True)

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
if not os.path.exists(r'E:\RepCon-main\methods\co-modeling contrastive\saved_models'):
    os.mkdir(r'E:\RepCon-main\methods\co-modeling contrastive\saved_models')
args.model_path=r'E:\RepCon-main\methods\co-modeling contrastive\saved_models\\'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

# torch.backends.cudnn.enabled = False


def main():

    # Read raw peptide dataset in FASTA form
    df_seq_train = pd.read_csv(r'E:\RepCon-main\seq dataset\{}\train.csv'.format(args.dataset))
    df_seq_valid = pd.read_csv(r'E:\RepCon-main\seq dataset\{}\valid.csv'.format(args.dataset))
    df_seq_test = pd.read_csv(r'E:\RepCon-main\seq dataset\{}\test.csv'.format(args.dataset))

    # Convert dataset from amino acid sequences to molecular graphs
    if not os.path.exists(r'E:\RepCon-main\graph dataset\{}'.format(args.dataset)):
        os.mkdir(r'E:\RepCon-main\graph dataset\{}'.format(args.dataset))
    if not os.path.exists(r'E:\RepCon-main\graph dataset\{}\test.bin'.format(args.dataset)):
        make_dgl_dataset(args)

    # Build data in DGL graph (coarse-grained molecular graphs)
    train_dataset_graph, train_label = dgl.load_graphs(r'E:\RepCon-main\graph dataset\{}\train.bin'.format(args.dataset))
    valid_dataset_graph, valid_label = dgl.load_graphs(r'E:\RepCon-main\graph dataset\{}\valid.bin'.format(args.dataset))
    test_dataset_graph, test_label = dgl.load_graphs(r'E:\RepCon-main\graph dataset\{}\test.bin'.format(args.dataset))


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
    train_dataset_seq, train_dataset_pad_len = make_seq_data_TDVAE(train_feat,args.src_len) # (54159,10)
    valid_dataset_seq, valid_dataset_pad_len = make_seq_data_TDVAE(valid_feat,args.src_len) # (4000,10)
    test_dataset_seq, test_dataset_pad_len = make_seq_data_TDVAE(test_feat,args.src_len) # (4000,10)

    train_dataset_seq, train_dataset_pad_len = train_dataset_seq.to(device), train_dataset_pad_len
    valid_dataset_seq, valid_dataset_pad_len = valid_dataset_seq.to(device), valid_dataset_pad_len
    test_dataset_seq, test_dataset_pad_len = test_dataset_seq.to(device), test_dataset_pad_len


    # Build DataLoaders
    train_dataset = MyDataSet_TDVAE(train_dataset_graph,train_dataset_seq,train_label,train_dataset_pad_len)
    train_loader = Data.DataLoader(train_dataset, args.batch_size, True,
                        collate_fn=collate)
    valid_dataset = MyDataSet_TDVAE(valid_dataset_graph,valid_dataset_seq,valid_label,valid_dataset_pad_len)
    valid_loader = Data.DataLoader(valid_dataset, args.batch_size, False,
                        collate_fn=collate)
    test_dataset = MyDataSet_TDVAE(test_dataset_graph,test_dataset_seq,test_label,test_dataset_pad_len)
    test_loader = Data.DataLoader(test_dataset, args.batch_size, False,
                        collate_fn=collate)
    
    #---------------- Train Phase ----------------#

    if args.mode == 'train':
        # 10,64,64,64
        x_size, processed_x_size, b_size, z_size = args.src_len, args.hidden, args.hidden, args.hidden
        y_size = 1
        Seq_model = TD_VAE(args, y_size, processed_x_size, b_size, z_size).to(args.device)
        Seq_optimizer = optim.Adam(Seq_model.parameters(), lr=args.seq_lr)
        Seq_scheduler = CosineAnnealingLR(Seq_optimizer, T_max=args.epochs // 10, eta_min=1e-5)

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
        for epoch in range(args.epochs):
            print(f'Train epoch:{epoch}')
            Seq_model.train()
            # Extract a batch of data
            for i,(graphs,sequences,labels,pad_len) in enumerate(train_loader):
                # sequences = sequences.to(torch.float32).to(args.device) # (512,10)
                sequences = sequences.to(args.device)  # (512,10)
                labels = labels.to(args.device) # (512,1)

                # self.processed_x: (512,10,64)

                y = Seq_model.forward(sequences, labels,10 - pad_len[i]-1)

                loss = Seq_model.calculate_loss(1,10 - pad_len[i]-1) + loss_mse(y,labels)

                # Update model parameters
                Seq_optimizer.zero_grad()
                loss.backward()
                Seq_optimizer.step()
                Seq_scheduler.step()

            # Validation
            if (epoch+1) % 1 == 0:
                with torch.no_grad():
                    Seq_model.eval()

                    # Derive prediction results for all samples in validation set
                    predicts = []
                    for i,(graphs,sequences,labels,pad_len) in enumerate(valid_loader):
                        outputs = Seq_model.forward(sequences,labels,10 - pad_len[i]-1)
                        # outputs = Seq_model.predict()
                        predicts += outputs.cpu().detach().numpy().tolist()
                    predicts = torch.tensor(predicts)

                    valid_mse = loss_mse(predicts*(args.label_max-args.label_min),valid_label*(args.label_max-args.label_min)).item()
                    if valid_mse_saved > valid_mse:
                        valid_mse_saved = valid_mse
                        valid_mse_dict[int(epoch)] = float(valid_mse)
                        print('best valid mse Epoch:',epoch+1)
                        print('Valid Performance:',valid_mse)
                        torch.save(Seq_model.state_dict(),args.model_path+'{}_Cont_Seq_TDVAE.pt'.format(args.dataset))

        if args.plot_pic:
            x_values = list(valid_mse_dict.keys())
            y_values = list(valid_mse_dict.values())
            print(valid_mse_dict)

            
            plt.figure(figsize=(10, 5))
            plt.plot(x_values, y_values, marker='o', linestyle='-', color='b') 
            plt.title('Epoch vs Val-Mse')
            plt.xlabel('Epoch')
            plt.ylabel('Acc')
            plt.grid(True)
            plt.show()

    # #---------------- Test Phase----------------#
    x_size, processed_x_size, b_size, z_size = args.src_len, args.hidden, args.hidden, args.hidden
    Seq_model_load = TD_VAE(args, x_size, processed_x_size, b_size, z_size).to(args.device)
    Seq_model_load.load_state_dict(torch.load(args.model_path+'{}_Cont_Seq_TDVAE.pt'.format(args.dataset)))
    Seq_model_load.eval()

    with torch.no_grad():
        predicts = []
        for i,(graphs,sequences,labels,pad_len) in enumerate(test_loader):
            outputs = Seq_model_load.forward(sequences,labels,10 - pad_len[i]-1)
            # outputs = Seq_model_load.predict()

            predicts = predicts + outputs.cpu().detach().numpy().tolist()
        predicts = torch.tensor(predicts)

    df_test = pd.read_csv(r'E:\RepCon-main\seq dataset\{}\test.csv'.format(args.dataset))

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

        if args.output_layer > 2:
            from sklearn.metrics import precision_score, recall_score, f1_score
            df_test_save['Precision'] = precision_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['Recall'] = recall_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')
            df_test_save['F1-score'] = f1_score(test_label.cpu(),predict_tensor.cpu().detach(),average='weighted')

        if not os.path.exists(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset)):
            os.mkdir(r'E:\RepCon-main\results\{}\co-modeling contrastive'.format(args.dataset))
        df_test_save.to_csv(r'E:\RepCon-main\results\{}\co-modeling contrastive\contrastive_seqlr{}_gralr{}_d{}_seed{}.csv'.format(args.dataset,args.seq_lr,args.graph_lr,args.d_model,args.seed))

    if args.task_type == 'Regression':
        predict = predicts.squeeze(1).cpu().detach().numpy().tolist()
        df_test_save = pd.DataFrame()
        labels = test_label.squeeze(1).tolist()
        df_test_save['feature'] = df_test['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        error = []
        for i in range(len(labels)):
            error.append((labels[i]-predict[i])*(args.label_max-args.label_min))
        absError = []
        squaredError = []
        for val in error:
            absError.append(abs(val))
            squaredError.append(val*val)
        MSE = sum(squaredError)/len(squaredError)
        MAE = sum(absError)/len(absError)

        from sklearn.metrics import r2_score
        R2 = r2_score(test_label.cpu(),predicts.cpu().detach())

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
