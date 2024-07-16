import torch
import torch.nn.functional as F


def info_nce_loss(args, features):
    if args == None:
        labels = torch.cat([torch.arange(int(features.shape[0]/2)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')
    
        features = F.normalize(features, dim=1)
    
        similarity_matrix = torch.matmul(features, features.T)
    
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda') # (1024,1024)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (1024,1023)
    
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # (1024,1)
    
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (1024,1022)
    
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')
    
        logits = logits / 0.07
    else:
        labels = torch.cat([torch.arange(int(features.shape[0]/2)) for i in range(args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(args.device)
    
        features = F.normalize(features, dim=1)
    
        similarity_matrix = torch.matmul(features, features.T)
    
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device) # (1024,1024)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (1024,1023)
    
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # (1024,1)
    
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (1024,1022)
    
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)
    
        logits = logits / args.temperature
    return logits, labels

def get_kl_loss(logits, mean, log_std):
    """
    p(f(z/x)||N(0,1))
    :param logits:
    :param mean:
    :param log_std:
    :return: kl loss
    """
    kl_divergence = (
            0.5
            / logits.size(0)
            * (
                    1
                    + 2 * log_std
                    - mean ** 2
                    - torch.exp(log_std) ** 2
            ).sum(1).mean()
    )

    return kl_divergence


def get_graph_seq_kl_loss(mean1,log_std1,mean2,log_std2):
    """
    p(f1(z1/x)||f2(z2/x))
    :param mean1:
    :param log_std1:
    :param mean2:
    :param log_std2:
    :return: kl loss
    """
    term1 = log_std1 - log_std2
    term2 = (torch.exp(log_std1) ** 2 + (mean1 - mean2) ** 2) / (2 * torch.exp(log_std2) ** 2)
    term3 = 0.5

    return (term1 + term2 - term3).sum()