import numpy as np
import torch
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


def read_test_pair_dataset(test_pair_path):
        pair_data = list()
        with open(test_pair_path) as f:
            data = f.readlines()
        for d in data:
            info = d[:-1].split(' ')
            pair_data.append(info)
                                           
        return pair_data


def code_embedding_extraction(model, test_dataloader, device):
    model.eval()
    extract_info = dict()    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            #get inputs
            code_inputs = batch[0].to(device)  
            attn_mask = batch[1].to(device)
            position_idx = batch[2].to(device)
            labels = batch[3]
            code_path = batch[4]

            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            # code_vec = code_vec[:, 0, :] # take <s> token (equiv. to [CLS])
            code_vec = code_vec.data.cpu()

            for idx in range(len(labels)):
                extract_info[code_path[idx]] = {'code_embedding' : code_vec[idx], 'label' : labels[idx]}
                                
            if (step+1) % 50 == 0 or (step+1) == len(test_dataloader):
                logger.info("Code_embedding extracting ... {}/{}".format(step+1, len(test_dataloader)))
        
    return extract_info


def batch_cosine_similarity(x1, x2):
    """
    ex) x1 size [256, 512], x2 size [256, 512]
    similarity size = [256, 1]
    """    
    x1 = F.normalize(x1).unsqueeze(1)
    x2 = F.normalize(x2).unsqueeze(1)    
    x2t = torch.transpose(x2, 1, 2)
    similarity = torch.bmm(x1, x2t).squeeze()
    return similarity


def computing_sim_from_df(extracted_df_dict, test_pairs_txt, device, split_batch_size=1024):          
    test_pairs_lst = read_test_pair_dataset(test_pairs_txt)
    id1_deepfeatures = []
    id2_deepfeatures = []
    labels = []
    similarities = []

    for i, (id1, id2, label) in enumerate(test_pairs_lst):
        df1 = extracted_df_dict[id1]['code_embedding']
        df2 = extracted_df_dict[id2]['code_embedding']
        
        id1_deepfeatures.append(df1)
        id2_deepfeatures.append(df2)
        labels.append(int(label))
            
    id1_deepfeatures = torch.stack(id1_deepfeatures, dim=0)
    id2_deepfeatures = torch.stack(id2_deepfeatures, dim=0)
    
    split_df1 = torch.split(id1_deepfeatures, split_batch_size)
    split_df2 = torch.split(id2_deepfeatures, split_batch_size)
    
    similarities = []
    for i, (df1, df2) in enumerate(zip(split_df1, split_df2)):
        df1 = df1.to(device)
        df2 = df2.to(device)
        sim = batch_cosine_similarity(df1,df2)
        similarities.extend(sim.data.cpu().tolist())
    
    return similarities, labels


def torch_cal_accuracy(y_score, y_true, freq=10000):    
    best_acc = 0
    best_th = 0
    th_array = np.linspace(0, 1, 100000)
    # for i in range(len(y_score)):
    for i, th in enumerate(th_array):
        # th = y_score[i]
        y_test = (y_score >= th).long()
        acc = torch.mean((y_test == y_true).float())
        if acc > best_acc:
            best_acc = acc
            best_th = th
        if (i+1) % freq == 0 or (i+1) == len(y_score):
            logger.info('Progress {}/{}'.format((i+1),len(th_array)))
    return best_acc, best_th    


def verification(similarities, labels, device='cpu', metric='roc'):
    assert metric in ['best_th', 'roc']
    
    if metric == 'best_th':
        similarities = torch.Tensor(similarities).to(device)
        labels = torch.Tensor(labels).to(device)    
        acc, th = torch_cal_accuracy(similarities, labels)   
        acc, th = float(acc.data.cpu()), float(th)   
        # logger.info('cosine verification accuracy: {} threshold: {}\n\n'.format(acc, th))    

    elif metric == 'roc':
        # 추후 필요시 구현
        raise NotImplementedError

    return acc, th


def code1v1verification(model, test_dataloder, device, test_pair_txt='config/valid_pair_10.txt'):    
    extracted_df_dict = code_embedding_extraction(model, test_dataloder, device)   
    # torch.save(extracted_df_dict, 'code_embedding.pth')
    # extracted_df_dict = torch.load('code_embedding.pth')
    logger.info("Test txt file: {}".format(test_pair_txt))    
    similarities, labels = computing_sim_from_df(extracted_df_dict, test_pair_txt, device)    
    acc, th = verification(similarities, labels, device, metric='best_th')
    return acc, th