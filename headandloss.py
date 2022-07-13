from os import O_TRUNC
from shutil import disk_usage
from numpy import deprecate
import yaml
import torch
import random
from torch import embedding, nn
import torch.nn.functional as F
from typing import Tuple

import logging

logger = logging.getLogger(__name__)

class ArcFace(nn.Module):
    """
    @Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
             https://arxiv.org/abs/1801.07698
    """
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))        
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        # cos(theta)             
        x, weight = F.normalize(x), F.normalize(self.weight)
        cosine = F.linear(x, weight)                

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
                
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_()
                
        return cosine


class UNPG(nn.Module):
    def __init__(self, s, wisk=1.0):
        super(UNPG, self).__init__()
        self.s = s
        self.wisk = wisk
        self.cross_entropy = nn.CrossEntropyLoss()
                         
    def forward(self, cosine, aux_sn, labels):                          
        # aux_sn = aux_sn.unsqueeze(0)                        
        one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
        aux_sn = one * aux_sn        
        cosine = torch.cat([cosine, aux_sn], dim=1)                      
        loss = self.cross_entropy(self.s * cosine, labels)
        return loss


class UPG(nn.Module):
    def __init__(self, s, wisk=1.0):
        super(UPG, self).__init__()
        self.s = s
        self.wisk = wisk
        self.cross_entropy = nn.CrossEntropyLoss()
                         
    def forward(self, cosine, sp_cosine, aux_sn, labels):                          
        # aux_sn = aux_sn.unsqueeze(0)                        
        one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
        aux_sn = one * aux_sn        
        cosine = torch.cat([cosine, aux_sn], dim=1)                      
        loss = self.cross_entropy(self.s * cosine, labels)
        return loss


def box_and_whisker_algorithm(similarities, wisk):        
    l = similarities.size(0)
    sorted_x = torch.sort(input=similarities, descending=False)[0]
    
    lower_quartile = sorted_x[int(0.25 * l)]
    upper_quartile = sorted_x[int(0.75 * l)]
            
    IQR = (upper_quartile - lower_quartile)        
    minimum = lower_quartile - wisk * IQR        
    maximum = upper_quartile + wisk * IQR
    mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
    sn_prime = sorted_x[mask]
    return sn_prime, sorted_x , minimum, maximum


def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return (similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])


def convert_label_to_similarity_for_upg(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    sp_inx = positive_matrix.nonzero(as_tuple=True)

    negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    sn_idx = negative_matrix.nonzero(as_tuple=True)

    sim = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return (sim[positive_matrix], sim[negative_matrix], sp_inx, sn_idx, similarity_matrix)


class HeadAndLoss(nn.Module):
    def __init__(self, in_feature, num_classes, head_name, aux_name, head_zoo='config/head.zoo.yaml'):
        super(HeadAndLoss, self).__init__()
        
        # self.criterion = criterion
        self.head_name = head_name
        self.aux_name = aux_name
                
        # Head config file        
        with open(head_zoo) as f:
            head_zoo = yaml.load(f, Loader=yaml.FullLoader)                
        opt = head_zoo.get(head_name)
        # assert opt != None                            
        self.head_cfg = {head_name : opt}                
        logger.info("Head config file {}".format(self.head_name))
        logger.info(str(self.head_cfg))    
        
        aux_opt = head_zoo.get('unpg') if 'unpg' in aux_name else None        
        self.aux_cfg = {aux_name : aux_opt}        
        logger.info("Aux config file {}".format(self.aux_name))
        logger.info(str(self.aux_cfg))    
        
        if head_name == 'arcface':
            self.head = ArcFace(in_feature=in_feature, out_feature=num_classes, s=opt['s'], m=opt['m'])                               
        
        if 'unpg' in aux_name:
            self.aux = UNPG(s=opt['s'], wisk=aux_opt['wisk'])                   
            self.wisk = aux_opt['wisk']
        elif 'upg' in aux_name:
            self.wisk = 2.0

        self.s = opt['s']
        self.m = opt['m']
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.wisk = 2.0                
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, deep_features, labels):                 
        metas = {}
        
        deep_features = self.dropout(deep_features)

        if self.head_name in ['arcface']:            
            cosine = self.head(deep_features, labels) 
                                                    
        norm_x = F.normalize(deep_features)
        
        if self.aux_name in ['upg']:    
            sp, sn, sp_idx, sn_idx, sim = convert_label_to_similarity_for_upg(norm_x, labels)
        else:
            sp, sn = convert_label_to_similarity(norm_x, labels)

        metas['spmin'] = sp.min()
        metas['snmax'] = sn.max()

        if 'unpg' in self.aux_name:
            aux_sn = []                                            
            sn_prime = box_and_whisker_algorithm(sn, wisk=self.wisk)    
            # sn_prime = sn
            aux_sn.append(sn_prime)                       
            aux_sn = torch.cat(aux_sn, dim=0)                                   
            loss = self.aux(cosine, aux_sn, labels) 

        """ elif 'upg' in self.aux_name:
            sn_prime = box_and_whisker_algorithm(sn, wisk=self.wisk)
            sp_row, sp_col = sp_idx
            
            new_sim = sim[(sp_row, sp_col)]        

            sp_cosine = cosine[sp_row]
            sp_label = labels[sp_row]           
            sp_one_hot = torch.zeros_like(sp_cosine)
            sp_one_hot.scatter_(1, sp_label.view(-1, 1), 1.0)

            new_sim.acos_()
            new_sim += self.m
            new_sim.cos_()

            sp_cosine[sp_one_hot.bool()] = new_sim

            cosine = torch.cat([cosine, sp_cosine], dim=0) 
            
            one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
            logit_n = one * sn_prime

            new_label = torch.cat([labels, sp_label], dim=0)
            cosine = torch.cat([cosine, logit_n], dim=1)

            loss = self.cross_entropy_loss(self.s * cosine, new_label) """

        if 'upg_ff_fw' in self.aux_name:
            sp, sorted_sp, minimum, maximum = box_and_whisker_algorithm(sp, wisk=2.0)            
            # sp = sorted_sp[sorted_sp >= minimum]
            _, sorted_sn, min_sn, max_sn = box_and_whisker_algorithm(sn, wisk=self.wisk)
            sn_prime = sorted_sn[sorted_sn <= max_sn]
            # sn_prime = sn
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            
            spw = cosine[one_hot.bool()]
            snw = cosine[~one_hot.bool()]

            sp.acos_()
            sp += self.m
            sp.cos_()

            logit_n = torch.cat((snw, sn_prime), dim=0)
            logit_p = torch.cat((spw, sp), dim=0)
            
            one = torch.ones(logit_p.size(0), device=logit_p.device).unsqueeze(1)
            logit_n = one * logit_n

            re_cosine = torch.cat((logit_p.unsqueeze(1), logit_n), dim=1)
            re_labels = torch.zeros_like(logit_p).long()
            
            loss = self.cross_entropy_loss(self.s * re_cosine, re_labels)


        return loss, metas
        

 