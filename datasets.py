import os
import pandas as pd
import pickle
import logging
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset


from parser.DFG import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from tqdm import tqdm

logger = logging.getLogger(__name__)


dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser


class ClassRandomSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_size, samples_per_class, image_dict, image_list):
        self.image_dict = image_dict
        self.image_list = image_list

        self.classes = list(self.image_dict.keys())
                
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.sampler_length = len(image_list) // batch_size
        assert self.batch_size % self.samples_per_class == 0, '#Samples per class must divide batchsize!'

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            draws = self.batch_size // self.samples_per_class
            
            for _ in range(draws):
                class_key = random.choice(self.classes)
                class_idx_lst = []
                for _ in range(self.samples_per_class):
                    # class_idx_lst.append(random.choice(self.image_dict[class_key])[-1])        
                    class_idx_lst.append(random.choice(self.image_dict[class_key])[-1])
                subset.extend(class_idx_lst)                                          

            yield subset

    def __len__(self):
        return self.sampler_length


#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


class InputFeaturesUnixcoder(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 #nl_tokens,
                 # nl_ids,
                 # url,
                 label,
                 problem_number,
                 code_path


    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        # self.nl_tokens = nl_tokens
        # self.nl_ids = nl_ids
        # self.url=url
        self.label = label
        self.problem_number = problem_number
        self.code_path = code_path


def convert_examples_to_features_by_unixcoder(data, label, tokenizer, args):
    """convert examples to token ids"""
    code = tokenizer.tokenize(data)
    code_tokens = code[:args.block_size-4]
    code_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
                
    return InputFeaturesUnixcoder(code_tokens, code_ids, label)


def convert_examples_to_features(item):
    data,tokenizer,args, label, problem_number, code_path = item
    #code
    parser=parsers[args.lang]
    #extract data flow
    code_tokens,dfg=extract_dataflow(data,parser,args.lang)        
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]

    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))        
    code_tokens=[y for x in code_tokens for y in x]  

    #truncating
    # code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]        
    if args.reverse_tokens:
        code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)]
        if len(code_tokens) > (512 - 3):
            code_tokens=code_tokens[len(code_tokens) - (512-3):]         
            # print(len(code_tokens))         
    else:
        code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:512-3]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]    
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length   
    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
        
    #nl
    # nl=' '.join(data['docstring_tokens'])
    # nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    # nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    # nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    # padding_length = args.nl_length - len(nl_ids)
    # nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    # return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,data['url'])
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg, label, problem_number, code_path)


def preprocess_script(script):
    '''
    간단한 전처리 함수
    주석 -> 삭제
    '    '-> tab 변환
    다중 개행 -> 한 번으로 변환
    '''
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
        #    if line.lstrip().startswith('#'):
        #        continue
        #    line = line.rstrip()
        #    if '#' in line:
        #        line = line[:line.index('#')]
        #    line = line.replace('\n','')
        #    line = line.replace('    ','\t')
        #    if line == '':
        #        continue
            preproc_lines.append(line)
        preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, code_folder=None, pool=None, mode='train'):
        self.valid_list = ['problem056', 'problem122', 'problem037', 
        'problem033', 'problem102', 'problem289', 'problem257', 
        'problem047', 'problem178', 'problem111']

        self.args=args
        self.reverse_tokens = args.reverse_tokens
        logger.info("reverse_tokens: {}".format(str(self.reverse_tokens)))

        problem_folders = os.listdir(code_folder)
        if mode == 'train':
            problem_folders = list(set(problem_folders) - set(self.valid_list))            
        else:
            problem_folders = self.valid_list
        
        if self.reverse_tokens: 
            strings='reverse_tokens'
            cache_file='config/preproc_data_512+128_{}_{}.pkl'.format(strings, mode)
        else:
            cache_file='config/preproc_data_512+128_revise_{}.pkl'.format(mode)
                
        if os.path.exists(cache_file) and mode =='train':
            self.examples=pickle.load(open(cache_file,'rb'))        
        else:
            preproc_scripts = []
            for problem_folder in problem_folders:
                scripts = os.listdir(os.path.join(code_folder,problem_folder))                                              
                problem_number = scripts[0].split('_')[0]
                label = int(problem_number[-3:]) - 1                               
                for script in scripts:
                    code_path = os.path.join(problem_folder, script)                                  
                    script_file = os.path.join(code_folder,problem_folder,script)
                    data = preprocess_script(script_file)                       
                    preproc_scripts.append((data, tokenizer, args, label, problem_number, code_path))                
                                        
            self.examples=pool.map(convert_examples_to_features, tqdm(preproc_scripts,total=len(preproc_scripts)))
            
            if mode == 'train':
                pickle.dump(self.examples,open(cache_file,'wb'))
            
            
        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
            logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
            logger.info("position_idx: {}".format(example.position_idx))
            logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
            logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
            # logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
            # logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))      
            logger.info("label: {}".format(example.label))
            logger.info("problem_number: {}".format(example.problem_number))
            logger.info("code_path: {}".format(example.code_path))

        self.example_per_class_dict = {}    
        count = 0
        for example in self.examples:            
            label = example.label                        
            
            if self.example_per_class_dict.get(label) is None:
                self.example_per_class_dict[label] = []
                self.example_per_class_dict[label].append([example, count])
                # print([example, count])
                count += 1
            else:
                self.example_per_class_dict[label].append([example, count])
                # print([example, count])
                count += 1
                 
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                            self.args.code_length+self.args.data_flow_length),dtype=np.bool)
        #calculate begin index of node and max length of input        
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
               
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].label),
              self.examples[item].code_path
              )



def preprocess_script_for_dacon_submit(script):
    '''
    간단한 전처리 함수
    주석 -> 삭제
    '    '-> tab 변환
    다중 개행 -> 한 번으로 변환
    '''
    lines = script.split('\n')        
    preproc_lines = []
    for line in lines:
    #    if line.lstrip().startswith('#'):
    #        continue
    #    line = line.rstrip()
    #    if '#' in line:
    #        line = line[:line.index('#')]
    #    line = line.replace('\n','')
    #    line = line.replace('    ','\t')
    #    if line == '':
    #        continue
        preproc_lines.append(line)
    preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script


class InputFeaturesForDaconSubmit(object):
    """A single training/test features for a example."""
    # code_tokens_1,code_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1
    def __init__(self,
             code_tokens_1,
             code_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             code_tokens_2,
             code_ids_2,
             position_idx_2,
             dfg_to_code_2,
             dfg_to_dfg_2,
             pair_id

    ):
        #The first code function
        self.code_tokens_1 = code_tokens_1
        self.code_ids_1 = code_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #The second code function
        self.code_tokens_2 = code_tokens_2
        self.code_ids_2 = code_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        
        #label
        self.pair_id=pair_id
        


def convert_examples_to_features_for_dacon_submit(item):
    code1, code2, tokenizer, args, pair_id = item

    cache_dict = {}
    #code
    parser=parsers[args.lang]
    
    for code_index, data in enumerate([code1, code2]):
        #extract data flow
        code_tokens,dfg=extract_dataflow(data, parser,args.lang)        
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]

        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))        
        code_tokens=[y for x in code_tokens for y in x]  

        #truncating
        # code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)] 
        if args.reverse_tokens:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)]
            if len(code_tokens) > (512 - 3):
                code_tokens=code_tokens[len(code_tokens) - (512-3):]         
                # print(len(code_tokens))         
        else:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:512-3]
        # code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:512-3]       
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]    
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.code_length+args.data_flow_length-len(code_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        code_ids+=[tokenizer.pad_token_id]*padding_length   

        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
        
        cache_dict[code_index] = code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg       

    code_tokens_1,code_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1 = cache_dict[0]
    code_tokens_2,code_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2 = cache_dict[1]

    return InputFeaturesForDaconSubmit(code_tokens_1,code_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,
                                       code_tokens_2,code_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                                       pair_id)
    

class DaconSubmitDataset(Dataset):
    def __init__(self, tokenizer, args, test_file='./test.csv', valid=False, pool=None):                
        self.args = args
        self.reverse_tokens = args.reverse_tokens
        test = pd.read_csv(test_file)
        
        if self.reverse_tokens: 
            strings='reverse_tokens'
            cache_file='config/preproc_dacon_submit_data_512+128_{}.pkl'.format(strings)
        else:
            cache_file='config/preproc_dacon_submit_data_512+128.pkl'        
                
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            preproc_scripts = []        
            for i in range(0, len(test)):
                code1 = test.loc[i]['code1']
                code2 = test.loc[i]['code2']
                pair_id = test.loc[i]['pair_id']              
                
                proc_code1 = preprocess_script_for_dacon_submit(code1)
                proc_code2 = preprocess_script_for_dacon_submit(code2)
                preproc_scripts.append((proc_code1, proc_code2, tokenizer, args, pair_id))

            # self.examples=[convert_examples_to_features_for_dacon_submit(x) for x in tqdm(preproc_scripts,total=len(preproc_scripts))]    
            self.examples=pool.map(convert_examples_to_features_for_dacon_submit, tqdm(preproc_scripts,total=len(preproc_scripts)))
            pickle.dump(self.examples,open(cache_file,'wb'))

        if valid:   
            self.examples = random.sample(self.examples, int(len(self.examples)*0.1))
            print(len(self.examples))
        
        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("code_tokens_1: {}".format([x.replace('\u0120','_') for x in example.code_tokens_1]))
            logger.info("code_ids_1: {}".format(' '.join(map(str, example.code_ids_1))))
            logger.info("position_idx_1: {}".format(example.position_idx_1))
            logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code_1))))
            logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg_1))))

            logger.info("code_tokens_2: {}".format([x.replace('\u0120','_') for x in example.code_tokens_2]))
            logger.info("code_ids_2: {}".format(' '.join(map(str, example.code_ids_2))))
            logger.info("position_idx_2: {}".format(example.position_idx_2))
            logger.info("dfg_to_code_2: {}".format(' '.join(map(str, example.dfg_to_code_2))))
            logger.info("dfg_to_dfg_2: {}".format(' '.join(map(str, example.dfg_to_dfg_2))))                
            
            logger.info("pair_id: {}".format(example.pair_id))
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx_1])
        max_length=sum([i!=1 for i in self.examples[item].position_idx_1])
        #sequence can attend to sequence
        attn_mask_1[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids_1):
            if i in [0,2]:
                attn_mask_1[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_1):
            if a<node_index and b<node_index:
                attn_mask_1[idx+node_index,a:b]=True
                attn_mask_1[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx_1):
                    attn_mask_1[idx+node_index,a+node_index]=True  
                    
        #calculate graph-guided masked function
        attn_mask_2= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx_2])
        max_length=sum([i!=1 for i in self.examples[item].position_idx_2])
        #sequence can attend to sequence
        attn_mask_2[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids_2):
            if i in [0,2]:
                attn_mask_2[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_2):
            if a<node_index and b<node_index:
                attn_mask_2[idx+node_index,a:b]=True
                attn_mask_2[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx_2):
                    attn_mask_2[idx+node_index,a+node_index]=True                      
                    
        return (torch.tensor(self.examples[item].code_ids_1),
                torch.tensor(attn_mask_1), 
                torch.tensor(self.examples[item].position_idx_1),                
                torch.tensor(self.examples[item].code_ids_2),
                torch.tensor(attn_mask_2),                 
                torch.tensor(self.examples[item].position_idx_2),                
                torch.tensor(self.examples[item].pair_id))

    