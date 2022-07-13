import errno
import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def histogram_plot(data1, num, save_name):
    
    plt.hist(data1, bins=num, density=False, cumulative=False, label='B',
                    range=(-1, 1), color='b', edgecolor='black', linewidth=1.2, alpha=0.5)
                        
    # plt.title('scatter', pad=10)
    plt.xlabel('Similarity between positive and negative pairs', labelpad=10)
    plt.ylabel('number of pairs', labelpad=10)
    plt.legend(loc='upper right')

    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', direction='in', pad=8, top=True, right=True)

    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)    


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def select_device(device='', batch_size=None, rank=-1):
    # device = 'cpu' or '0' or '0,1,2,3', rank = print only once during distributed parallel
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device {} requested'.format(device)  # check availablity
        
    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size {} not multiple of GPU count {}'.format(batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        
        if rank in [-1, 0]:
            for i in range(0, ng):
                if i == 1:
                    s = ' ' * len(s)
                print("{}CUDA:{} ({}, {}MB)".format(s, i, x[i].name, x[i].total_memory / c))
    else:
        print(f'Using torch {torch.__version__} CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu') 


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise