# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)

def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

class Args:
    train = 'train'
    valid = 'valid'
    test = None
    batch_size = 256  
    optim = 'bert'
    lr = 1e-4
    epochs = 10
    dropout = 0.1
    seed = 9595
    output = '.snap/test'
    fast = False
    tiny = False
    tqdm = False

    # Model Loading
    load = None
    load_lxmert = None
    load_lxmert_qa = None
    from_scratch = None


    # Optimization
    mce_loss=False

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    llayers=9
    xlayers=5
    rlayers=5

    # LXMERT Pre-training Config
    taskMatched =False
    taskMaskLM =False
    taskObjPredict =False
    taskQA =False
    visualLosses ='obj,attr,feat'
    qaSets =None
    wordMaskRate =0.15
    objMaskRate =0.15

    # Training configuration
    multiGPU =False
    num_workers =0
    config =None
    save_folder ="test"

    # Invalid parameters just designed to accomodate sgg code
    #config-file =None
    algorithm =None

    def get(self,attr,value):
        return getattr(self,attr) == value

    

args = Args()

# Bind optimizer class.
args.optimizer = get_optimizer(args.optim)


# Set seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Added by harold. Allows additional parameters specified by the json file.
import commentjson
from attrdict import AttrDict
from pprint import pprint

'''
mode:
1 for vqa english model
2 for english pretrain
3 for multilingual (english + german) pretrain
4 for vqa multilingual finetune training
5 for vqa multilingual model
'''

mode = 5

if mode == 1:
    args.config = r'configs/vqa.json'
    args.load = r'snap/vqa'
elif mode == 2:
    args.config = r'configs/pretrain/unsupervised.json'
elif mode == 3:
    args.config = r'configs/pretrain/unsupervised_multilingual.json'
    args.load = None
elif mode == 4:
    args.config = r'configs/vqa_multilingual_finetune.json'
    args.load = r'snap/vqa_multilingual'
elif mode == 5:
    args.config = r'configs/vqa_multilingual.json'
    args.load = r'snap/vqa_multilingual'


args.output = r'snap/vqa_test'
args.test = 'val'

args.weight_disable = True
args.kl_divergence = False
args.multiGPU =False
args.num_workers =0
args.save_folder ="test"

if args.config is not None:
    with open(args.config) as f:
        config_json = commentjson.load(f)
    dict_args = vars(args)
    dict_args.update(config_json)  # Update with overwrite
    args = AttrDict(dict_args)


import shutil
import os
output = args.output
if not os.path.exists(output):
    os.mkdir(output)
shutil.copyfile(args.config, os.path.join(output, os.path.basename(args.config)))

if 0:
    # Set up logs
    import sys
    run_log_counter = 0

    while(os.path.exists(args.output + '/run_{}.log'.format(run_log_counter))):
        run_log_counter += 1

    file_log = open(args.output + '/run_{}.log'.format(run_log_counter),'w')  # File where you need to keep the logs
    file_log.write("")
    class Unbuffered:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            file_log.write(data)    # Write the data of stdout here to a text file as well
        def flush(self):
            pass
    sys.stdout = Unbuffered(sys.stdout)
    from pprint import pprint
    pprint(args)

    print("\n\n\n\n")
    with open(args.config) as f:
        print(f.read())
