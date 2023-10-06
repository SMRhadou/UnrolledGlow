import argparse
import logging
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from core.unrolling import Unrolled_glow
from core.training import train
from core.optimizee import objective_function
from data import loadDataset
import core.utils as utils


torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
random.seed(100)

def main(args, modelPath): 
    utils.printing(vars(args))

    # GPUs
    device = utils.setGPUs(args)

    # Load data
    trainset, validset, testset = loadDataset([3, args.iSize, args.iSize], args.nBits, Dataset=args.dataset)
    with open(f'./data/{args.dataset}_testset.pkl', 'wb') as ObjFile:
        pickle.dump(testset, ObjFile)

    # Model
    model = Unrolled_glow(args.nLayers, 3, args.depth, args.nLevels)
    if not args.dualGPUs:
        model = nn.DataParallel(model, device_ids=[0]).to(device)
    else:
        model = nn.DataParallel(model, device_ids=[int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"] if i.isdigit()]).to(device)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train(model, trainset, validset, optimizer, objective_function, args, device, modelPath=modelPath)


    print('OK!')

if __name__ == '__main__':

    def f(x, *y, z):
        sum = 0
        for i in y:
            sum += i*x
        return sum + z
    
    t = f(2, 3, 4, z=5)


    FLAGS = argparse.ArgumentParser()
    _, args = utils.parser(FLAGS)
    modelPath = utils.Logging_Saving(args)

    plt.set_loglevel (level = 'warning')
    pil_logger = logging.getLogger('PIL')  
    pil_logger.setLevel(logging.INFO)

    main(args, modelPath)