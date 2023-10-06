import argparse
import logging
import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

def parser(FLAGS):
    FLAGS = argparse.ArgumentParser(description='Unrolled GLOW')
    FLAGS.add_argument('--Trial', type=str, default='try_noise_again', help='Trial')
    # dataset
    FLAGS.add_argument('--dataset', type=str, default='CelebA-HQ', help='dataset')
    FLAGS.add_argument('--nBits', type=int, default=5, help='nBits')
    FLAGS.add_argument('--iSize', type=int, default=64, help='dimension of the images after resizing')
    # Inverse Problem Parameters
    FLAGS.add_argument('--problem', type=str, default='inpainting', help='Supported problems: denoising and inpainting')
    FLAGS.add_argument('--noise_sigma', type=int, default=0.1, help='Noise std')
    FLAGS.add_argument('--painting_size', type=int, default=24, help='painting size')
    # Unrolling Parameters
    FLAGS.add_argument('--nLayers', type=int, default=5, help='nLayers')
    # Glow Parameters
    FLAGS.add_argument('--depth', type=int, default=18, help='depth of the flow')
    FLAGS.add_argument('--nLevels', type=int, default=4, help='nChannels')
    # Training parameters
    FLAGS.add_argument('--batchSize', type=int, default=8, help='batchSize')
    FLAGS.add_argument('--nEpochs', type=int, default=100, help='nEpochs')
    FLAGS.add_argument('--lr', type=float, default=1e-5, help='lr')
    FLAGS.add_argument('--lr_dual', type=float, default=1e-3, help='lr_dual')
    FLAGS.add_argument('--eps', type=float, default=0.05, help='epsilon')
    # Other configurations
    FLAGS.add_argument('--constrained', action="store_true")
    FLAGS.add_argument('--noisyOuts', action="store_true")
    FLAGS.add_argument('--supervised', action="store_true")
    FLAGS.add_argument('--dualGPUs', action="store_true")
    FLAGS.add_argument('--gpuID', type=str, default="0", help='choose a GPU')
    return FLAGS, FLAGS.parse_args()

def setGPUs(args):
    if not args.dualGPUs:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.debug("Number of available GPUs is {} whose IDs are {}".format(torch.cuda.device_count(), os.environ["CUDA_VISIBLE_DEVICES"]))
    return device

def Logging_Saving(args):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logfile = f"./logs/logfile_{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG)

    if not os.path.exists(f"savedModels/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}"):
        os.makedirs(f"savedModels/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}")
    modelPath = f"./savedModels/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}/"

    with open(modelPath+"args.pkl", 'wb') as ObjFile:
        pickle.dump(args, ObjFile)
    return modelPath

def printing(args):
    logging.debug("="*60)
    for i, item in args.items():
        logging.debug("{}: {}".format(i, item))
    logging.debug("="*60)

def scale_image(image, nBits):
    n_bins = 2.0 ** nBits
    image = image * 255
    if nBits < 8:
        image = torch.floor(image / 2 ** (8 - nBits))
    image = image / n_bins #- 0.5
    return image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size(model):
    return np.sum([tensor.nelement() * tensor.element_size() for tensor in model.parameters()])

def prepare_image(img, args=None, clean_img=None, perturbation=False):
    if perturbation:
        clean_img[:,img.shape[1]//2-args.painting_size//2:img.shape[1]//2+args.painting_size//2,
         img.shape[2]//2-args.painting_size//2:img.shape[2]//2+args.painting_size//2] = img[:,
         img.shape[1]//2-args.painting_size//2:img.shape[1]//2+args.painting_size//2,
         img.shape[2]//2-args.painting_size//2:img.shape[2]//2+args.painting_size//2]
        return clean_img.numpy().transpose((1,2,0))
    else:
        return img.numpy().transpose((1,2,0))

def save_layer_image(image, corrupt, outs, epoch, ibatch, args, perturbation=False):
    if not os.path.exists(f"figs/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}"):
        os.makedirs(f"figs/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}")
    idxs = [0, 1, 2, 3, 4, 5, 6]
    plt.figure(figsize=(7, 7))
    k=0
    for idx in idxs:
        plt.subplot(len(idxs), len(outs.keys())+1, k+1)
        plt.imshow(prepare_image(corrupt[idx]))
        plt.axis('off')
        if k == 0:
            plt.title('Input', fontsize=8)
        for i in range(len(outs.keys())-1):
            if perturbation:
                img = prepare_image(outs[i+1][idx].detach().cpu(), args, image[idx].detach().cpu().clone(), perturbation)
            else:
                img = prepare_image(outs[i+1][idx].detach().cpu())
            plt.subplot(len(idxs), len(outs.keys())+1, k+i+2)
            plt.imshow(img)
            plt.axis('off')
            if k == 0 and i == args.nLayers:
                plt.title('Output', fontsize=8)
            if k == 0:
                plt.title(f'Layer {i+1}', fontsize=8)
        plt.subplot(len(idxs), len(outs.keys())+1, k+len(outs.keys())+1)
        plt.imshow(prepare_image(image[idx]))
        plt.axis('off')
        if k == 0:
            plt.title('Target', fontsize=8)
        k += 7
    plt.tight_layout()
    plt.savefig(f'./figs/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}/sample_{epoch}_{ibatch}_{args.constrained}.png')
    plt.close()

def save_perturbation_image(args, perturbation_size):
    if not os.path.exists(f"figs/perturbation_{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}"):
        os.makedirs(f"figs/perturbation_{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}")
    
    plt.figure(figsize=(len(perturbation_size)+1, 6), gridspec_kw = {'wspace':0, 'hspace':0})

    i = 0
    for pSize in perturbation_size:
        with open(f'./results/{args.dataset}_True_{pSize}.pkl', 'rb') as ObjFile:
            _, corrupt, restored, _, _ = pickle.load(ObjFile)
        
        with open(f'./results/{args.dataset}_False_{pSize}.pkl', 'rb') as ObjFile:
            _, corrupt_un, restored_un, _, _ = pickle.load(ObjFile)
        
        idxs = [0, 6, 3]
        k = 0
        for idx in idxs:
            plt.subplot(len(idxs)*2, len(perturbation_size)+1, k+1)
            plt.imshow(prepare_image(corrupt[idx]))
            plt.ylabel('ours', fontsize=8)
            plt.axis('off')
            if k == 0:
                plt.title('Input', fontsize=8)
            img = prepare_image(restored[idx].detach().cpu())
            plt.subplot(len(idxs)*2, len(perturbation_size)+1, k+i+2)
            plt.imshow(img)
            plt.axis('off')
            if k == 0:
                plt.title(f'p={pSize}', fontsize=8)
            
            k += len(perturbation_size)+1
            plt.subplot(len(idxs)*2, len(perturbation_size)+1, k+1)
            plt.imshow(prepare_image(corrupt_un[idx]))
            plt.axis('off')
            plt.ylabel('unconstrained', fontsize=6)
            img = prepare_image(restored_un[idx].detach().cpu())
            plt.subplot(len(idxs)*2, len(perturbation_size)+1, k+i+2)
            plt.imshow(img)
            plt.axis('off')

            k += len(perturbation_size)+1
        i += 1
    
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.tight_layout()
    plt.savefig(f'./figs/{args.problem}_{args.Trial}_{args.nLayers}_{args.lr}_{args.lr_dual}_constrained_{args.constrained}/perturbation_{args.constrained}.png')
    plt.close()
