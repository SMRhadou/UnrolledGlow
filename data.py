import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def loadDataset(shape, nBits=5, Dataset='CIFAR10'):
    transform=transforms.Compose([
            transforms.Resize(shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPosterize(nBits, p=1.0),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if Dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
    elif Dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
    elif Dataset == 'CelebA':
        tempset = torchvision.datasets.ImageFolder(
        root='./data/CelebA/CelebA', transform=transform)
        idx = np.arange(len(tempset))
        nb_eval = int(len(tempset) * 0.05)
    elif Dataset == 'CelebA-HQ':
        tempset = torchvision.datasets.ImageFolder(
        root='./data/CelebAMask-HQ/', transform=transform)
        idx = np.arange(len(tempset))
        nb_eval = int(len(tempset) * 0.05)
    elif Dataset == 'AnimeFaces':
        tempset = torchvision.datasets.ImageFolder(
        root='./data/AnimeFaces/', transform=transform)
        idx = np.arange(len(tempset))
        nb_eval = int(len(tempset) * 0.05)
    else:
        raise NotImplementedError
    
    eval_idx = np.random.choice(idx, (nb_eval,2), replace=False)

    mask = np.ones_like(idx)
    mask[eval_idx] = 0

    train_idx = idx[mask == 1]
    
    trainset = torch.utils.data.Subset(tempset, train_idx.tolist())
    validset = torch.utils.data.Subset(tempset, eval_idx[:,0].tolist())
    testset = torch.utils.data.Subset(tempset, eval_idx[:,1].tolist())
    return trainset, validset, testset


def image_corruption(images, args, perturbation=False, pSize=0):
    if perturbation:
        n = torch.randn_like(images)
        images = images + pSize*n/torch.norm(n, p=2, dim=(1,2,3), keepdim=True)
    if args.problem == 'denoising':
        return noisy_model(images, args)
    elif args.problem == 'inpainting':
        return painting_model(images, args)

def noisy_model(images, args):
    A = torch.eye(images.reshape((images.shape[0],-1)).shape[1], device=images.device)
    return images + args.noise_sigma*torch.randn_like(images), A

def painting_model(images, args, mean1=None, mean2=None):
    if mean1 is None:
        mean1 = images.shape[2]//2
    if mean2 is None:
        mean2 = images.shape[3]//2
    mask = torch.ones((images.shape[1], images.shape[2], images.shape[3]))
    mask[:, mean1-args.painting_size//2:mean1+args.painting_size//2,
         mean2-args.painting_size//2:mean2+args.painting_size//2] = 0
    return images * mask, torch.diag(mask.reshape((-1,)))