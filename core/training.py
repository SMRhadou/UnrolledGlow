import numpy as np
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import core.evaluation as eval
from core.optimizee import grad_penalty, Lagrang_loss, MSE
from core.utils import scale_image
from data import image_corruption


# %% Training 
def train(model, dataset, validset, optimizer, objective_function, args, device, **kwargs):
    nu = torch.zeros((model.module.nLayers-1,2), device=torch.device(device)).double()

    trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batchSize, num_workers=4)
    validloader = DataLoader(validset, shuffle=False, batch_size=args.batchSize, num_workers=4)
    del dataset
    
    best, epochBest, batchBest = np.inf, 0, 0
    for epoch in range(args.nEpochs):
        for ibatch, (images, _) in tqdm(enumerate(trainloader)):
            model, nu = learning_step(model, images, optimizer, objective_function, nu, args, 
                                      device=device, epoch=epoch, ibatch=ibatch, **kwargs)

            if ibatch % 1000 == 0 or ibatch ==50:
                model, best, epochBest, batchBest = valid_step(model, validloader, best, args, objective_function, 
                                                               device=device, epoch=epoch, epochBest=epochBest,
                                                               ibatch=ibatch, batchBest=batchBest, **kwargs)
    return model


def learning_step(model, images, optimizer, objective_function, nu, args, **kwargs):
    device = kwargs["device"] 
    epoch = kwargs["epoch"]
    ibatch = kwargs["ibatch"]
    
    model.train()
    corrupt_img, A = image_corruption(images, args) #blurring, inpainting, noising
    corrupt_img = scale_image(corrupt_img, args.nBits)
    restored_img, outsx, outsz = model(corrupt_img.to(device), objective_function=objective_function, noisyOuts=args.noisyOuts)
    cons, grad = grad_penalty(objective_function, corrupt_img.to(device), outsx, outsz, A.to(device), eps=args.eps, **kwargs)
    
    if args.constrained:
        loss = Lagrang_loss(restored_img, images, cons, nu, device)
        #loss = objective_function(corrupt_img.to(device), restored_img, outsz[args.nLayers], A.to(device))  + torch.sum(nu * cons) 
    else:
        loss = MSE(restored_img, images.to(device))
        #loss = objective_function(corrupt_img.to(device), restored_img, outsz[args.nLayers], A.to(device))
    logging.debug("\n epoch {} - batch {}: loss {:.3f}".format(epoch, ibatch, loss))
    logging.debug('In x-direction')
    logging.debug('gradients {}'.format(list(grad.T[0].detach().cpu().numpy().squeeze())))
    logging.debug('constraints {}'.format(list(cons.T[0].detach().cpu().numpy().squeeze())))
    logging.debug('nu {}'.format(list(nu.T[0].detach().cpu().numpy().squeeze())))
    logging.debug("In z-direction")
    logging.debug('gradients {}'.format(list(grad.T[1].detach().cpu().numpy().squeeze())))
    logging.debug('constraints {}'.format(list(cons.T[1].detach().cpu().numpy().squeeze())))
    logging.debug('nu {}'.format(list(nu.T[1].detach().cpu().numpy().squeeze())))
    logging.debug('Obj. function across layers: {}'.format([objective_function(corrupt_img.to(device),
                                                                                outsx[i+1].to(device), outsz[i+1].to(device),
                                                                                A.to(device)).item() for i in range(model.module.nLayers)]))
    logging.debug('MSE across layers: {}'.format([torch.norm(outsx[i+1].to(device)-images.to(device), p='fro').mean().item() for i in range(model.module.nLayers)]))
    logging.debug('Latent variables across layers: {}'.format([torch.norm(outsz[i+1].to(device), p='fro').mean().item() for i in range(model.module.nLayers)]))
    del images, corrupt_img, restored_img, outsx, outsz, grad
    torch.cuda.empty_cache()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    del loss
    torch.cuda.empty_cache()

    if args.constrained:
        # Dual update
        nu_temp = nu + args.lr_dual * cons
        nu = nn.ReLU()(nu_temp)
        nu = nu.detach()

    del cons
    torch.cuda.empty_cache()

    return model, nu

def valid_step(model, validloader, best, args, objective_function, **kwargs):
    device = kwargs["device"]
    modelPath = kwargs["modelPath"]
    epoch = kwargs["epoch"]
    ibatch = kwargs["ibatch"]
    epochBest = kwargs["epochBest"]
    batchBest = kwargs["batchBest"]

    model.eval()    
    validloss = eval.evaluate(model, validloader, args, objective_function=objective_function, device=device, epoch=epoch, ibatch=ibatch)

    # Save model
    if validloss < best:
        best = validloss
        epochBest = epoch
        batchBest = ibatch
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "valid_loss": validloss
                    }, modelPath+f"model_best.pth")
    logging.debug("validloss {}, best: {} achieved at epoch {} - batch {}\n".format(validloss, 
                                                                                    best, epochBest, batchBest))

    del validloss
    torch.cuda.empty_cache()
    return model, best, epochBest, batchBest



def unconstrained_learning(model, trainloader, validloader, optimizer, args, **kwargs):
    device = kwargs["device"] 
    modelPath = kwargs["modelPath"]
    best = np.inf
    for epoch in tqdm(range(args.nEpochs)):
        for ibatch, (images, _) in enumerate(trainloader):
            if ibatch == 40000:
                break
            model.train()
            corrupt_img, A = image_corruption(images, args) #blurring, inpainting, noising
            corrupt_img = scale_image(corrupt_img, args.nBits)
            restored_img, _, _ = model(corrupt_img.to(device), noisyOutputs=False)
            loss = torch.norm(restored_img - images.to(device), p='fro') #MSE
            logging.debug("epoch {} - batch {}: loss {}".format(epoch, ibatch, loss))
            del corrupt_img, restored_img
            
            loss.div(images.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del loss, images
            torch.cuda.empty_cache()

            # Validation
            model.eval()
            if ibatch % 300 == 0 and ibatch > 0:
                validloss = eval.evaluate(model, validloader, args, device=device, epoch=epoch, ibatch=ibatch)

                # Save model
                if validloss < best:
                    best = validloss
                    epochBest = epoch
                    batchBest = ibatch
                    torch.save({"epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "valid_loss": validloss
                                }, modelPath+"model_best.pth")
                logging.debug("validloss {}, best: {} achieved at epoch {} - batch {}\n".format(validloss, 
                                                                                                best, epochBest, batchBest))
                
                del validloss
                torch.cuda.empty_cache()
    return model
