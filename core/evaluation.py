import torch
import logging
import pickle

from core.utils import scale_image, save_layer_image
from data import image_corruption

def evaluate(model, loader, args, perturbation=False, pSize=0, **kwargs):
    objective_function= kwargs.get('objective_function', None)
    device = kwargs['device']
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
        ibatch = kwargs["ibatch"]
    else:
        epoch = 'Test'
        ibatch = 0

    validloss = 0
    for i, (images, _) in enumerate(loader):
        if i == 200:
            break
        corrupt_img, _ = image_corruption(images, args, perturbation=perturbation, pSize=pSize) #blurring, inpainting, noising
        corrupt_img_scale = scale_image(corrupt_img, args.nBits)
        restored_img, outsx, _ = model(corrupt_img_scale.to(device), objective_function=objective_function, noisyOuts=args.noisyOuts)
        validloss += (torch.norm(restored_img.detach().cpu() - images, p='fro')**2).item() #MSE
        if i == 0:
            if epoch == 'Test':
                with open(f'./results/{args.dataset}_{args.constrained}_{pSize}.pkl', 'wb') as ObjFile:
                    pickle.dump((images, corrupt_img, restored_img, epoch, ibatch), ObjFile)
            save_layer_image(images, corrupt_img, outsx, epoch, ibatch, args, perturbation=False)
            break
        del images, corrupt_img, restored_img, outsx, corrupt_img_scale
        torch.cuda.empty_cache()


        
    validloss /= (len(loader)*args.batchSize)
        
    logging.debug("Epoch {} - Batch {}, Loss {:.4f}".format(epoch, ibatch, validloss))

    return validloss

