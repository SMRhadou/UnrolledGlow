import torch
import numpy as np
from torch.autograd.functional import jacobian

def gaussian_log_p(z, mean, sigma):
    k = z.shape[1]
    return -k/2 * np.log(2*np.pi) -0.5 * torch.logdet(sigma) -0.5 * torch.diag((z-mean) @ torch.inverse(sigma) @ (z-mean).T)


def objective_function(corrupt_img, estx, estz, A):
    if len(corrupt_img.shape) > 2:
        corrupt_img = corrupt_img.reshape(corrupt_img.shape[0], -1)
    if len(estx.shape) > 2:
        estx = estx.reshape(estx.shape[0], -1)
    if len(estz.shape) > 2:
        estz = estz.reshape(estz.shape[0], -1)
    return torch.sum(100*torch.norm(corrupt_img - estx @ A.T, dim=1)**2 - gaussian_log_p(estz, 0, torch.eye(estz.shape[1], device=estz.device)))

def compute_gradient(objective_function, corrupt_img, estx, estz, A):
    devx, devz = jacobian(objective_function, (corrupt_img, estx, estz, A), create_graph=True)[1:3]
    #return torch.mean(torch.sqrt(torch.norm(devx, p=2, dim=1)**2 + torch.norm(devz, p=2, dim=1)**2))
    return torch.mean(torch.norm(devx, p=2, dim=1)),  + torch.mean(torch.norm(devz, p=2, dim=1))

def grad_penalty(objective_function, corrupt_img, outsx, outsz, A, **kwargs):
    eps = kwargs['eps']
    torch.cuda.empty_cache()
    L = len(outsx.keys()) - 1
    gradVector = torch.zeros((L+1,2)).to(kwargs['device'])
    for l in outsz.keys():
        estx = outsx[l].reshape((outsx[l].shape[0], -1))
        estz = outsz[l].reshape((outsx[l].shape[0], -1))
        gradVector[l, 0] = compute_gradient(objective_function, corrupt_img, estx, estz, A)[0]
        gradVector[l, 1] = compute_gradient(objective_function, corrupt_img, estx, estz, A)[1]
    cons = gradVector[2:,:] / gradVector[1:-1,:] - (1-eps)
    return cons, gradVector     #L+1 x 2

def Lagrang_loss(restored_img, images, cons, nu, device):
    return MSE(restored_img, images.to(device)) + torch.sum(nu * cons)

def MSE(restored_img, images):
    loss = torch.norm(restored_img - images, p='fro')**2
    return loss.div(images.shape[0])
