import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

from core.model import Glow

class Unrolled_glow(nn.Module):
    def __init__(self, nLayers, in_channel, depth, nLevels, sigma=1e-2):
        super(Unrolled_glow, self).__init__()
        self.nLayers = nLayers
        self.layers = nn.ModuleList()
        for l in range(self.nLayers):
            layer = nn.ModuleDict()
            layer['glow'] = Glow(in_channel, depth, nLevels)
            layer['scalars'] = nn.ParameterList([nn.Parameter(torch.randn(1,1)*sigma), #mu
                                nn.Parameter(torch.randn(1,1)*sigma)])                 #lambda
            self.layers.append(layer)

    def forward(self, input:torch.tensor , A=None, noisyOuts=False, **kwargs):
        shape = input.shape
        if len(input.shape) > 2:
            input = input.reshape(input.shape[0], -1)
        if A is None:
            A = torch.eye(input.shape[1], device=input.device)
        # Initialization
        X = input 
        outs = {} 
        outsz = {}
        outs[0] = X
        for l in range(self.nLayers):
            if l < self.nLayers-1 and noisyOuts:
                X = X.reshape(shape)
                n = torch.randn_like(X)
                X = X + (1/(l+1))*((n/torch.norm(n, p=2, dim=(1,2,3), keepdim=True)).permute((1,2,3,0))).permute((3,0,1,2))
                # if torch.sum(X.isnan()) > 0:
                #     break
                X = torch.clamp(X, min=0, max=1)
            if len(X.shape) > 2:
                X = X.reshape(input.shape[0], -1)
            X = X - nn.ReLU()(self.layers[l]['scalars'][0]) * ((input - X @ A.T) @ A)      # x-space update
            _,_, outsZ = self.layers[l]['glow'](X.reshape(shape))               # convert to the z-space
            outsZ[-1] /= (1 + nn.ReLU()(self.layers[l]['scalars'][1]))          # z-space update
            X = self.layers[l]['glow'].reverse(outsZ)                           # convert back to the x-space
            X = torch.clamp(X, min=0, max=1)
            outs[l+1] = X
            outsz[l+1] = outsZ[-1]
            del outsZ
            torch.cuda.empty_cache()
        return X, outs, outsz
    
    def forward_noise(self, input:torch.tensor , A=None, **kwargs):
        beta = kwargs.get('beta', 0)
        shape = input.shape
        if len(input.shape) > 2:
            input = input.reshape(input.shape[0], -1)
        if A is None:
            A = torch.eye(input.shape[1], device=input.device)
        # Initialization
        X = input 
        outs = {} 
        outsz = {}
        outs[0] = X
        for l in range(self.nLayers):
            if l < self.nLayers-1:
                X = X.reshape(shape)
                n = torch.randn_like(X)
                X = X + beta*(1/(l+1))*((n/torch.norm(n, p=2, dim=(1,2,3), keepdim=True)).permute((1,2,3,0))).permute((3,0,1,2))
                # if torch.sum(X.isnan()) > 0:
                #     break
                #torch.clamp(X, min=0, max=1)
            if len(X.shape) > 2:
                X = X.reshape(input.shape[0], -1)
            X = X - nn.ReLU()(self.layers[l]['scalars'][0]) * ((input - X @ A.T) @ A)      # x-space update
            _,_, outsZ = self.layers[l]['glow'](X.reshape(shape))               # convert to the z-space
            outsZ[-1] /= (1 + nn.ReLU()(self.layers[l]['scalars'][1]))          # z-space update
            X = self.layers[l]['glow'].reverse(outsZ)                           # convert back to the x-space
            outs[l+1] = X
            outsz[l+1] = outsZ[-1]
            del outsZ
            torch.cuda.empty_cache()
        return X, outs, outsz

    # def forward_noise(self, X:torch.tensor, **kwargs):
    #     beta = kwargs.get('beta', 100)
    #     W = kwargs['SysID']
    #     # Initialize Z
    #     X = X*100
    #     Z = torch.zeros((self.M, X.shape[1]), device=X.device)
    #     Z = torch.randn_like(Z, device=X.device) * beta
    #     true_outs = {}
    #     outs = {} 
    #     true_outs[0] = Z
    #     outs[0] = Z
    #     for l in range(self.nLayers):
    #         W1 = self.layers[l]['W1']
    #         W2 = self.layers[l]['W2']
    #         theta = self.layers[l]['theta']
    #         #Z = Z - (W.T @ (W @ Z - X))
    #         Z = W1.float()@Z +  W2.float()@X
    #         Z = torch.sign(Z) * torch.maximum(torch.abs(Z) - theta, torch.zeros_like(Z))
    #         true_outs[2*l+1] = Z/100
    #         # if  l < self.nLayers-1:
    #         #     grad = torch.norm(jacobian(objective_function, (Z, X, W), create_graph=True)[0], p=2, dim=0)
    #         #     Z = Z + torch.randn_like(Z) * torch.log(grad) * beta 
    #         outs[l+1] = Z/100
    #         true_outs[2*l+2] = Z/100
    #     return Z/100, outs, true_outs