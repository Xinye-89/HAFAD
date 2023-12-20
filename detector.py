import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
import sys
sys.setrecursionlimit(1000000) 

from models.d2gmm.utils import *

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, upper=False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
    
class Detector(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 2, latent_dim=3, mask=None, if_mask=False, activation_func=nn.Tanh(), len_feat=None, fusion1_sign=None, fusion2_sign=None): 
        super(Detector, self).__init__()
        self.mask=mask
        self.if_mask=if_mask
        self.fusion1_sign=fusion1_sign
        self.fusion2_sign=fusion2_sign
        print('---------len_feat---------')
        print(len_feat)

        layers = []
        
        layers +=[nn.Linear(len_feat,20)]
        nn.BatchNorm1d(20),
        layers +=[activation_func]
        layers +=[nn.Linear(20, 10)]
        nn.BatchNorm1d(10),
        layers +=[activation_func]
        layers +=[nn.Linear(10,1)]

        self.encoder = nn.Sequential(*layers)

        layers = []

        layers +=[nn.Linear(1,10)]
        nn.BatchNorm1d(10),
        layers +=[activation_func]
        layers +=[nn.Linear(10,20)]
        nn.BatchNorm1d(20),
        layers +=[activation_func]
        layers +=[nn.Linear(20, len_feat)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim,10)]
        layers += [activation_func]   
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))  
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))  
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))  
        
        self.w = torch.nn.Parameter(torch.tensor([0.5]))  
        self.w2 = torch.nn.Parameter(torch.tensor([0.5]))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x, x2, y):
        x3=x.clone()
        if self.fusion1_sign==True:
            x3 = torch.cat((self.w.item()*x, (1-self.w.item())*x2.unsqueeze(1)), dim=1)
            enc = self.encoder(x3)
        else:
            enc = self.encoder(x)

        dec = self.decoder(enc) #--old--
        
        if self.if_mask:
            dec.loc[self.mask,:] =0 #--new--
        
        rec_cosine = F.cosine_similarity(x3, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x3, dec)
        criterion = nn.MSELoss(reduction='none')
        rec_mse = criterion(x3, dec)
        rec_mse = rec_mse.mean(dim=1)
        
        if self.fusion2_sign==True:
            z = torch.cat([self.w2.item()*(enc+torch.tensor(y,dtype=torch.float32).unsqueeze(-1)*0.01), (1-self.w2.item())*rec_euclidean.unsqueeze(-1),  (1-self.w2.item())*rec_cosine.unsqueeze(-1)], dim=1)
        else :
            z = torch.cat([enc, rec_euclidean.unsqueeze(-1),  rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return x3, enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)
        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + to_var(torch.eye(D)*eps) 
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        #det_cov = torch.cat(det_cov).cuda()  *old*
        det_cov=torch.cat(det_cov)
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag
