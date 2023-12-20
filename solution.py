import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
import IPython
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.metrics import roc_auc_score

from models.d2gmm.detector import *
from models.d2gmm.utils import *
from models.d2gmm.g_loader import *
from models.d2gmm.logger import Logger

class Solution(object):
    DEFAULTS = {}   
    def __init__(self, data_loader_new, config, mask, n_gmm, latent_dim, fusion2_sign, latent_feat2): #--new--
        # Data loader
        self.__dict__.update(Solution.DEFAULTS, **config)
        self.data_loader_new = data_loader_new
        self.mask=mask #--new--
        self.if_mask=config['if_mask']
        self.activation_func=config['activation_func']
        self.len_feat=config['len_feat']
        self.fusion2_sign=fusion2_sign
        self.latent_feat2=latent_feat2
        self.n_gmm=n_gmm
        self.latent_dim=latent_dim

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        self.detecor = Detector(n_gmm=self.n_gmm, latent_dim=self.latent_dim, mask=self.mask, if_mask=self.if_mask, activation_func=self.activation_func, len_feat=self.len_feat, fusion1_sign=self.fusion1_sign, fusion2_sign=self.fusion2_sign) #--new--

        # Optimizers
        for name, param in self.detecor.named_parameters():
            if name=='w':
                print(name,param)
            else:
                print(name)
        self.optimizer = torch.optim.Adam(self.detecor.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.detecor.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

    def load_pretrained_model(self):
        self.detecor.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_detecor.pth'.format(self.pretrained_model))))

    def build_tensorboard(self):
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.detecor.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.data_loader_new)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()

        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            for i, (input_data, labels, cut_latent_feat1, cut_latent_feat2) in enumerate(tqdm(self.data_loader_new)):
                for name, param in self.detecor.named_parameters():
                    if name=='w':
                        print(name,param)
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                total_loss,sample_energy, recon_error, cov_diag = self.detecor_step(input_data, cut_latent_feat1, cut_latent_feat2)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    IPython.display.clear_output()

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1

                        plt.show()

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.detecor.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_detecor.pth'.format(e+1, i+1)))

    def detecor_step(self, input_data, cut_latent_feat1, cut_latent_feat2):
        self.detecor.train()
        input_data2, enc, dec, z, gamma = self.detecor(input_data, cut_latent_feat1, cut_latent_feat2)  
        total_loss, sample_energy, recon_error, cov_diag = self.detecor.loss_function(input_data2, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag) #--important--

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.detecor.parameters(), 5)
        self.optimizer.step()

        return total_loss,sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        self.detecor.eval()
        self.data_loader_new.dataset.mode="train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels, cut_latent_feat1, cut_latent_feat2) in enumerate(self.data_loader_new):
            input_data = self.to_var(input_data)
            input_data2, enc, dec, z, gamma = self.detecor(input_data, cut_latent_feat1, cut_latent_feat2)
            phi, mu, cov = self.detecor.compute_gmm_params(z, gamma)
            
            batch_gamma_sum = torch.sum(gamma, dim=0)
            
            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
            
            N += input_data2.size(0)
            
        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels, cut_latent_feat1, cut_latent_feat2) in enumerate(self.data_loader_new):
            input_data = self.to_var(input_data)
            input_data2, enc, dec, z, gamma = self.detecor(input_data, cut_latent_feat1, cut_latent_feat2)
            sample_energy, cov_diag = self.detecor.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
            
            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())


        train_energy = np.concatenate(train_energy,axis=0)
        train_z = np.concatenate(train_z,axis=0)
        train_labels = np.concatenate(train_labels,axis=0)

        self.data_loader_new.dataset.mode="test"
        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels, cut_latent_feat1, cut_latent_feat2) in enumerate(self.data_loader_new):
            input_data = self.to_var(input_data)
            input_data2, enc, dec, z, gamma = self.detecor(input_data, cut_latent_feat1, cut_latent_feat2)
            sample_energy, cov_diag = self.detecor.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy,axis=0)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)

        thresh = np.percentile(combined_energy, 100 - 20)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        auc=roc_auc_score(gt, pred)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(accuracy, precision, recall, f_score, auc))
        
        return accuracy, precision, recall, f_score
