import numpy as np
import utils as ut

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImputeClass:
    """
    Base model class that defines the interface for training, prediction, and evaluation.
    This class is designed to be subclassed by specific model implementations.
    """
    
    def __init__(self, **kwargs):
        self.model = None
    
    def train_model(self, X, W, mask, **kwargs):
        raise NotImplementedError("Subclasses must implement train_model method")
    
    def predict(self, X, **kwargs):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def eval_model(self, X, W, **kwargs):
        print("Observed W: ", W.shape)
        mse = np.mean((self.predict(X, W=W, **kwargs) - W) ** 2)
        return {'MSE': mse}

########################################################
# Ground-true Imputation
########################################################

class GroundTrueImpute(ImputeClass):
    def __init__(self, **kwargs):
        super().__init__()
        
    def train_model(self, X, W, mask, **kwargs):
        return None
    
    def predict(self, X, **kwargs):
        assert "W" in kwargs, "W is required for GroundTrueImpute"
        return kwargs["W"]
    
########################################################
# Mean Imputation
########################################################
    
class MeanImpute(ImputeClass):
    def __init__(self, default_value=0, **kwargs):
        super().__init__()
        self.model = []
        self.default_value = default_value
        
    def train_model(self, X, W, mask, **kwargs):
        self.model = []
        for q in range(mask.shape[-1]):
            if np.sum(mask[:, q]) > 0:
                self.model.append(W[mask[:, q]==1, q].mean())
            else:
                self.model.append(self.default_value)
                
    def predict(self, X, **kwargs):
        n_samples, n_tasks = X.shape[0], len(self.model)
        W_impute = np.zeros((n_samples, n_tasks))
        
        for q in range(n_tasks):
            W_impute[:, q] = self.model[q]
            
        return W_impute
    
########################################################
# Random Forest Imputation
########################################################
    
class RandomForestImpute(ImputeClass):
    def __init__(self, default_value=0, **kwargs):
        super().__init__()
        self.default_value = default_value
        self.training_kwargs = kwargs
        
    def train_model(self, X, W, mask, **kwargs):
        self.model = []
        for q in range(mask.shape[-1]):
            idx = (mask[:, q]==1)
            if np.sum(idx) > 0:
                model = RandomForestRegressor(**self.training_kwargs)
                model.fit(X[idx, :], W[idx, q])
                self.model.append(model)
            else:
                self.model.append(self.default_value)
    
    def predict(self, X, **kwargs):
        n_samples, n_tasks = X.shape[0], len(self.model)
        W_impute = np.zeros((n_samples, n_tasks))
        
        for q in range(n_tasks):
            if self.model[q] is not None:
                W_impute[:, q] = self.model[q].predict(X)
            else:
                assert self.model[q] == self.default_value, "Model is not trained for task %d" % q
                W_impute[:, q] = self.model[q]
                raise Warning("No training data for task %d" % q, "Impute with default value" % self.default_value)
        
        return W_impute

########################################################
# AUTOENCODER
########################################################
class AutoEncoderImpute(ImputeClass):
    def __init__(self, input_dim, encoder_dim_list, output_dim, device='cpu', seed=1, **kwargs):
        super().__init__()
        self.device = device
        self.model = AutoEncoder(input_dim, encoder_dim_list, output_dim, seed=seed).to(device)
        
    def train_model(self, X, W, mask, **kwargs):
        ut.train_nn(self.model, (X, W, mask), device=self.device, **kwargs)
        
    def predict(self, X, **kwargs):
        self.model.eval()
        preds = self.model(torch.Tensor(X).to(self.device))
        self.model.train()
        return preds.cpu().detach().numpy()    
    
        
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim_list, output_dim, seed=1):
        super().__init__()
        ut.seed_torch(seed)
        
        assert len(encoder_dim_list) > 0, "Hidden dimension list must be non-empty"
        
        model_list = [nn.Linear(input_dim, encoder_dim_list[0])]
        for i in range(len(encoder_dim_list) - 1):
            model_list.append(nn.LeakyReLU())
            # model_list.append(nn.Dropout(p=0.1))
            model_list.append(nn.Linear(encoder_dim_list[i], encoder_dim_list[i+1]))
        
        model_list.append(nn.LeakyReLU())
        model_list.append(nn.Linear(encoder_dim_list[-1], output_dim))
        
        self.encoder = nn.Sequential(*model_list)
        
    def forward(self, X):
        return self.encoder(X)
    
    def loss(self, X, W, M, device='cpu', *args, **kwargs):
        X, W, M = X.float().to(device), W.float().to(device), M.float().to(device)
        X_impute = self.forward(X)
        loss = 0
        for q in range(M.shape[-1]):
            if torch.sum(M[:, q]) > 0:
                loss += nn.MSELoss()(X_impute[M[:, q]==1, q], W[M[:, q]==1, q])
        return loss
    
########################################################
# Variational Autoencoder
########################################################
class VariationalImpute(ImputeClass):
    def __init__(self, input_dim, encoder_dim_list, latent_dim, decoder_dim_list, output_dim, device='cpu', seed=1,**kwargs):
        super().__init__()
        ut.seed_torch(seed)
        self.device = device
        self.model = VariationalAutoEncoder(input_dim, encoder_dim_list, latent_dim, decoder_dim_list, output_dim, seed=seed, **kwargs).to(device)
        
    def train_model(self, X, W, mask, **kwargs):
        ut.train_nn(self.model, (X, W, mask), device=self.device, **kwargs)
        
    def predict(self, X, seed=1, **kwargs):
        self.model.eval()
        use_mean = True if 'use_mean' not in kwargs else kwargs['use_mean']
        preds = self.model(torch.Tensor(X).to(self.device), mean=use_mean, seed=seed)
        self.model.train()
        return preds.cpu().detach().numpy()    
    
class OnlineVariationalImpute(VariationalImpute):
    def __init__(self, input_dim, encoder_dim_list, latent_dim, decoder_dim_list, output_dim, device='cpu', **kwargs):
        super().__init__(input_dim, encoder_dim_list, latent_dim, decoder_dim_list, output_dim, device=device, **kwargs)
        
    def train_model(self, X, W, mask, propen=None, **kwargs):
        if propen is None:
            propen = mask.mean(0, keepdims=True).repeat(X.shape[0], 0) # Fixed propensity e.g. RMT
        else:
            print("Using online propensity")
        ut.train_nn(self.model, (X, W, mask, propen), device=self.device, **kwargs)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim_list, latent_dim, decoder_dim_list, output_dim, sigma_prior=None, seed=1):
        super().__init__()
        ut.seed_torch(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Define encoder
        self.encoder_model_list = [nn.Linear(input_dim, encoder_dim_list[0])]
        for i in range(len(encoder_dim_list) - 1):
            self.encoder_model_list.append(nn.LeakyReLU())
            # self.encoder_model_list.append(nn.Dropout(p=0.1))
            self.encoder_model_list.append(nn.Linear(encoder_dim_list[i], encoder_dim_list[i+1]))
        self.encoder = nn.Sequential(*self.encoder_model_list)
        
        # Define mu and logvar
        self.mu_model = nn.Linear(encoder_dim_list[-1], latent_dim)
        self.logvar_model = nn.Linear(encoder_dim_list[-1], latent_dim)
        
        # Define decoder
        if len(decoder_dim_list) > 0:
            self.decoder_model_list = [nn.Linear(decoder_dim_list[-1], output_dim)]
            for i in range(len(decoder_dim_list) - 1, 0, -1):
                self.decoder_model_list.append(nn.Linear(decoder_dim_list[i-1], decoder_dim_list[i]))
                # self.decoder_model_list.append(nn.Dropout(p=0.1))
                self.decoder_model_list.append(nn.LeakyReLU())
            self.decoder_model_list.append(nn.Linear(latent_dim, decoder_dim_list[0]))
        else:
            self.decoder_model_list = [nn.Linear(latent_dim, output_dim)]
        
        self.decoder = nn.Sequential(*self.decoder_model_list[::-1])
        
        # Define Prior
        if sigma_prior is None:
            self.sigma_0 = nn.Parameter(torch.ones(1, latent_dim), requires_grad=False)
        else:
            self.sigma_0 = nn.Parameter(torch.tensor(sigma_prior), requires_grad=False)
            
    def encode(self, X):
        Z = self.encoder(X)
        mu, logvar = self.mu_model(Z), self.logvar_model(Z)
        std = torch.exp(torch.clip(0.5 * logvar, -50, 50))
        Z_recon = mu + std * torch.randn_like(std)
        return Z_recon, mu, logvar
    
    def decode(self, Z):
        return self.decoder(Z)
    
    def forward(self, X, mean=False, seed=1):
        ut.seed_torch(seed)
        Z_recon, mu, logvar = self.encode(X)
        return self.decode(Z_recon) if not mean else self.decode(mu)
    
    def loss(self, X, W, M, propen=None, device='cpu', *args, **kwargs):
        X, W, M = X.float().to(device), W.float().to(device), M.float().to(device)
        Z_recon, mu, logvar = self.encode(X)
        W_recon = self.decode(Z_recon)
        
        if propen is None:
            propen = torch.ones_like(W)
            
        recon_loss = 0
        for q in range(M.shape[-1]):
            if torch.sum(M[:, q]) > 0:
                idx = M[:, q]==1
                recon_loss += (1/propen[idx,q] * (W_recon[idx, q] - W[idx, q])**2).mean()
        KL_loss = -0.5 * torch.sum(1 + logvar + 2*self.sigma_0.log() - mu**2/self.sigma_0**2 - torch.exp(logvar) / self.sigma_0**2)
        return recon_loss + KL_loss
    
    

########################################################
# Image Variational Autoencoder
########################################################
class ImageVariationalImpute(ImputeClass):
    def __init__(self, input_shape, latent_dim, device='cpu', seed=1, **kwargs):
        super().__init__()
        ut.seed_torch(seed)
        self.device = device
        self.model = ImageVariationalAutoEncoder(input_shape, latent_dim, seed=seed, device=device, **kwargs).to(device)
        
    def train_model(self, X, W, mask, mask_action, **kwargs):
        if isinstance(mask_action, np.ndarray):
            mask_action = np.expand_dims(mask_action, axis=0).repeat(X.shape[0], 0)
        else:
            mask_action = mask_action.unsqueeze(0).repeat_interleave(X.shape[0], 0)
        ut.train_nn(self.model, (X, W, mask, mask_action), device=self.device, **kwargs)
        
    def predict(self, X, seed=1, **kwargs):
        self.model.eval()
        use_mean = True if 'use_mean' not in kwargs else kwargs['use_mean']
        preds = self.model(torch.Tensor(X).to(self.device), mean=use_mean, seed=seed)
        self.model.train()
        return preds.cpu().detach().numpy()


class ImageVariationalAutoEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim, sigma_prior=None, device='cpu', seed=1):
        super().__init__()
        ut.seed_torch(seed)
        self.input_shape = input_shape  # (C, H, W)
        self.latent_dim = latent_dim
        self.device = device
        
        # 16 32
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2),  # [batch, 16, 12, 12]
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  # [batch, 32, 4, 4]
            nn.Flatten()  # [batch, 128*4*4]
        )
        
        self.mu_layer = nn.Linear(16 * 4 * 4, latent_dim)
        self.logvar_layer = nn.Linear(16 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 16 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (16, 4, 4)),  # [batch, 128, 4, 4]
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2),  # [batch, 16, 2, 2]
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2),  # [batch, 32, 14, 14]
            nn.ConvTranspose2d(1, 1, kernel_size=4),  # [batch, 1, 28, 28]
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Define Prior
        if sigma_prior is None:
            self.sigma_0 = nn.Parameter(torch.ones(1, latent_dim).to(device), requires_grad=False)
        else:
            self.sigma_0 = nn.Parameter(torch.tensor(sigma_prior).to(device), requires_grad=False)
    
    def encode(self, X):        
        # Convolutional encoding
        x = self.encoder_conv(X)
        
        # Get latent parameters
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        
        # Sample from latent space
        std = torch.exp(torch.clip(0.5 * logvar, -10, 10))
        Z = mu + std * torch.randn_like(std).to(self.device)
        
        return Z, mu, logvar
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder_conv(h)

    def forward(self, X, mean=False, seed=1):
        ut.seed_torch(seed)
        Z, mu, logvar = self.encode(X)
        return self.decode(Z) if not mean else self.decode(mu)
    
    def loss(self, X, W, M, mask_action, propen=None, device='cpu', *args, **kwargs):
        X, W, M, = X.float().to(device), W.float().to(device), M.float().to(device)
        Z, mu, logvar = self.encode(X)
        X_recon = self.decode(Z)
        
        num_patch = M.shape[-1]
        if propen is None:
            propen = torch.ones_like(W).to(device)

        recon_loss = 0
        for q in range(num_patch):
            if torch.sum(M[:, q]) > 0:
                idx = M[:, q]==1
                # patch_mask = (W[[0],q,].abs() > 0).int()
                patch_mask = mask_action[0,q,].to(device)
                # print((1 / propen[idx,q,] *  patch_mask * (X_recon[idx,] - W[idx,q,])**2).shape)
                # print((1/propen[idx,q,] * patch_mask * (X_recon[idx,]- W[idx,q,])**2).flatten(1).shape)
                recon_loss += ((1/propen[idx,q,] * patch_mask * (X_recon[idx,]- W[idx,q,])**2).flatten(1).sum(1) / torch.sum(patch_mask)).mean()
        
        # patch_mask = (W[[0],].abs().sum(1) == 0).int()
        # print(((patch_mask * X_recon).reshape(-1, 1)).shape)
        # recon_loss += (patch_mask * X_recon).reshape(-1, 1).sum(1) / torch.sum(patch_mask)
        
        # KL divergence
        KL_loss = -0.5 * torch.sum(1 + logvar + 2*self.sigma_0.log() - mu**2/self.sigma_0**2 - torch.exp(logvar) / self.sigma_0**2)
        
        return recon_loss + KL_loss
