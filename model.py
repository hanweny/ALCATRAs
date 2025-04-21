import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as ut

class ModelClass:
    """
    Base model class that defines the interface for training, prediction, and evaluation.
    This class is designed to be subclassed by specific model implementations.
    """
    
    def __init__(self, classification=False):
        self.classification = classification
        self.model = None
    
    def train_model(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses must implement train_model method")
    
    def predict(self, X, return_uncertainty=False):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def scores(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses must implement scores method")
    
    def eval_model(self, X, y, **kwargs):
        result_dict = {}
        if self.classification:
            acc = (np.array(self.predict(X)) == y).mean()
            result_dict['acc'] = acc
        else:
            mse = np.mean((self.predict(X) - y) ** 2)
            result_dict['MSE'] = mse
        return result_dict
    
class RandomForestModel(ModelClass):
    def __init__(self, classification=False, **kwargs):
        super().__init__(classification)
        if self.classification:
            self.model = RandomForestClassifier(**kwargs)
        else:
            self.model = RandomForestRegressor(**kwargs)
            
    def train_model(self, X, y, **kwargs):
        self.model.fit(X, y)
    
    def predict(self, X, return_uncertainty=False):
        preds = self.model.predict(X)
        if not return_uncertainty:
            return preds
        
        if self.classification:
            probs = self.model.predict_proba(X)
            entropy = np.sum(-probs * np.log(probs), axis=1)
            return preds, entropy
        else:
            std = np.vstack([tree.predict(X) for tree in self.model.estimators_]).T.std(1)
            return preds, std
    
########################################
#### Neural Network Model
########################################

class NeuralNetworkModel(ModelClass):
    def __init__(self, input_dim, hidden_dim_list, output_dim, seed, device='cpu', classification=False, **kwargs):
        super().__init__(classification)
        self.device = device
        self.classification = classification
        self.model = NeuralNet(input_dim, hidden_dim_list, output_dim, seed, classification=classification, **kwargs)
        
    def train_model(self, X, y, reinit=False, **kwargs):
        if reinit:
            self.model.net.apply(ut.reinit_weights)
        ut.train_nn(self.model, (X, y), device=self.device, **kwargs)
    
    def predict(self, X, return_uncertainty=False):
        X = torch.Tensor(X).float().to(self.device)
        return self.model.predict(X, return_uncertainty)


class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, seed, classification=False, **kwargs):
        super().__init__()
        
        ut.seed_torch(seed)
        self.classification = classification
        
        self.model_list = [nn.Linear(input_dim, hidden_dim_list[0], bias=True)]
        for i in range(1, len(hidden_dim_list)):
            self.model_list.append(nn.LeakyReLU())
            self.model_list.append(nn.Dropout(0.1))
            self.model_list.append(nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i], bias=True))
        self.model_list.append(nn.Dropout(0.1))
        self.model_list.append(nn.Linear(hidden_dim_list[-1], output_dim, bias=True))
        self.net = nn.Sequential(*self.model_list)

        # multi-class classification return logits
        self.output_activation = nn.Sigmoid() if output_dim == 1 and self.classification else nn.Identity()
        self.loss_fn = nn.MSELoss() if not self.classification else nn.BCELoss() if output_dim == 1 else nn.CrossEntropyLoss()
        
    def forward(self, X):
        return self.net(X)
    
    def predict(self, X, return_uncertainty=False):
        if self.classification:
            return self.predict_classification(X, return_uncertainty)
        else:
            return self.predict_regression(X, return_uncertainty)
        
    def predict_classification(self, X, return_uncertainty=False):
        self.eval()
        probs = self.output_activation(self.forward(X)).detach().cpu().numpy()  # logits preserves the scale
        y_hat = (probs > 0.5).astype(int) if probs.shape[1] == 1 else np.argmax(probs, axis=1)
        y_hat = y_hat.flatten()
        self.train()
        
        if not return_uncertainty:
            return y_hat
        
        entropy = np.sum(-probs * np.log(probs), axis=1)
        return y_hat, entropy
    
    def predict_regression(self, X, return_uncertainty=False):
        if return_uncertainty:
            preds = torch.stack([self.net(X) for _ in range(30)], dim=0).detach().cpu().numpy()
            return preds.mean(0).flatten(), preds.std(0).flatten()
        else:
            self.eval()
            preds = self.net(X).detach().cpu().numpy()
            self.train()
            return preds.flatten()
    
    def loss(self, X, y, device='cpu'):
        self.net.train()
        X, y = torch.Tensor(X).float().to(device), torch.Tensor(y).float().to(device).reshape(-1,1)
        preds = self.output_activation(self.forward(X))
        loss = self.loss_fn(preds, y)
        return loss
    
########################################
#### Neural Network Model
########################################

class ImageClassificationModel(ModelClass):
    def __init__(self, device='cpu', seed=0, **kwargs):
        super().__init__(True)
        self.classification = True
        self.device = device
        self.model = SimpleCNN(seed=seed, **kwargs).to(device)
    
    def train_model(self, X, y, reinit=False, **kwargs):
        if reinit:
            self.model.layers.apply(ut.reinit_weights)
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X).float().to(self.device)
        ut.train_nn(self.model, (X, y), device=self.device, **kwargs)
    
    def predict(self, X, return_uncertainty=False, **kwargs):
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X).float().to(self.device)
        return self.model.predict(X, return_uncertainty)

class SimpleCNN(nn.Module):
    def __init__(self, seed=0):
        super(SimpleCNN, self).__init__()
        ut.seed_torch(seed)
        
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 7, 128),
        # )
        # self.predict_layer = nn.Linear(128, 10)
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(4),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*3*3, 64)
        )
        self.predict_layer = nn.Linear(64, 10)
        
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.predict_layer(self.layers(x))
    
    def predict(self, X, return_uncertainty=False):
        self.eval()
        probs = self.output_activation(self.forward(X)).detach().cpu().numpy() + 1e-10
        y_hat = np.argmax(probs, axis=1)
        self.train()
        if not return_uncertainty:
            return y_hat
        
        entropy = np.sum(-probs * np.log(probs), axis=1)
        return y_hat, entropy
    
    def representation(self, x):
        return self.layers(x)
    
    def loss(self, X, y, device='cpu'):
        X, y = X.to(device), y.to(device)
        loss_fn = nn.CrossEntropyLoss()
        logits = self.forward(X)
        return loss_fn(logits, y)
    
    def representation_seq(self, X_seq):
        results = self.representation(X_seq[:,0,]).clone().unsqueeze(1).repeat_interleave(X_seq.shape[1], 1)
        for t in range(1, X_seq.shape[1]):
            results[:,t,] = self.representation(X_seq[:,t,])
        return results.detach()