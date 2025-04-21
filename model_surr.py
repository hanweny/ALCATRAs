import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RandomForestModel, NeuralNetworkModel, ImageClassificationModel


class SurrogateModelClass:
    """
    Base model class that defines the interface for training, prediction, and evaluation.
    This class is designed to be subclassed by specific model implementations.
    """
    
    def __init__(self, ImputeModel, classification=False):
        self.model = None
        self.classification = classification
        self.impute_model = ImputeModel
    
    def train_model(self, X, W, mask, y, **kwargs):
        raise NotImplementedError("Subclasses must implement train_model method")
    
    def predict(self, X, **kwargs):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def eval_model(self, X, y, **kwargs):
        result_dict = {}
        if self.classification:
            acc = (self.predict(X, **kwargs, y=y) == y).mean()
            result_dict['acc'] = acc
        else:
            mse = np.mean((self.predict(X, **kwargs, y=y) - y) ** 2)
            result_dict['MSE'] = mse
        return result_dict
    
    
# Return the ground-true y for the unlabeled set (usually is unknown)
class GroundTrueSurrgateModel(SurrogateModelClass):
    def __init__(self, ImputeModel, classification=False, **kwargs):
        super().__init__(ImputeModel, classification)
        self.model = None

    def train_model(self, X, W, mask, y, **kwargs):
        pass
    
    def predict(self, X, **kwargs):
        assert "y" in kwargs, "y is required for GroundTrueModel"
        return kwargs["y"], np.zeros(X.shape[0])
    
# Classic surrogate model, using X->y yhat as the surrogate labels    
class XSurrogateRandomForestModel(SurrogateModelClass):
    def __init__(self, ImputeModel, classification=False, **kwargs):
        super().__init__(ImputeModel, classification)
        self.model = RandomForestModel(classification=self.classification, **kwargs)
    
    def train_model(self, X, W, mask, y, **kwargs):
        self.model.train_model(X, y, **kwargs)
        
    def predict(self, X, return_uncertainty=False, **kwargs):
        return self.model.predict(X, return_uncertainty) 
        
# Surrogate model with task information, using (X,W)->yhat as the surrogate labels
class XWSurrogateRandomForestModel(SurrogateModelClass):
    def __init__(self, ImputeModel, classification=False, **kwargs):
        super().__init__(ImputeModel, classification)
        self.model = RandomForestModel(classification=self.classification, **kwargs)
    
    def train_model(self, X, W, mask, y, **kwargs):
        W_tilde = self.impute_model.predict(X, W=W, mask=mask, **kwargs)
        W_pseudo = W * mask[:,-1,:] + W_tilde * (1 - mask[:,-1,:])
        self.model.train_model(np.hstack((X, W_pseudo)), y, **kwargs)
    
    def predict(self, X, return_uncertainty=False, **kwargs):
        W_tilde = self.impute_model.predict(X, **kwargs)
        new_X = np.hstack((X, W_tilde))
        return self.model.predict(new_X, return_uncertainty)
    

class XWNeuralNetworkSurrogateModel(SurrogateModelClass):
    def __init__(self, ImputeModel, classification=False, **kwargs):
        super().__init__(ImputeModel, classification)
        self.model = NeuralNetworkModel(classification=self.classification, **kwargs)
        
    def train_model(self, X, W, mask, y, **kwargs):
        W_tilde = self.impute_model.predict(X, W=W, mask=mask, **kwargs)
        W_pseudo = W * mask[:,-1,:] + W_tilde * (1 - mask[:,-1,:])
        self.model.train_model(np.hstack((X, W_pseudo)), y, **kwargs)
        
    def predict(self, X, return_uncertainty=False, **kwargs):
        W_tilde = self.impute_model.predict(X, **kwargs)
        new_X = np.hstack((X, W_tilde))
        return self.model.predict(new_X, return_uncertainty)


########################################################
# Image Surrogate Model
########################################################
class ImageSurrogateModel(SurrogateModelClass):
    def __init__(self, ImputeModel, **kwargs):
        super().__init__(ImputeModel, True)
        self.model = ImageClassificationModel(**kwargs)

    def train_model(self, X, W, mask, y, mask_action=None, **kwargs):
        if self.impute_model is None:
            self.model.train_model(X, y, **kwargs)
            return
        
        if mask_action is None:
            raise ValueError("mask_action is required for ImageSurrogateModel")
        Xhat = self.impute_model.predict(X, W=W, mask=mask, **kwargs)
        new_X = Xhat * mask_action.sum(0) + X * (1 - mask_action.sum(0))
        self.model.train_model(new_X, y, **kwargs)
        
    def predict(self, X, mask_action=None, **kwargs):
        if self.impute_model is None:
            return self.model.predict(X, **kwargs)
        
        if mask_action is None:
            raise ValueError("mask_action is required for ImageSurrogateModel")
        Xhat = self.impute_model.predict(X, **kwargs)
        new_X = Xhat * mask_action.sum(0) + X * (1 - mask_action.sum(0))
        return self.model.predict(new_X, **kwargs)