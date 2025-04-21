import numpy as np
import utils as ut

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_impute import AutoEncoder

class TaskPolicy:
    def __init__(self, P, Q, **kwargs):
        self.P, self.Q = P, Q
    
        #each regime takes in X, W, and prev_masks
        # Return the sampling probability of each task
        self.regime = [None for _ in range(Q)] 
        
    def select_task(self, X, W, seed=1):
        task_index = np.ones((X.shape[0], self.Q)) * np.nan
        masks = np.zeros((X.shape[0], self.Q+1, self.Q))
        for q in range(1, self.Q+1):
            task_index[:, q-1], masks[:, q, :] = self.select_task_q(X, W, masks[:, :q, :], seed=seed)
        return task_index, masks
                
    def select_task_q(self, X, W, prev_masks, seed=1, **kwargs):
        
        n_sample, q = X.shape[0], prev_masks.shape[1] - 1
        task_index = np.ones(n_sample) * np.nan
        
        # Check the remaining tasks
        availability = ut.check_task_availability(prev_masks)
        idx = (availability == 1).any(1)
        
        task_probs = self.regime[q](X[idx], W[idx], prev_masks[idx], seed=seed) * availability[idx]
        task_index[idx] = task_probs.argmax(1)
        
        # task_index[idx] = self.regime[q](X[idx], W[idx], prev_masks[idx])
        mask = ut.convert_task_index_to_mask(task_index, prev_masks[:,-1,:])
        return task_index, mask
    
    def train_model(self, X, W, M, A, R, y, **kwargs):
        pass

    def eval_model(self, X, W, all_masks):
        raise NotImplementedError("Subclasses must implement eval_model method")
    
#############################################################################
# Optimal Policy
#############################################################################

class OptimalPolicy(TaskPolicy):
    def __init__(self, P, Q, optimal_policy):
        super().__init__(P, Q)
        self.regime = optimal_policy
        
#############################################################################
# Random Policy
#############################################################################

class RandomPolicy(TaskPolicy):
    def __init__(self, P, Q):
        super().__init__(P, Q)

        def random_proba(X, W, prev_masks, seed=1):
            np.random.seed(seed)
            return np.random.dirichlet(np.ones(Q + 1), size=X.shape[0])

        self.regime = [
            lambda X, W, prev_masks, seed=1: random_proba(X, W, prev_masks, seed)
            for _ in range(Q)
        ]
        

#############################################################################
# Neural Network Policy - DTR
#############################################################################

class DTRPolicy(TaskPolicy):
    def __init__(self, modelClass, P, Q, hidden_dim_list, image=False, device='cpu', seed=1, **kwargs):
        super().__init__(P, Q)            
        self.model = modelClass(P, Q, hidden_dim_list, image, device, seed, **kwargs)
        
        def regime(X, W, prev_masks, seed=1):
            self.model.eval()
            
            X, W = torch.Tensor(X).to(device), torch.Tensor(W).to(device)
            prev_masks = torch.Tensor(prev_masks).to(device)
            
            masks = prev_masks
            left_over_step = self.Q-prev_masks.shape[1]
            if left_over_step > 0:
                makeup_masks = prev_masks[:,[-1],:].repeat_interleave(left_over_step, 1)
                masks = torch.cat([masks, makeup_masks], 1)
                            
            probs = F.gumbel_softmax(self.model(X, W, masks), tau=1, hard=False).detach().cpu().numpy()
            # probs = F.softmax(self.model(X, W, masks), dim=2).detach().cpu().numpy()
            # print("Probs:  ", probs[:,prev_masks.shape[1]-1,:].mean(0))
            self.model.train()
            return probs[:,prev_masks.shape[1]-1,:]
        
        self.regime = [regime for _ in range(Q)]
        self.device = device
        

    def train_model(self, X, W, M, A, R, y, **kwargs):
        self.model.train()
        ut.train_nn(self.model, (X, W, M, A, R), device=self.device, **kwargs)


class DTRLearning(nn.Module):
    def __init__(self,P, Q, hidden_dim_list, image=False, device='cpu', seed=1):
        super(DTRLearning, self).__init__()
        ut.seed_torch(seed)
        self.P, self.Q = P, Q
        self.image = image
        
        if not self.image:
            self.model_list = [ut.TLinear(Q, P+Q, hidden_dim_list[0])]
            for i in range(1, len(hidden_dim_list)):
                self.model_list.append(nn.LeakyReLU())
                self.model_list.append(nn.Dropout(0.1))
                self.model_list.append(ut.TLinear(Q, hidden_dim_list[i-1], hidden_dim_list[i]))
            self.model_list.append(ut.TLinear(Q, hidden_dim_list[-1], Q+1))
            self.dtr = nn.Sequential(*self.model_list)
            
        else:
            self.net =  nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2),  # [batch, 16, 12, 12]
                nn.Conv2d(16, 32, kernel_size=5, stride=2),  # [batch, 32, 4, 4]
                nn.Flatten(),  # [batch, 128*4*4]
                nn.Linear(32*4*4, 128)
            )
            self.last_layer = ut.TLinear(Q, 128, Q+1)
            self.dtr = lambda x_seq: self.last_layer(torch.stack([self.net(x_seq[:,i,]) for i in range(Q)], 1))
        
        self.device = device
    
    def forward(self, X, W, prev_masks):
        if self.image:
            return self._forward_image(X, W, prev_masks)
        else:
            return self._forward_tabular(X, W, prev_masks)
        
    def _forward_tabular(self, X, W, prev_masks):
        # Maybe project input_seq with LSTM?
        X_seq = X.unsqueeze(1).repeat_interleave(self.Q, 1)
        W_seq = W.unsqueeze(1).repeat_interleave(self.Q, 1)
        input_seq = torch.cat([X_seq, W_seq*prev_masks], 2)
        return self.dtr(input_seq)
    
    def _forward_image(self, X, W, prev_masks):
        xt_seq = (prev_masks.reshape(prev_masks.shape + (1,1,1)) * W.unsqueeze(1)).sum(2) + X.unsqueeze(1)
        return self.dtr(xt_seq)
    
    def loss(self, X, W, M, A, R, theta=1, **kwargs):
        raise NotImplementedError("Loss function must be implemented in the subclass")
            

class SALearning(DTRLearning):
    def __init__(self, P, Q, hidden_dim_list, image=False, device='cpu', seed=1):
        super().__init__(P, Q, hidden_dim_list, image, device, seed)
    
    # for training, all Q steps of M are given. Masks is of size Q. A is task index
    def loss(self, X, W, M, A, R, theta=1, device='cpu'):
        X, W, M, A, R = X.to(device).float(), W.to(device).float(), M.to(device).float(), A.to(device).float(), R.to(device).float()

        logits = self(X, W, M)
        # indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        if self.image:
            indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        else:
            indicators = (F.gumbel_softmax((logits).clip(-50, 50), tau=1, hard=False) * ((A+1)/2)).sum(-1)
        weighted_reward = (R * indicators.mean(1)).mean()
        return -weighted_reward


class OWLearning(DTRLearning):
    def __init__(self, P, Q, hidden_dim_list, image=False, device='cpu', seed=1):
        super().__init__(P, Q, hidden_dim_list, image, device, seed)
    
    # for training, all Q steps of M are given. Masks is of size Q. A is task index
    def loss(self, X, W, M, A, R, theta=1, device='cpu'):
        X, W, M, A, R = X.to(device).float(), W.to(device).float(), M.to(device).float(), A.to(device).float(), R.to(device).float()

        logits = self(X, W, M)
        if self.image:
            indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        else:
            indicators = (F.gumbel_softmax((logits).clip(-50, 50), tau=1, hard=False) * ((A+1)/2)).sum(-1)
        # indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        weighted_reward = (R * indicators.prod(1)).mean()
        return -weighted_reward


class StreakLearning(DTRLearning):
    def __init__(self, P, Q, hidden_dim_list, image=False, device='cpu', seed=1):
        super().__init__(P, Q, hidden_dim_list, image, device, seed)
    
    # for training, all Q steps of M are given. Masks is of size Q. A is task index
    def loss(self, X, W, M, A, R, theta=5, device='cpu'):
        X, W, M, A, R = X.to(device).float(), W.to(device).float(), M.to(device).float(), A.to(device).float(), R.to(device).float()
        
        logits = self(X, W, M)
        if self.image:
            indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        else:
            indicators = ((F.gumbel_softmax((logits).clip(-50, 50), tau=1, hard=False) * ((A+1)/2))).sum(-1)
        # predicted_actions = F.gumbel_softmax(logits, tau=1, hard=False) / 2 - 1
        # indicators = (1 / (torch.exp(predicted_actions) + 1)).prod(2)
        # print("indicators: ", indicators[0,])
        # print("logits: ", logits * A)
        # indicators = (1 / (torch.exp(torch.clip(-theta * A * logits, -50, 50)) + 1)).prod(2)
        survival_length = ((A+1)/2).sum(2).sum(1)
        
        survival_prob = torch.zeros(self.Q+1)
        lengths, counts = torch.unique(survival_length, return_counts=True)
        survival_prob[lengths.long()] = counts / counts.sum()
        
        weighted_reward = 0
        # Deal with only one stages
        for Tbar in range(1, 2):
            idx = (survival_length == Tbar)
            if idx.sum() > 0:
                # print("At Tbar = ", Tbar, "Averaged reward", R[idx,].mean(), "weighted reward: ", (R[idx,] * indicators[idx,:Tbar].prod(1)).mean())
                weighted_reward += (R[idx,] * indicators[idx,:Tbar].prod(1)).mean() * survival_prob[Tbar] * Tbar
        
        # For multiple treatment sessions
        for Tbar in range(2, self.Q+1):
            idx = (survival_length == Tbar)
            if idx.sum() > 0:
                pending_reward = 0
                pending_reward += (1 / (Tbar+1))**0.8 * indicators[idx,:Tbar].mean(1)
                for t in range(1, Tbar):
                    pending_reward += ((t+1) / (Tbar+1))**0.8  * indicators[idx,:t].prod(1) * indicators[idx,t:Tbar].mean(1)
                pending_reward += indicators[idx,:Tbar].prod(1)

                weighted_reward +=  (R[idx,] * pending_reward).mean() * survival_prob[Tbar] * Tbar
                # print("At Tbar = ", Tbar, "Averaged reward", R[idx,].mean(), "weighted reward: ", pending_reward.mean())
        
        return -weighted_reward.mean()


#############################################################################
# Feature Acquisition Policy  - POMDP
# This POMDP trains a feature acquisition policy and a prediction policy at the same time
#############################################################################
class FeatureAcquisitionPOMPDP(TaskPolicy):
    def __init__(self, P, Q, hidden_dim_list, classification=False, image=False, device='cpu', seed=1, **kwargs):
        super().__init__(P, Q)            
        self.model = POMPDPQLearning(P, Q, hidden_dim_list, classification, image, device, seed, **kwargs).to(device)
        self.classification = classification
        self.image = image
        
        def regime(X, W, prev_masks, seed=1):
            self.model.eval()
            X, W = torch.Tensor(X).to(device), torch.Tensor(W).to(device)
            prev_masks = torch.Tensor(prev_masks).to(device).sum(1)
            actions, q_values = self.model.acquire_feature(X, W, prev_masks)
            self.model.train()
            return F.softmax(q_values, 1).detach().cpu().numpy()
        
        self.regime = [regime for _ in range(Q)]
        self.device = device
        
    def predict(self, X, W, M, return_uncertainty=False):
        if self.classification:
            return self._predict_class(X, W, M, return_uncertainty)
        else:
            return self._predict_reg(X, W, M, return_uncertainty)
        
    def _predict_class(self, X, W, M, return_uncertainty=False):
        X, W, M = torch.Tensor(X).to(self.device), torch.Tensor(W).to(self.device), torch.Tensor(M).to(self.device)
        self.model.eval()
        probs = F.softmax(self.model.task_policy(self.model.compute_belief(X, W, M.sum(1))), dim=-1).detach().cpu().numpy()
        self.model.train()
        if not return_uncertainty:
            return probs.argmax(-1), None
        else:
            entropy = np.sum(-probs * np.log(probs + 1e-10), axis=1)
            return probs.argmax(-1), entropy
        
    def _predict_reg(self, X, W, M, return_uncertainty=False):
        X, W, M = torch.Tensor(X).to(self.device), torch.Tensor(W).to(self.device), torch.Tensor(M).to(self.device)
        if not return_uncertainty:
            self.model.eval()
            preds = self.model.task_policy(self.model.compute_belief(X, W, M.sum(1)))
            self.model.train()
            return preds.detach().cpu().numpy().flatten(), None
        else:
            preds = torch.stack([self.model.task_policy(self.model.compute_belief(X, W, M.sum(1))) for _ in range(30)], dim=0)
            return preds.mean(0).detach().cpu().numpy().flatten(), preds.std(0).detach().cpu().numpy().flatten()
        
    def train_model(self, X, W, M, y, **kwargs):
        self.model.train()
        ut.train_nn(self.model, (X, W, M, y), device=self.device, loss_func=self.model.loss_model, **kwargs)

    def train_policy(self, X, W, M, y, **kwargs):
        self.model.train()
        ut.train_nn(self.model, (X, W, M, y), device=self.device, loss_func=self.model.loss_policy, **kwargs)
        

class POMPDPQLearning(nn.Module):
    def __init__(self, P, Q, hidden_dim, classification=False, image=False, device='cpu', seed=1):
        super().__init__()
        ut.seed_torch(seed)
        self.P, self.Q = P, Q
        self.classification = classification
        self.image = image
        
        if not self.image:
            self.belief_net = nn.Linear(P+Q, P, bias=True)
            self.task_policy = nn.Sequential(*[nn.Linear(P, hidden_dim), nn.Dropout(0.1), nn.Linear(hidden_dim, 1)])
            self.q_net = nn.Linear(P, Q+1)
            
        else:
            self.belief_net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2),  # [batch, 16, 12, 12]
                nn.Conv2d(16, 32, kernel_size=5, stride=2),  # [batch, 32, 4, 4]
                nn.Flatten(),  # [batch, 128*4*4]
                nn.Linear(32*4*4, 128)
            )
            self.task_policy = nn.Linear(128, 10) # 10 classes
            self.q_net = nn.Linear(128, Q+1)
            
        if self.classification:
            self.prediction_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.prediction_loss = nn.MSELoss(reduction='none')
            
        self.q_value_loss = nn.MSELoss()
        self.device = device
        
    def compute_belief(self, X, W, mask):
        if not self.image:
            belief = torch.cat([X, W*mask], 1)
        else:
            belief = (mask.reshape(mask.shape + (1,1,1)) * W).sum(1) + X
        return self.belief_net(belief)
    
    
    def information_gain(self, X, W, prev_mask, curr_mask, y):
        past_belief = self.compute_belief(X, W, prev_mask)
        curr_belief = self.compute_belief(X, W, curr_mask)
        if not self.classification:
            prev_loss = self.prediction_loss(self.task_policy(past_belief), y)
            curr_loss = self.prediction_loss(self.task_policy(curr_belief), y)
        else:
            prev_loss = self.prediction_loss(self.task_policy(past_belief), y.squeeze(1))
            curr_loss = self.prediction_loss(self.task_policy(curr_belief), y.squeeze(1))
        return prev_loss - curr_loss
    
    def acquire_feature(self, X, W, prev_mask):
        belief = self.compute_belief(X, W, prev_mask)
        q_values = self.q_net(belief) - torch.cat([prev_mask, torch.zeros(prev_mask.shape[0], 1)], 1) * 1e10  
        actions = q_values.argmax(-1)
        return actions, q_values        
    
    def loss_model(self, X, W, M, y, device='cpu', **kwargs):
        X, W, M, y = X.to(device).float(), W.to(device).float(), M.to(device).float(), y.to(device).reshape(-1,1)
        zero_mask = torch.zeros_like(M[:,0,]).float().to(W.device)
        # preds = self.task_policy(self.compute_belief(X, W, M[:,-1,]))
        preds = self.task_policy(self.compute_belief(X, W, zero_mask))
        if not self.classification:
            return nn.MSELoss()(preds, y.float())
        else:
            return nn.CrossEntropyLoss()(preds, y.long().squeeze(1))
    
    def loss_policy(self, X, W, M, y, device='cpu', **kwargs):
                
        X, W, M, y = X.to(device).float(), W.to(device).float(), M.to(device).float(), y.to(device).reshape(-1,1)
        y = y.float() if not self.classification else y.long()
        
        active = torch.ones(X.shape[0]).bool()
        masks = torch.zeros(X.shape[0], self.Q+1, self.Q+1)

        loss = 0
        reward = torch.zeros((self.Q, X.shape[0])).to(device)
        
        t = 1
        while t < self.Q+1 and active.any():
            # if the feature is not acquired, the mask is still 0 (use imputed values)
            belief = self.compute_belief(X, W, masks[:,:t,:-1].sum(1) * M[:,:t,:].sum(1))
            q_values = self.q_net(belief)
            available_actions = 1 - masks[:,:t,:].sum(1)
            q_values = q_values + (available_actions-1) * 1e10   
            actions = q_values.argmax(-1)
        
            # Process stop action
            stop_mask = (actions == self.Q) & active
            if stop_mask.any():
                y_pred_stop = self.task_policy(self.compute_belief(X, W, masks[:,:t,:-1].sum(1) * M[:,:t,:].sum(1)))
                if not self.classification:
                    reward[t-1, stop_mask] = -self.prediction_loss(y_pred_stop[stop_mask], y[stop_mask])[:,0]
                else:
                    reward[t-1, stop_mask] = -self.prediction_loss(y_pred_stop[stop_mask], y[stop_mask].squeeze(1))
                active[stop_mask] = False
            
            # Process feature acquisition
            acquire_mask = (actions < self.Q) & active
            if acquire_mask.any():
                feature_idx = actions[acquire_mask]
                batch_idx = torch.arange(X.shape[0])[acquire_mask]
                masks[batch_idx, t, feature_idx] = 1
            
                # Reward for acquisition
                if not self.classification:
                    reward[t-1, acquire_mask] = self.information_gain(
                        X, W, masks[:,:t,:-1].sum(1) * M[:,:t,:].sum(1), masks[:,:t+1,:-1].sum(1) * M[:,:t,:].sum(1), y)[acquire_mask,0]
                else:
                    reward[t-1, acquire_mask] = self.information_gain(
                        X, W, masks[:,:t,:-1].sum(1) * M[:,:t,:].sum(1), masks[:,:t+1,:-1].sum(1) * M[:,:t,:].sum(1), y)[acquire_mask]
                
            # Update Q-values
            idx = stop_mask | acquire_mask
            next_belief = self.compute_belief(X[idx,], W[idx,], masks[idx,:t+1,:-1].sum(1) * M[idx,:t+1,:].sum(1))
            q_target = reward[t-1, idx] + 0.99 * self.q_net(next_belief).max(dim=-1).values
            q_pred = q_values[idx, actions[idx]]
            loss += self.q_value_loss(q_pred, q_target)
            
            t += 1

        return loss
