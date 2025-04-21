
import utils as ut

import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from collections import Counter

import torch
import torch.nn as nn

def AL_learning(f, policy, optimal_policy, costModel, labeled_data, unlabeled_data, test_data, Budget, batch_size, select_criteria, mask_action=None,
                g=None, f_params={}, g_params={'threshold': 0.3}, h_params={}, policy_params={}):
    
    results = defaultdict(list)
    
    XL, WL, yL, tiL, ML = ut.deep_copy(labeled_data)
    XU, WU, yU, tiU, MU = ut.deep_copy(unlabeled_data)
    Xte, Wte, yte, tiTe, Mte = ut.deep_copy(test_data)
    
    numLabeled = 0
    
    # prev_scores = np.ones(XL.shape[0])
    X_new = ut.deep_copy(XL)
    propen = ML[:, -1, :].mean(0, keepdims=True).repeat(XL.shape[0], 0).clip(1e-3,)
    # propen = propen / propen.sum(1, keepdims=True)
    cost = costModel(XL, WL, ML[:,:-1,:], optimal_policy.select_task(XL, WL)[1])
    results['cost'] = [cost[0]]
    # results['cost_detailed'] = [cost[1]]
    
    while numLabeled < Budget and XU.shape[0] > 0:
        try:
            _, prev_scores = f.predict(X_new, return_uncertainty=True)
        except:
            prev_scores = np.ones(X_new.shape[0])            
        
        # Surrogate Training
        if g is not None: 
            # Employ Surrogate Training
            if g.impute_model and (numLabeled // batch_size) % 2 == 0:
                print(f"Updating imputation model for surrogates at number {numLabeled}")
                g.impute_model.train_model(XL, WL, ML[:,-1,:], propen=propen, mask_action = mask_action, **h_params)
                # results['h'].append(g.impute_model.eval_model(Xte, Wte))
                W_pred = g.impute_model.predict(Xte, W=Wte, mask_action=mask_action)
                if mask_action is None:
                    results['h'].append({'MSE-M': (((W_pred - Wte)*Mte[:,-1,:])**2).mean(), 'MSE': ((W_pred - Wte)**2).mean()})
            
            g.train_model(XL, WL, ML, yL, mask_action=mask_action, **g_params)
            ypred = g.predict(Xte, y=yte, W=Wte, mask_action=mask_action, return_uncertainty=False)
            # results['g'].append({'MSE': ((ypred - yte)**2).mean()})
            results['g'].append(g.eval_model(Xte, yte, W=Wte, mask_action=mask_action)) # provide W for GroundTrueSurrogateModel
    
            # Generate pseudo-labels and surrogate training
            y_pseudo, uncertainty = g.predict(XU, W=WU, y=yU, mask_action=mask_action, return_uncertainty=True)
            certain_idx = np.where(uncertainty <= np.quantile(uncertainty, g_params['threshold']))[0]
            print("The MSE of the surrogate selected instances: ", ((y_pseudo[certain_idx] - yU[certain_idx])**2).mean(), " with size ", len(certain_idx))
            f.train_model(np.vstack((XL, XU[certain_idx])), np.hstack((yL, y_pseudo[certain_idx])), **f_params)
        else:
            f.train_model(XL, yL, **f_params)
            
        _, after_scores = f.predict(X_new, return_uncertainty=True)
        
        results['numLabeled'].append(XL.shape[0])
        results['f'].append(f.eval_model(Xte, yte))
        print("Iteration-results: ", results['f'][-1])
        results['reward'].append((prev_scores - after_scores) - (results['cost'][-1] ))
        # print("reward: ", results['reward'][-1])

        # Select Uncertain instances
        idxSel = select_criteria(f, XU, batch_size, labeled_data=(XL, np.hstack(results['cost'])))
        X_new = ut.deep_copy(XU[idxSel])

        # Select Tasks
        if policy is None:
            M_select = np.zeros((XU[idxSel].shape[0], ML.shape[1], ML.shape[2]))
            ti_select = np.ones((len(idxSel), tiL.shape[1])) * np.nan
            propen = np.vstack([propen, propen[[0],:].repeat(len(idxSel), 0)])
        else:
            if (numLabeled // batch_size) % 5 == 0:
                print(f"Training policy at number {numLabeled}")
                policy.train_model(XL, WL, ML[:,:-1,:], ut.convert_task_index_to_action(tiL), np.hstack(results['reward']), yL, **policy_params)
                # results['cost_test'].append(costModel(Xte, Wte, policy.select_task(Xte, Wte)[1], optimal_policy.select_task(Xte, Wte)[1])[1])

            ti_select, M_select = policy.select_task(XU[idxSel], WU[idxSel])
            # new_propen = policy.regime[0](XU[idxSel], WU[idxSel], np.zeros_like(MU[idxSel,:1,:]))[:,:-1]
            # new_propen = new_propen / new_propen.sum(1, keepdims=True)
            new_propen = MU[idxSel,-1,:].mean(0, keepdims=True).repeat(len(idxSel), 0).clip(1e-3, )
            propen = np.vstack([propen, new_propen])
            print("Decide to do task at step 1:  ", Counter(ti_select[:,0]))
                                                                
        XL, yL, WL = np.vstack((XL, XU[idxSel])), np.hstack((yL, yU[idxSel])), np.vstack((WL, WU[idxSel]))
        numLabeled += len(idxSel)
        
        tiL, ML = np.vstack((tiL, ti_select)), np.vstack((ML, M_select))
        cost_select = costModel(XU[idxSel], WU[idxSel], M_select, optimal_policy.select_task(XU[idxSel], WU[idxSel])[1])
        results['cost'].append(cost_select[0])
        results['cost_detailed'].append({k: v.mean() for k, v in cost_select[1].items()})
            
        # Remove instances from the unlabeled set
        mask = torch.ones(XU.shape[0], dtype=bool)
        mask[idxSel] = False
        XU, yU, WU, MU = XU[mask], yU[mask], WU[mask], MU[mask]

    return format_dict(dict(results))


def format_dict(results):
    for model in ['f', 'g', 'h']:
        if model in results:
            list_of_dicts = results.pop(model)
            new_dict = defaultdict(list)
            for d in list_of_dicts:
                for k, v in d.items():
                    new_dict[k].append(v)
            for k in new_dict:
                new_dict[k] = np.array(new_dict[k])
            results[model] = dict(new_dict)
    for k in results:
        if k == 'cost':
            results[k] = np.array([c.sum() for c in results[k]])[1:]
        elif k == 'reward':
            results[k] = np.hstack(results[k])
        elif isinstance(results[k], list):
            results[k] = np.array(results[k])
        else:
            pass
    return results


######################################################
# Uncertainty Selection Criteria
######################################################
# def EnsembleRegressorUncertainty(f, X, batch_size, **kwargs):
#     assert isinstance(f.model, RandomForestRegressor), "f must be a RandomForestRegressor"
#     std = np.vstack([tree.predict(X) for tree in f.model.estimators_]).T.std(1)
#     return np.argsort(std)[-batch_size:]

def Uncertainty(f, X, batch_size, **kwargs):
    preds, uncertainty = f.predict(X, return_uncertainty=True)
    return np.argsort(uncertainty)[-batch_size:]

def CostSensitiveUncertainty(f, X, batch_size, labeled_data, **kwargs):
    XL, costL = labeled_data
    lr = LinearRegression()
    lr.fit(XL, costL)
    est_cost = lr.predict(X)
    preds, uncertainty = f.predict(X, return_uncertainty=True)
    scores = uncertainty / (est_cost + 1e-3)
    return np.argsort(scores)[-batch_size:]

def ImageCostSensitiveUncertainty(f, X, batch_size, labeled_data, **kwargs):
    XL, costL = labeled_data
    lr = LinearRegression()
    
    feature_extractor = nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.MaxPool2d(2, 2),
        nn.Flatten()
    )
    XL_feat = feature_extractor(torch.Tensor(XL)).numpy()
    lr.fit(XL_feat, costL)
    
    X_feat = feature_extractor(torch.Tensor(X)).numpy()
    est_cost = lr.predict(X_feat)
    preds, uncertainty = f.predict(X, return_uncertainty=True)
    scores = uncertainty - est_cost
    return np.argsort(scores)[-batch_size:]