
import utils as ut

import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

def reinit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            

def POMPDP_learning(fa, optimal_policy, costModel, labeled_data, unlabeled_data, test_data, Budget, batch_size, f_params={}, policy_params={}):
    
    results = defaultdict(list)
    
    XL, WL, yL, tiL, ML = ut.deep_copy(labeled_data)
    XU, WU, yU, tiU, MU = ut.deep_copy(unlabeled_data)
    Xte, Wte, yte, tiTe, Mte = ut.deep_copy(test_data)
    
    numLabeled = 0
    
    cost = costModel(XL, WL, ML[:,:-1,:], optimal_policy.select_task(XL, WL)[1])
    results['cost'] = [cost[0]]
    results['cost_detailed'] = [cost[1]]
    
    while numLabeled < Budget and XU.shape[0] > 0:
        
        # fa.model.task_policy.apply(reinit_weights)
        
        fa.train_model(XL, WL, ML, yL, **f_params)
            
        results['numLabeled'].append(XL.shape[0])
        
        preds, _ = fa.predict(Xte, np.zeros_like(Wte), np.zeros_like(Mte), return_uncertainty=False)
        if fa.classification:
            print("Preds: ", preds, "yte: ", yte, "acc: ", (preds == yte).mean())
            results['f'].append({'acc': (preds == yte).mean()})
        else:
            results['f'].append({'MSE': ((preds - yte)**2).mean()})
        
        # Select Uncertain instances
        preds, scores = fa.predict(XU, np.zeros_like(WU), np.zeros_like(MU), return_uncertainty=True)
        idxSel = scores.argsort()[-batch_size:]

        if (numLabeled // batch_size) % 5 == 0:
            fa.train_policy(XL, WL, ML, yL, **policy_params)
            
        ti_select, M_select = fa.select_task(XU[idxSel], WU[idxSel])
        print("Decide to do task:  ", Counter(ti_select[:,0]))

        XL, yL, WL = np.vstack((XL, XU[idxSel])), np.hstack((yL, yU[idxSel])), np.vstack((WL, WU[idxSel]))
        numLabeled += len(idxSel)
        
        tiL, ML = np.vstack((tiL, ti_select)), np.vstack((ML, M_select))
        cost_select = costModel(XU[idxSel], WU[idxSel], M_select, optimal_policy.select_task(XU[idxSel], WU[idxSel])[1])
        results['cost'].append(cost_select[0])
        results['cost_detailed'].append(cost_select[1])
        results['cost_test'].append(costModel(Xte, Wte, fa.select_task(Xte, Wte)[1], optimal_policy.select_task(Xte, Wte)[1])[1])
        
        belief = fa.model.compute_belief(torch.Tensor(Xte), torch.Tensor(Wte), torch.zeros(Xte.shape[0], Mte.shape[-1]))
        y_pred = fa.model.task_policy(belief).detach().cpu().numpy()
        if fa.classification:
            results['g'].append({'acc': (y_pred.argmax(1) == yte).mean()})
        else:
            results['g'].append({'MSE': ((y_pred.flatten() - yte)**2).mean()})
        
        # belief = fa.model.belief_net(torch.Tensor(Xte)).detach().cpu().numpy()
        # results['h'].append({'MSE-M': (((belief - Wte)*Mte[:,-1,:])**2).mean(), 
        #                      'MSE': (((belief - Wte))**2).mean()})

        # Remove instances from the unlabeled set
        mask = torch.ones(XU.shape[0], dtype=bool)
        mask[idxSel] = False
        XU, yU, WU, MU = XU[mask], yU[mask], WU[mask], MU[mask]
    
        if len(results['numLabeled']) % 5 == 0:
            print(f"Iteration: {len(results['numLabeled'])}, numLabeled: {XL.shape[0]}, Loss: {results['f'][-1]}")

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
        if isinstance(results[k], list) and k != 'cost':
            results[k] = np.array(results[k])
        if k == 'cost':
            results[k] = np.array([c.sum() for c in results[k]])[1:]
    return results