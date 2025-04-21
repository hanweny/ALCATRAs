
import AL
import POMPDP
import model
import model_surr
import model_impute
import model_policy

import utils as ut
import simulation as sim

import pickle as pkl
import sys
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


result = dict()

TASK_COSTS, COSTS_PARAMS = None, None
def FashionMNISTCostModel(X, W, M, optM, task_costs=None):
    global TASK_COSTS, COSTS_PARAMS
    
    feature_extractor = nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.MaxPool2d(2, 2),
        nn.Flatten()
    )
    if COSTS_PARAMS is None:
        costs, COSTS_PARAMS = ut.function_transform(feature_extractor(torch.Tensor(X)).numpy(), lo=15, hi=None, noise=0, seed=1, nonlinear=False)
    
    n_tasks = M.shape[-1]
    TASK_COSTS = np.ones(n_tasks) / n_tasks  / 2 if task_costs is None else task_costs

    # base costs
    base_cost = ut.sigmoid(ut.apply_saved_function_transform(feature_extractor(torch.Tensor(X)).numpy(), COSTS_PARAMS))
    task_cost = (M[:, -1, :] * TASK_COSTS).sum(1)
    return base_cost + task_cost, {'base': base_cost, 'task': task_cost, 'num': np.array(X.shape[0])}

def load_data(seed=1):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    ut.seed_torch(seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_size = 10000 // 64

    def load_data(loader, size, device='cpu'):
        X, y = next(iter(loader))
        for i in range(size-1):
            X_, y_ = next(iter(train_loader))
            X = torch.cat((X, X_), dim=0)
            y = torch.cat((y, y_), dim=0)
        return X.to(device), y.to(device)

    image, label = load_data(train_loader, train_size, device)
    
    mask = torch.zeros_like(image[0,], dtype=torch.long)

    num_actions = 4
    mask_action = mask.clone().unsqueeze(0).repeat_interleave(num_actions, 0)

    mask_action[0, :, 2:14, 2:14] = 1
    mask_action[1, :, 2:14, 14:26] = 1
    mask_action[2, :, 14:26, 2:14] = 1
    mask_action[3, :, 14:26, 14:26] = 1

    mask_region = mask_action.sum(0)

    X = image * (1-mask_region)
    W = mask_action * image.unsqueeze(1)
    y = label
    
    X, W, y = X.detach().cpu().numpy(), W.detach().cpu().numpy(), y.detach().cpu().numpy()
    mask_action = mask_action.detach().cpu().numpy()
    
    return X, W, y, mask_action, num_actions

def run_simulation(nL, seed):
    
    start_time = time.time()
    
    X, W, y, mask_action, num_actions = load_data(seed)

    nU, nte, Budget, batch_size = 2000, 5000, 640, 16
    random_policy = model_policy.RandomPolicy(None, num_actions)
    ti, M = random_policy.select_task(X, W, seed=seed)
    
    Labeled_data, Unlabeled_data, Test_data = sim.split_data(X, W, y, M, ti, nL, nU, nte, seed)
    
    f_name = f"results/fashionMNIST/L{nL}_U{nU}_test{nte}_B{Budget}_seed{seed}.pkl"
    
    f_params = {'epochs': 50, 'lr': 5e-3, 'verbose': 30}
    
    # # Classic AL
    f = model.ImageClassificationModel(seed=1, device=device)
    result['AL: classic'] = AL.AL_learning(
        f, None, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty,
        f_params=f_params
    )
    print("Classic AL done", result['AL: classic']['f'])
    pkl.dump(result, open(f_name, 'wb'))

    # # Cost-Sensitive AL
    f = model.ImageClassificationModel(seed=1, device=device)
    result['AL: cost-sensitive'] = AL.AL_learning(
        f, None, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.ImageCostSensitiveUncertainty,
        f_params=f_params
    )
    print("Cost-Sensitive AL done", result['AL: cost-sensitive']['f'])
    pkl.dump(result, open(f_name, 'wb'))

    # # Surrogate learning to impute the surrogate y
    f = model.ImageClassificationModel(seed=1, device=device)
    g = model_surr.ImageSurrogateModel(None, seed=1, device=device)
    result['Surrogate-Learning'] = AL.AL_learning(
        f, None, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty, g=g,
        f_params={'epochs': 80, 'lr': 5e-3},
        g_params={'epochs': 50, 'lr': 5e-3, 'threshold': 0.1}
    )
    print("Surrogate Learning done", result['Surrogate-Learning']['f'])
    pkl.dump(result, open(f_name, 'wb'))

    # # Feature Acquisition Policy    
    fa = model_policy.FeatureAcquisitionPOMPDP((1, 28, 28), num_actions, None, classification=True, image=True, device=device, seed=1)
    result['POMPDP'] = POMPDP.POMPDP_learning(
        fa, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, 
        f_params={'epochs': 100, 'lr': 8e-3}, policy_params={'epochs': 50, 'lr': 5e-3}
    )
    print("POMPDP done", result['POMPDP']['f'])
    pkl.dump(result, open(f_name, 'wb'))
    
    # ALCATRAz 
    f = model.ImageClassificationModel(seed=1, device=device)
    h = model_impute.ImageVariationalImpute(input_shape=(1, 28, 28), latent_dim=512, device=device) 
    g_h = model_surr.ImageSurrogateModel(h, seed=1, device=device)
    streak_policy = model_policy.DTRPolicy(model_policy.StreakLearning, (1, 28, 28), num_actions, None, image=True, seed=1)
    result['Proposed Framework: ALCATRAs Online'] = AL.AL_learning(
        f, streak_policy, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty, mask_action=mask_action, g=g_h,
        f_params={'epochs': 80, 'lr': 5e-3},
        g_params={'epochs': 50, 'lr': 5e-3, 'threshold': 0.5}, 
        h_params={'epochs': 200, 'lr': 8e-3}, 
        policy_params={'epochs': 100, 'lr': 8e-3}
    )
    print("ALCATRAs-online done", result['Proposed Framework: ALCATRAs Online']['f'])
    pkl.dump(result, open(f_name, 'wb'))
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


    # # ALCATRAz-Random
    # f = model.ImageClassificationModel(seed=1, device=device)
    # h = model_impute.ImageVariationalImpute(input_shape=(1, 28, 28), latent_dim=512, device=device) 
    # g_h = model_surr.ImageSurrogateModel(h, seed=1, device=device)
    # result['Proposed Framework: ALCATRAs-Random'] = AL.AL_learning(
    #     f, random_policy, random_policy, FashionMNISTCostModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty, mask_action=mask_action, g=g_h,
    #     f_params={'epochs': 80, 'lr': 5e-3},
    #     g_params={'epochs': 50, 'lr': 5e-3, 'threshold': 0.6}, 
    #     h_params={'epochs': 250, 'lr': 8e-3}, 
    #     policy_params={'epochs': 50, 'lr': 5e-3}
    # )
    # print("ALCATRAs-Random done", result['Proposed Framework: ALCATRAs-Random']['f'])
    
    # pkl.dump(result, open(f_name, 'wb'))
    
    return result

def main():
    nL = int(sys.argv[1])
    seed = int(sys.argv[2])
    run_simulation(nL, seed)
    
if __name__ == "__main__":
    main()