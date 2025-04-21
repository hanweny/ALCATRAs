import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

sigmoid = lambda x: 1/(1 + np.exp(-x))
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
softplus = lambda x: F.softplus(torch.Tensor(x)).numpy()
softmax_range = lambda x, a, b: a + (b-a) * F.sigmoid(torch.Tensor(x)).numpy()
normalize_squared = lambda x: x**2 / np.sum(x**2, axis=1, keepdims=True)

NONLINEAR_FUNC_BASIS = [lambda x: x**i for i in range(1, 4)] + [np.sign, np.arctan]

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True#s as reproducibility docs
        
def deep_copy(obj):
    if isinstance(obj, np.ndarray):
        return obj.copy()
    elif isinstance(obj, torch.Tensor):
        return obj.clone().to(obj.device)
    else:
        return copy.deepcopy(obj)
    

########################################################################################################
### Task Index and Onehot
########################################################################################################
def convert_task_index_to_mask(task_index, prev_mask):
    mask = np.zeros(prev_mask.shape)
    idx = (~np.isnan(task_index)) & (task_index < prev_mask.shape[1])
    mask[idx, np.array(task_index[idx]).astype(int)] = 1
    return mask + prev_mask

def convert_mask_to_task_index(mask):
    n_samples, n_step, n_tasks = mask.shape
    
    idx = np.ones((n_samples, n_tasks)) * np.nan

    for q in range(1, n_step):
        id, task_idx = np.where(mask[:,q,:] - mask[:,q-1,:] == 1)
        idx[id,q-1] = task_idx

    for i in range(idx.shape[0]):
        nan_idx = np.where(np.isnan(idx[i,:n_step-1]))[0]
        if len(nan_idx) > 0:
            idx[i, nan_idx[0]] = n_tasks
            
    return idx

def convert_task_index_to_onehot(task_index):
    Q = task_index.shape[1]
    onehot = np.zeros((task_index.shape[0], Q, Q))
    for q in range(Q):
        task_select = task_index[:,q]
        idx = (~np.isnan(task_select)) & (task_select < Q)
        onehot[idx,q,np.array(task_select[idx]).astype(int)] = 1
    return onehot
    
def convert_task_mask_to_onehot(task_mask):
    Q = task_mask.shape[2]
    onehot = np.zeros((task_mask.shape[0], Q, Q))
    for q in range(Q):
        onehot[:,q,:] = task_mask[:,q+1,:] - ((task_mask[:,:q+1,:Q].sum(1))>0).astype(int)
    return onehot

def convert_task_index_to_action(task_index):
    num_samples, num_tasks = task_index.shape
    A = np.ones((num_samples, num_tasks, num_tasks+1)) * -1
    for q in range(num_tasks):
        idx = ~np.isnan(task_index[:,q])
        A[idx, q, np.array(task_index[idx,q]).astype(int)] = 1
    return A

def check_task_availability(prev_masks):
    n_samples, n_step, n_tasks = prev_masks.shape
    completed_tasks = prev_masks.sum(1)
    
    availability = np.ones((n_samples, n_tasks+1))

    for q in range(n_tasks):
        availability[np.where(completed_tasks[:,q] > 0)[0], q] = 0

    availability[np.where((completed_tasks > 0).sum(1) < (n_step-1)), :] = 0
    
    return availability

def compare_task_masks(M, optM):
    n_samples = M.shape[0]
    est_task_index = convert_mask_to_task_index(M)
    opt_task_index = convert_mask_to_task_index(optM)
    
    n_streak, n_rest, n_length = np.zeros(n_samples, dtype=int), np.zeros(n_samples, dtype=int), np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        n_pred, n_opt = (~np.isnan(est_task_index[i,:])).sum(), (~np.isnan(opt_task_index[i,:])).sum()
        n_length[i] = max(n_pred, n_opt)
        
        arr_to_add = n_streak
        for j in range(n_length[i]):
            pred, opt = est_task_index[i,j], opt_task_index[i,j]
            if pred == opt or (np.isnan(pred) and np.isnan(opt)):
                arr_to_add[i] += 1
            else:
                arr_to_add = n_rest
    
    return n_streak, n_rest, n_length


def visualize_task_index(task_index):
    # Convert each row of task_index to a string
    task_index_strings = []
    for row in task_index:
        # Convert each element to string and join with commas
        row_string = ','.join([str(int(x)) for x in row if not np.isnan(x)])
        task_index_strings.append(row_string)
    return task_index_strings

########################################################################################################
# Function Generation
########################################################################################################
COST_PARAMS = None
def function_transform(X, lo=1, hi=None, noise=1, seed=1, output_dim=1, nonlinear=False, f_basis=NONLINEAR_FUNC_BASIS):
    if output_dim == 1:
        if nonlinear:
            return nonlinear_transform(X, lo, hi, noise, seed, f_basis)
        else:
            return linear_transform(X, lo, hi, noise, seed)
    else:
        return multidim_transform(X, output_dim, nonlinear, lo, hi, noise, seed, f_basis)
    

def apply_saved_function_transform(X, params, noise=None, seed=1):
    if type(params) == list:
        return apply_saved_multidim_transform(X, params, noise, seed)
    else:
        if params['nonlinear']:
            return apply_saved_nonlinear_transform(X, params, noise, seed)
        else:
            return apply_saved_linear_transform(X, params, noise, seed)
        
##############################################
## Nonlinear Function Generation
##############################################
## transformation = \sum_{i=1}^{k} \beta_i \prod_{j=1}^{m} f_j(X_{i,j}) + \epsilon
## k sampled from [lo, hi], m sampled from [1, 2] (interaction at most cosists of 2 features)
def nonlinear_transform(X, lo=1, hi=None, noise=0, seed=1, f_basis=NONLINEAR_FUNC_BASIS):
    np.random.seed(seed)
    n_samples, n_features = X.shape
    
    if hi is None:
        hi = n_features
    else:
        hi = min(hi, n_features)
        
    num_var = np.random.randint(lo, hi + 1)
    
    # Initialize storage for transformed variables and parameters
    transformed_vars = []
    transform_params = {
        'nonlinear': True,
        'num_var': num_var,
        'noise': noise,
        'f_basis': f_basis,
        'selected_features': [],
        'selected_functions': [],
        'coefficients': []
    }
    
    interaction_terms = np.ones((n_samples, num_var))
    for i in range(num_var):
        # For each variable, decide how many features to interact (1 means no interaction)
        interaction_size = np.random.randint(1, 3)
        
        # Select which features to use
        selected_features = np.random.choice(n_features, size=interaction_size, replace=False)
        
        # Select which function to apply to each feature
        selected_functions = np.random.choice(len(f_basis), size=interaction_size, replace=True)
        
        # Apply the selected functions to the selected features
        transformed_vars = np.hstack([f_basis[func_idx](X[:, feat]).reshape(-1,1) for feat, func_idx in zip(selected_features, selected_functions)])
        interaction_terms[:,i] = transformed_vars.prod(1)
        
        # Store parameters
        transform_params['selected_features'].append(selected_features.tolist())
        transform_params['selected_functions'].append(selected_functions.tolist())
    
    # Create linear combination of transformed variables
    coefficients = np.random.normal(0, 1, num_var)
    transform_params['coefficients'] = coefficients.tolist()
    
    # Combine all transformed variables using the coefficients
    transformed_X = (interaction_terms * coefficients).sum(1) + np.random.normal(0, noise, n_samples)
    
    return transformed_X, transform_params


def apply_saved_nonlinear_transform(X, transform_params, noise=None, seed=1):
    np.random.seed(seed)
    n_samples = X.shape[0]
    num_var = transform_params['num_var']
    
    interaction_terms = np.ones((n_samples, num_var))

    for i in range(num_var):
        f_basis = transform_params['f_basis']
        selected_features = transform_params['selected_features'][i]
        selected_functions = transform_params['selected_functions'][i]
        
        # Apply the selected functions to the selected features
        transformed_vars = np.hstack([f_basis[func_idx](X[:, feat]).reshape(-1,1) for feat, func_idx in zip(selected_features, selected_functions)])
        interaction_terms[:,i] = transformed_vars.prod(1)
            
    # Combine all transformed variables using the coefficients
    coefficients = transform_params['coefficients']
    noise = transform_params['noise'] if noise is None else noise
    transformed_X = (interaction_terms * coefficients).sum(1) + np.random.normal(0, noise, n_samples)
    return transformed_X

##############################################
## Linear Function Generation
##############################################
def linear_transform(X, lo=1, hi=None, noise=1, seed=1, f_basis=None):
    np.random.seed(seed)
    n_samples, n_features = X.shape
    
    if hi is None:
        hi = n_features
    else:
        hi = min(hi, n_features)
        
    num_var = np.random.randint(lo, hi + 1)
    selected_features = np.random.choice(n_features, size=num_var, replace=False)

    transform_params = {
        'nonlinear': False,
        'num_var': num_var,
        'noise': noise,
        'selected_features': selected_features.tolist(),
        'coefficients': []
    }
    
    coefficients = np.random.normal(0, 1, num_var)
    transform_params['coefficients'] = coefficients.tolist()
    
    transformed_X = (X[:, selected_features] * coefficients).sum(1) + np.random.normal(0, noise, n_samples)
    return transformed_X, transform_params

def apply_saved_linear_transform(X, transform_params, noise=None, seed=1):
    np.random.seed(seed)
    n_samples = X.shape[0]
    num_var = transform_params['num_var']
    noise = transform_params['noise'] if noise is None else noise
    selected_features = transform_params['selected_features']
    coefficients = transform_params['coefficients']
    transformed_X = (X[:, selected_features] * coefficients).sum(1) + np.random.normal(0, noise, n_samples)
    return transformed_X

##############################################
## Define multi-dimensional output
##############################################

def multidim_transform(X, output_dim, nonlinear, lo=1, hi=None, noise=1, seed=1, f_basis=NONLINEAR_FUNC_BASIS):
    np.random.seed(seed)
    seed_list = np.random.randint(0, output_dim*10, output_dim)
    param_list = []
    transformed_X = np.zeros((X.shape[0], output_dim))
    fun = nonlinear_transform if nonlinear else linear_transform
    for j in range(output_dim):
        transformed_X[:,j], param = fun(X, lo, hi, noise, seed=seed_list[j], f_basis=f_basis)
        param_list.append(param)
    return transformed_X, param_list

def apply_saved_multidim_transform(X, param_list, noise=None, seed=1):
    np.random.seed(seed)

    n_samples = X.shape[0]
    output_dim = len(param_list)
    transformed_X = np.zeros((n_samples, output_dim))
    
    seed_list = np.random.randint(0, output_dim*10, output_dim)

    fun = apply_saved_nonlinear_transform if param_list[0]['nonlinear'] else apply_saved_linear_transform
    for j in range(output_dim):
        transformed_X[:,j] = fun(X, param_list[j], noise, seed_list[j])
    return transformed_X


########################################################################################################
# Neural Network Training
########################################################################################################

class TLinear(nn.Module):
    def __init__(self, T, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(torch.rand(T, size_in, size_out)).requires_grad_()
        self.bias = nn.Parameter(torch.rand(T, size_out)).requires_grad_()
    
    def forward(self, x):
        return torch.matmul(x.unsqueeze(2), self.weights).squeeze(2) + self.bias
    
def train_nn(model, data, lr=1e-3, epochs=100, batch_size=128, T_max=500, verbose=True, device='cpu', loss_func=None, **kwargs):
    if loss_func is None:
        loss_func = model.loss
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    data_loader = DataLoader(MultiDataset(data), batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        training_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            loss = loss_func(*batch, device=device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_loss += loss.item()
        if verbose and epoch % max(verbose, 50) == 0:
            print(f"Epoch {epoch} (lr={scheduler.get_last_lr()[0]:.3f}) training loss: {training_loss/len(data_loader):.3f}")
    
    
class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = datasets[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)
    
def reinit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

########################################################################################################
# Aggregate Results by budget percentage usage
########################################################################################################
def aggregate_score_by_budget_density(cost_array_batch, score_array_batch, budget, lb=0.05, hb=0.95, num_bins=10):
    # Create percentage bins
    # percentages = np.hstack([np.linspace(0, 0.5, int(0.7 * num_bins)), np.linspace(0.5, 1, int(0.3 * num_bins))])
    percentages = np.linspace(0, 1, num_bins + 1)
    bin_edges = percentages * budget
    # Initialize arrays to store scores for each bin across all simulations
    
    # If the cost_array_batch is a single array, reshape it
    if len(cost_array_batch.shape) == 1:
        cost_array_batch = cost_array_batch.reshape(1,-1)
        score_array_batch = score_array_batch.reshape(1,-1)
        
    # Exclude extreme trajectories
    start_scores = score_array_batch[:,0]
    lo, hi = np.quantile(start_scores, [lb, hb])
    idxSel = (start_scores >= lo) & (start_scores <= hi)
    cost_array_batch = cost_array_batch[idxSel]
    score_array_batch = score_array_batch[idxSel]
    
    all_bin_scores = np.ones((cost_array_batch.shape[0], num_bins)) * np.nan

    # Process each simulation result
    for i in range(cost_array_batch.shape[0]):
        cost_array = cost_array_batch[i]
        score_array = score_array_batch[i] 
        
        # Calculate cumulative costs
        cum_costs = np.cumsum(cost_array)
        
        # Determine where to stop (when budget is exceeded)
        valid_indices = cum_costs <= budget
        cum_costs = cum_costs[valid_indices]
        scores = score_array[valid_indices]
        
        if len(cum_costs) == 0:
            continue
        
        # Assign scores to bins
        for j in range(len(bin_edges) - 1):
            start, end = bin_edges[j], bin_edges[j+1]
            mask = (cum_costs > start) & (cum_costs <= end)
            
            if np.any(mask):
                all_bin_scores[i,j] = scores[mask][-1]
        all_bin_scores[i] = pd.Series(all_bin_scores[i]).ffill().bfill().values
        
    
    # Calculate statistics for each bin
    bin_centers = []
    mean_scores = []
    std_scores = []
        
    for j in range(num_bins):
        bin_centers.append(percentages[j+1])
        scores = all_bin_scores[:,j]
        # lo, up = np.quantile(scores, [lb, hb])
        # filtered_scores = scores[(scores >= lo) & (scores <= up)]
        # mean_scores.append(np.mean(filtered_scores) if len(filtered_scores) > 0 else np.mean(scores))
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
        # std_scores.append(np.std(filtered_scores))
    
    return bin_centers, mean_scores, std_scores


# def aggregate_score_by_budget_density(cost_array_batch, score_array_batch, budget, num_bins=10):
#     # Create percentage bins
#     # percentages = np.hstack([np.linspace(0, 0.5, int(0.7 * num_bins)), np.linspace(0.5, 1, int(0.3 * num_bins))])
#     percentages = np.linspace(0, 1, num_bins + 1)
#     bin_edges = percentages * budget
#     # Initialize arrays to store scores for each bin across all simulations
#     all_bin_scores = [[] for _ in range(num_bins)]
    
#     # If the cost_array_batch is a single array, reshape it
#     if len(cost_array_batch.shape) == 1:
#         cost_array_batch = cost_array_batch.reshape(1,-1)
#         score_array_batch = score_array_batch.reshape(1,-1)
        
#     start_scores = score_array_batch[:,0]
#     lo, hi = np.percentile(start_scores, [5, 95])
#     idxSel = (start_scores >= lo) & (start_scores <= hi)
#     cost_array_batch = cost_array_batch[idxSel]
#     score_array_batch = score_array_batch[idxSel]
        
#     # Process each simulation result
#     for i in range(cost_array_batch.shape[0]):
#         cost_array = cost_array_batch[i]
#         score_array = score_array_batch[i] 
        
#         # Calculate cumulative costs
#         cum_costs = np.cumsum(cost_array)
        
#         # Determine where to stop (when budget is exceeded)
#         valid_indices = cum_costs <= budget
#         cum_costs = cum_costs[valid_indices]
#         scores = score_array[valid_indices]
        
#         if len(cum_costs) == 0:
#             continue
        
#         # Assign scores to bins
#         for j in range(len(bin_edges) - 1):
#             start, end = bin_edges[j], bin_edges[j+1]
#             mask = (cum_costs > start) & (cum_costs <= end)
            
#             if np.any(mask):
#                 all_bin_scores[j].extend(scores[mask].tolist())
    
#     # Calculate statistics for each bin
#     bin_centers = []
#     mean_scores = []
#     std_scores = []
        
#     for i in range(num_bins):
#         if all_bin_scores[i]:
#             bin_centers.append(percentages[i+1])
            
#             scores_array = np.array(all_bin_scores[i])
#             # lo, up = np.percentile(scores_array, [0,100])
#             filtered_scores = scores_array[(scores_array >= lo) & (scores_array <= up)]
#             mean_scores.append(np.mean(filtered_scores) if len(filtered_scores) > 0 else np.mean(scores_array))
            
#             std_scores.append(np.std(all_bin_scores[i]))
    
#     return bin_centers, mean_scores, std_scores


def gaussian_smooth(x, y, sigma=1.0):
    x = np.array(x)
    y = np.array(y)
    
    # If there are too few points, return the original data
    if len(x) <= 3:
        return y
    
    # Create the smoothed array
    y_smooth = np.zeros_like(y)
    
    # Apply Gaussian kernel to each point
    for i in range(len(x)):
        # Calculate Gaussian weights for all points
        weights = np.exp(-0.5 * ((x - x[i]) / sigma)**2)
        # Normalize weights
        weights = weights / np.sum(weights)
        # Apply weighted average
        y_smooth[i] = np.sum(weights * y)
    
    return y_smooth

########################################################################################################
# Plotting
########################################################################################################
import matplotlib.pyplot as plt

def plot_budget_performance_curve(bin_centers, mean_scores, std_scores=None, label=None, title=None, color=None, alpha=0.7, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # Convert inputs to numpy arrays if they aren't already
    bin_centers = np.array(bin_centers)
    mean_scores = np.array(mean_scores)
    
    # Plot the mean curve
    line = ax.plot(bin_centers, mean_scores, '-o', label=label, color=color, alpha=alpha, markersize=3)
    
    # If color wasn't specified, get the color that was just used
    if color is None:
        color = line[0].get_color()
    
    # Add confidence band if std_scores are provided
    if std_scores is not None:
        std_scores = np.array(std_scores)
        ax.fill_between(
            bin_centers, 
            mean_scores - std_scores, 
            mean_scores + std_scores, 
            color=color, 
            alpha=0.2
        )
    
    # Set labels and title
    # The code is setting the limits for the x-axis in a plot, but the specific values for the limits
    # are missing and need to be filled in.
    # ax.set_xlim(, )
    ax.set_xlabel('Budget Consumption in Percentages (%)', fontsize=12)
    # ax.set_ylabel('Performance Score')
    ax.set_title('Performance vs Budget Consumption')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # if label is not None:
    #     ax.legend()
    
    if title is not None:
        ax.set_title(title[0], fontsize=title[1])
    
    return ax