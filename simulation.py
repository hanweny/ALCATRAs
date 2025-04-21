import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import utils as ut
import model_policy as mp
from scipy.stats import kendalltau
import itertools

N, P, Q = None, None, None

def define_global_variable(n_param, p_param, q_param):
    global N, P, Q, W_TRANSFORMATION_PARAMS, OPT_POLICY, Y_PARAMS, feature_importance, COST_PARAMS, W_bias, TASK_COSTS
    N, P, Q = n_param, p_param, q_param
    W_TRANSFORMATION_PARAMS, OPT_POLICY = [], []
    Y_PARAMS, feature_importance = None, None
    COST_PARAMS = None
    W_bias = None
    TASK_COSTS = None
    return N, P, Q


def generate_data(N, P, Q, nonlinear={"W":True, "Policy":True, "y":True, "cost": False}, noise={"W": None, "Policy": None, "y": None, "cost": 0}, 
                  classification=False, seed=1):
    define_global_variable(N, P, Q)
    global W_bias
    np.random.seed(seed)
    X = np.random.rand(N, P)
    
    # define task variablesx
    if noise['W'] is None:
        # W_bias = np.random.uniform(1, 7, (1, Q))
        # W_bias = np.random.normal(-5, 5, (1, Q))
        noise['W'] = np.random.uniform(5, 10, Q)    
        
    W = define_task(X, nonlinear=nonlinear["W"], lo=3, noise=noise['W'], seed=seed)
    # W += np.random.normal(W_bias, 1, size=(N, Q))  
    
    # define optimal policy
    if noise['Policy'] is None:
        noise['Policy'] = 1        
    opt_policy_list = define_optimal_policy(X, W, nonlinear=nonlinear["Policy"], lo=5, hi=None, noise=noise['Policy'], seed=seed)
    optimal_policy = mp.OptimalPolicy(P, Q, opt_policy_list)
    opt_task_index, opt_masks = optimal_policy.select_task(X, W)

    # define regressor y
    if noise['y'] is None:
        noise['y'] = 1
    y = define_regressor_y(X, W, opt_masks, nonlinear=nonlinear["y"], lo=P, noise=noise['y'], classification=classification, seed=seed)
    
    
    # Define cost model
    define_base_cost_model(X, nonlinear=nonlinear["cost"], lo=3, noise=noise['cost'], seed=seed)
    
    return X, W, y, optimal_policy, opt_task_index, opt_masks


def split_data(X, W, y, opt_masks, opt_task_index, nL, nU, nte, seed):
    Xtr, Xte, Wtr, Wte, ytr, yte, Mtr, Mte, titr, tiTe = train_test_split(X, W, y, opt_masks, opt_task_index, test_size=nte, random_state=seed)
    XL, XU, WL, WU, yL, yU, ML, MU, tiL, tiU = train_test_split(Xtr, Wtr, ytr, Mtr, titr, train_size=nL, random_state=seed)
    XU, WU, yU, MU, tiU = XU[:nU], WU[:nU], yU[:nU], MU[:nU], tiU[:nU]
    # XU, Xte, WU, Wte, yU, yte, MU, Mte, tiU, tiTe = train_test_split(XU, WU, yU, MU, tiU, train_size=nU, random_state=seed)
    Labeled_data = (XL, WL, yL, tiL, ML)
    Unlabeled_data = (XU, WU, yU, tiU, MU)
    Test_data = (Xte, Wte, yte, tiTe, Mte)
    return Labeled_data, Unlabeled_data, Test_data


################################################################
# Define Tasks
###############################################################
W_TRANSFORMATION_PARAMS = []
W_bias = None
def define_task(X, nonlinear=False, lo=1, hi=None, noise=None, seed=1):
    global W_TRANSFORMATION_PARAMS
    np.random.seed(seed)
    
    assert len(noise) == Q
    
    # Initialize transformation parameters
    W = np.zeros((N, Q))
    
    for q in range(Q):
        W[:, q], w_params = ut.function_transform(X, lo, hi, noise[q], seed=q, nonlinear=nonlinear)
        W_TRANSFORMATION_PARAMS.append(w_params)
    
    return W

def generate_task(X, noise=None, seed=1):      
    # by default it applies the function_define noises.  
    W = np.zeros((X.shape[0], Q))
    for q in range(Q):
        W[:, q] = ut.apply_saved_function_transform(X, W_TRANSFORMATION_PARAMS[q], noise, seed)
    
    return W


################################################################
# Define Optimal Policy
################################################################
POLICY_TRANSFORMATION_PARAMS = []
OPT_POLICY = []

def define_optimal_policy(X, W, nonlinear=False, lo=1, hi=None, noise=None, seed=1):
    global POLICY_TRANSFORMATION_PARAMS, OPT_POLICY, Q
    
    np.random.seed(seed)
        
    seed_list = np.random.randint(0, 10000, Q)
    
    for q in range(Q):  # at each step
        _, step_params = ut.function_transform(
            np.hstack([X, W]), lo, hi, noise, seed=seed_list[q], output_dim=Q+1, nonlinear=nonlinear
        )
        POLICY_TRANSFORMATION_PARAMS.append(step_params)
        OPT_POLICY.append(
            lambda X, W, prev_masks, **kwargs: ut.normalize_squared(
                ut.apply_saved_function_transform(
                    np.hstack([X, prev_masks[:,-1,:] * W]), step_params, noise=None, seed=seed_list[q]
                )
            )
        )
    
    return OPT_POLICY

################################
# Define cost model 
###############################
COST_PARAMS = None
def define_base_cost_model(X, nonlinear=False, lo=3, hi=None, noise=0, seed=1):
    global COST_PARAMS
    np.random.seed(seed)
    cost, COST_PARAMS = ut.function_transform(X, lo, hi, noise, seed=seed, nonlinear=nonlinear)
    return COST_PARAMS

def calculate_efficiency_cost(ti_pred, ti_true):
    n_a, n_b = (~np.isnan(ti_pred)).sum(), (~np.isnan(ti_true)).sum()

    B = np.zeros(len(ti_true)+1)
    B[np.array(ti_true[:n_b]).astype(int)] = n_b - np.arange(n_b)

    # The underlying optimal scores for the predicted task rank
    A = B[np.array(ti_pred[:n_a]).astype(int)]
    if len(A) == 1:
        return ti_pred[0] != ti_true[0]
    missing_idx = list(set(ti_true[:n_b]) - set(ti_pred[:n_a]))
    # A = np.concatenate([A, sorted(B[np.array(missing_idx).astype(int)])])
    pairs = itertools.combinations(range(0, len(A)), 2)
    distance = 0
    for x, y in pairs:
        if A[x] - A[y] <= 0:
            distance += 1
    return distance / (len(A) * (len(A) - 1) / 2)


TASK_COSTS = None
def costModel(X, W, M, optM, task_costs = None):
    global COST_PARAMS, TASK_COSTS
    n_samples, n_tasks = M.shape[0], M.shape[-1]

    # Base cost from covariates
    # base_cost = 0.5 + 1 * ut.softplus(ut.apply_saved_function_transform(X, COST_PARAMS, noise=0))
    base_cost = ut.softmax_range(ut.apply_saved_function_transform(X, COST_PARAMS, noise=0), 0, 1)
    
    # Cost from completing the tasks        
    if task_costs is not None:
        TASK_COSTS = task_costs
        
    if TASK_COSTS is None:
        TASK_COSTS = np.ones(n_tasks) / n_tasks
    
    task_cost = (M[:,-1,:] * TASK_COSTS).sum(1)
    
    # Cost from not completing the tasks in order
    # Efficiency costs = out_of_order + mismatch, ranging from 0 to 1
    n_streak, n_rest, n_length = ut.compare_task_masks(M, optM)
                
    out_of_order_penalty = np.where(n_length==0, 0, 1 - n_streak / n_length.clip(1, np.inf))
    mismatch_penalty = np.where(n_length == n_streak, 0, 1 - n_rest / (n_length - n_streak).clip(1, np.inf))
    # efficiency_penalty = (0.9 * out_of_order_penalty + 0.1 * mismatch_penalty)

    ti_pred, ti_true = ut.convert_mask_to_task_index(M), ut.convert_mask_to_task_index(optM)
    ranking_penalty = np.array([calculate_efficiency_cost(a, b) for a, b in zip(ti_pred, ti_true)])
    
    efficiency_penalty = 0.5 * ranking_penalty + 0.3 * out_of_order_penalty + 0.2 * mismatch_penalty
    
    # rank_dist = np.array([kendalltau(a, b, nan_policy='omit').statistic for a, b in zip(ti_pred, ti_true)])
    # efficiency_penalty = 1 - (np.where(np.isnan(rank_dist), -1, rank_dist) + 1) / 2
    
    # Scale the task costs to 1
    
    task_total_costs = (task_cost * 0.2 + efficiency_penalty * 0.8) 
    
    detailed_costs = {
        'base_cost': base_cost,
        'task_cost': task_cost,
        'efficiency_penalty': efficiency_penalty,
        'ranking_penalty': ranking_penalty,
        'out_of_order_penalty': out_of_order_penalty,
        'mismatch_penalty': mismatch_penalty,
        'task_total_costs': task_total_costs,
        'n_streak': n_streak,
        'n_rest': n_rest,
        'n_length': n_length,
    }
    # rank_dist = (np.where(np.isnan(rank_dist), -1, rank_dist) + 1) / 2
    
    return 0.1 * base_cost + 0.9 * task_total_costs, detailed_costs
    

################################
# Define regressor y
###############################
Y_PARAMS, feature_importance = None, None
def define_regressor_y(X, W, opt_masks, nonlinear=False, lo=3, hi=None, noise=None, classification=False, seed=1):
    global Y_PARAMS, feature_importance

    np.random.seed(seed)
        
    Y_base, Y_PARAMS = ut.function_transform(X, lo, hi, noise, seed=seed, nonlinear=nonlinear)
    
    if feature_importance is None:
        feature_importance = [x/Q for x in np.arange(1, Q+1)][::-1]
    
    #feature_importance[-1]
    task_effects = np.zeros(N)
    mask_step = ut.convert_task_mask_to_onehot(opt_masks)
    for q in range(Q):
        task_effects += feature_importance[q] * (W * mask_step[:,q,:]).sum(1)
    
    y = Y_base * 0.1 + task_effects
    
    # Rescale the y
    y = 0.3 * y
    
    if classification:
        y = y - y.mean()
        y = np.where(y > 0, 1, 0)
    
    return y