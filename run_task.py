
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
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

import pickle as pkl

result = dict()

def run_simulation(nL, Q, seed):
    
    # Aim to run once
    P = 10
    nU, nte, Budget, batch_size = 1000, 5000, 16, 16
    
    np.random.seed(seed)
    task_costs = np.random.rand(Q)
    task_costs = task_costs / task_costs.sum()
    custom_costModel = lambda X, W, M, optM: sim.costModel(X, W, M, optM, task_costs=task_costs)
    

    sim.define_global_variable(nL+nU+nte, P, Q)
    X, W, y, optimal_policy, opt_task_index, opt_masks = sim.generate_data(
        50000, P, Q, nonlinear={"W":True, "Policy":False, "y":True, "cost":False}, noise={"W":None, "Policy":0, "y": None, "cost": 0}, seed=seed
    )
    
    random_policy = model_policy.RandomPolicy(P, Q)
    ti, M = random_policy.select_task(X, W, seed=1)
    
    idx = ~((opt_masks[:,-1,:].sum(1) == 0) | (M[:,-1,:].sum(1) == 0))
    Labeled_data, Unlabeled_data, Test_data = sim.split_data(X[idx], W[idx], y[idx], M[idx], ti[idx], nL, nU, nte, 42)
    
    XL, WL, yL, tiL, ML = ut.deep_copy(Labeled_data)
    Xte, Wte, yte, tiTe, Mte = ut.deep_copy(Test_data)
    CL = sim.costModel(XL, WL, ML, optimal_policy.select_task(XL, WL)[1], task_costs=task_costs)[0]
    
    f_name = f"results/tasks2/L{nL}_U{nU}_test{nte}_P{P}_Q{Q}_B{Budget}_seed{seed}.pkl"

    # result = pkl.load(open(f_name, "rb"))
    
    # # Feature Acquisition Policy    
    fa_policy = model_policy.FeatureAcquisitionPOMPDP(P, Q, 4, seed=1)
    result['Feature Acquisition: Q-Learning'] = POMPDP.POMPDP_learning(
        fa_policy, optimal_policy, custom_costModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, 
        f_params={'epochs': 50, 'lr': 5e-3, 'verbose': False},
        policy_params={'epochs': 500, 'lr': 5e-3}
    )['cost_test'][-1]
    print("FA: ", Counter(ut.visualize_task_index(fa_policy.select_task(Xte, Wte, seed=1)[0])).most_common(5))
    print("FA avg length: ", fa_policy.select_task(Xte, Wte, seed=1)[1][:,-1,:].sum(1).mean())

    #### DTR Policy
        
    num_epochs = 500
    lr = 5e-3
    
    # streak-learning
    streak_policy = model_policy.DTRPolicy(model_policy.StreakLearning, P, Q, [4], image=False, seed=1)
    streak_policy.train_model(XL, WL, ML[:,:-1,:], ut.convert_task_index_to_action(tiL), -CL + CL.max(), yL, epochs=num_epochs, lr=lr)
    result['Streak-Learning'] = sim.costModel(Xte, Wte, streak_policy.select_task(Xte, Wte)[1], optimal_policy.select_task(Xte, Wte)[1], task_costs=task_costs)[1]
    print(result['Streak-Learning'].keys())
    print("Streak-learning: ", Counter(ut.visualize_task_index(streak_policy.select_task(Xte, Wte, seed=1)[0])).most_common(5))
    print("Streak avg length: ", streak_policy.select_task(Xte, Wte, seed=1)[1][:,-1,:].sum(1).mean())
    
    # ALCATRAz: OWL
    OWL_Policy = model_policy.DTRPolicy(model_policy.OWLearning, P, Q, [4], seed=1)
    OWL_Policy.train_model(XL, WL, ML[:,:-1,:], ut.convert_task_index_to_action(tiL), -CL + CL.max(), yL, epochs=num_epochs, lr=lr)
    result['Direct-Search'] = sim.costModel(Xte, Wte, OWL_Policy.select_task(Xte, Wte)[1], optimal_policy.select_task(Xte, Wte)[1], task_costs=task_costs)[1]
    print("OWL-learning: ", Counter(ut.visualize_task_index(OWL_Policy.select_task(Xte, Wte, seed=1)[0])).most_common(5))
    print("OWL avg length: ", OWL_Policy.select_task(Xte, Wte, seed=1)[1][:,-1,:].sum(1).mean())
    
    
    # ALCATRAz: Random Policy
    result['Random Policy'] = sim.costModel(Xte, Wte, Mte, optimal_policy.select_task(Xte, Wte)[1], task_costs=task_costs)[1]
    print("Random-learning: ", Counter(ut.visualize_task_index(random_policy.select_task(Xte, Wte, seed=1)[0])).most_common(5))
    print("Random avg length: ", random_policy.select_task(Xte, Wte, seed=1)[1][:,-1,:].sum(1).mean())

    pkl.dump(result, open(f_name, 'wb'))
    
    return result

def main():
    nL = int(sys.argv[1])
    Q = int(sys.argv[2])
    seed = int(sys.argv[3])
    run_simulation(nL, Q, seed)
    
if __name__ == "__main__":
    main()