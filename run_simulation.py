
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

result = dict()

def run_simulation(nL, Q, seed):
        
    P = 10
    # P, Q = 10, 15
    nU, nte, Budget, batch_size = 1000, 5000, 640, 16
    classification = False
    
    # nU = nL + Budget
    
    sim.define_global_variable(nL+nU+nte, P, Q)
    X, W, y, optimal_policy, opt_task_index, opt_masks = sim.generate_data(
        50000, P, Q, nonlinear={"W":True, "Policy":False, "y":True, "cost":False}, noise={"W":None, "Policy":1, "y": 5, "cost": 0}, seed=seed
    )
    
    
    random_policy = model_policy.RandomPolicy(P, Q)
    ti, M = random_policy.select_task(X, W, seed=1)
    Labeled_data, Unlabeled_data, Test_data = sim.split_data(X, W, y, M, ti, nL, nU, nte, 42)
    # Labeled_data, Unlabeled_data, Test_data = sim.split_data(X, W, y, opt_masks, opt_task_index, nL, nU, nte, 42)
    
    f_name = f"results/framework/L{nL}_U{nU}_test{nte}_P{P}_Q{Q}_B{Budget}_seed{seed}.pkl"
    result = pkl.load(open(f_name, 'rb'))
    
    
    # f = model.RandomForestModel(classification=classification, n_estimators=30, max_depth=5, random_state=42)
    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    f_params = {'epochs': 50, 'lr': 5e-3, 'verbose': 30}
    

    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    # f.train_model(
    #     np.vstack([Labeled_data[0], Unlabeled_data[0]]), np.hstack([Labeled_data[2], Unlabeled_data[2]]), 
    #     f_params={'epochs': 150, 'lr': 5e-3, 'verbose': 30}
    # )
    # result['Optimal'] = f.eval_model(Test_data[0], Test_data[2])
    # print("Optimal: ", f.eval_model(Test_data[0], Test_data[2]))
    
    
    # # # Classic AL
    # # f_rf = model.RandomForestRegressorModel(n_estimators=30, max_depth=5, random_state=42)
    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    # result['AL: classic'] = AL.AL_learning(
    #     f, None, optimal_policy, sim.costModel,Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty,
    #     f_params=f_params
    # )
    # print("Classic AL done", result['AL: classic']['f'])
    
    
    # # Cost-Sensitive AL
    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    # result['AL: cost-sensitive'] = AL.AL_learning(
    #     f, None, optimal_policy, sim.costModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.CostSensitiveUncertainty,
    #     f_params=f_params
    # )
    # print("Cost-Sensitive AL done", result['AL: cost-sensitive']['f'])
    
    
    # # # Surrogate learning to impute the surrogate y
    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    # g = model_surr.XSurrogateRandomForestModel(None, n_estimators=30, max_depth=5, random_state=42)
    # result['Surrogate-Learning'] = AL.AL_learning(
    #     f, None, optimal_policy, sim.costModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty, g=g,
    #     f_params={'epochs': 50, 'lr': 5e-3},
    #     g_params={'epochs': 30, 'lr': 5e-3, 'threshold': 0.2}
    # )
    # print("Surrogate Learning done", result['Surrogate-Learning']['f'])
    
    
    # # Feature Acquisition Policy    
    fa_policy = model_policy.FeatureAcquisitionPOMPDP(P, Q, 4, classification=False, image=False, seed=1)
    result['Feature Acquisition Framework'] = POMPDP.POMPDP_learning(
        fa_policy, optimal_policy, sim.costModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, 
        f_params={'epochs': 50, 'lr': 5e-3, 'verbose': False},
        policy_params={'epochs': 30, 'lr': 5e-3}
    )
    print("Feature Acquisition Framework done", result['Feature Acquisition Framework']['f'])
    
    
    # # ALCATRAz 
    # f = model.NeuralNetworkModel(input_dim=P, hidden_dim_list=[4], output_dim=1, seed=1, device='cpu')
    # # h = model_impute.GroundTrueImpute()
    # # h = model_impute.MeanImpute()
    # # h = model_impute.AutoEncoderImpute(input_dim=P, encoder_dim_list=[8, 4], latent_dim=Q, decoder_dim_list=[], output_dim=Q, seed=1, device='cpu')
    # # h = model_impute.VariationalImpute(input_dim=P, encoder_dim_list=[4], latent_dim=Q, decoder_dim_list=[], output_dim=Q, seed=1, device='cpu')
    # h = model_impute.OnlineVariationalImpute(input_dim=P, encoder_dim_list=[8, 4], latent_dim=Q, decoder_dim_list=[], output_dim=Q, seed=1, device='cpu')
    # g_h = model_surr.XWSurrogateRandomForestModel(h, n_estimators=30, max_depth=5, random_state=42)
    # streak_policy = model_policy.DTRPolicy(model_policy.StreakLearning, P, Q, [4], image=False, seed=1)
    # result['Proposed Framework: ALCATRAs Online'] = AL.AL_learning(
    #     f, streak_policy, optimal_policy, sim.costModel, Labeled_data, Unlabeled_data, Test_data, Budget, batch_size, AL.Uncertainty, g=g_h,
    #     f_params={'epochs': 50, 'lr': 5e-3},
    #     g_params={'epochs': 30, 'lr': 5e-3, 'threshold': 0.5}, 
    #     h_params={'epochs': 250, 'lr': 5e-3}, 
    #     policy_params={'epochs': 50, 'lr': 5e-3}
    # )
    # print("ALCATRAs-online done", result['Proposed Framework: ALCATRAs Online']['f'])
    
    
    
    pkl.dump(result, open(f_name, 'wb'))
    
    return result

def main():
    nL = int(sys.argv[1])
    Q = int(sys.argv[2])
    seed = int(sys.argv[3])
    run_simulation(nL, Q, seed)
    
if __name__ == "__main__":
    main()