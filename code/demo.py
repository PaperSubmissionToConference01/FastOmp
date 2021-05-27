import models
import algos as alg
import math as mt
import numpy as np
import pandas as pd
import time as tm
import sys
    

def run_experiment(features, target, model, eps, tau, N_samples, sparsityRange, numberCPUs, SDS_OMP = True, FAST_OMP = True, SDS_MA = True, Random = True) :

    '''
    Run a set of experiments for selected algorithms. All results are saved to text files.
    
    INPUTS:
    features -- the feature matrix for the regression
    target -- the observations for the regression
    model -- choose if 'logistic' or 'linear' regression
    eps -- parameter epsilon for FAST_OMP
    tau -- parameter m/M for the FAST_OMP
    N_samples -- number of runs for the randomized algorithms
    sparsityRange -- range for the parameter k for a set of experiments
    numberCPUs -- number of CPUs used to run the algorithms
    SDS_OMP -- if True, test this algorithm
    FAST_OMP -- if True, test this algorithm
    SDS_MA -- if True, test this algorithm
    Random -- if True, test this algorithm
    '''
    
    length_data =  len(sparsityRange) * len(numberCPUs)
 
 
    # ----- run FAST_OMP
    
    if SDS_OMP :
        print('----- testing SDS_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time_mn': np.zeros(length_data), 'rounds_mn': np.zeros(length_data),'metric_mn': np.zeros(length_data), 'time_sd': np.zeros(length_data), 'rounds_sd': np.zeros(length_data),'metric_sd': np.zeros(length_data)})

        h = 0
        for j in range(len(sparsityRange)) :

            for l in range(len(numberCPUs)) :
        
                # perform experiments
                out = [alg.SDS_OMP(features, target, model, sparsityRange[j], numberCPUs[l]) for i in range(N_samples)]
                out = np.array(out)
            
                # save data to file
                results.loc[j + h + l,'k']         = sparsityRange[j]
                results.loc[j + h + l,'n_cpus']    = numberCPUs[l]
                results.loc[j + h + l,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
                results.loc[j + h + l,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
                results.loc[j + h + l,'metric_mn'] = np.mean([out[i,2] for i in range(N_samples)])
                results.loc[j + h + l,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
                results.loc[j + h + l,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
                results.loc[j + h + l,'metric_sd'] = np.std([out[i,2]  for i in range(N_samples)])
                results.to_csv('SDS_OMP.csv', index = False)
        
            h += l

        
    # ----- run FAST_OMP
    
    if FAST_OMP :
        print('----- testing FAST_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time_mn': np.zeros(length_data), 'rounds_mn': np.zeros(length_data),'metric_mn': np.zeros(length_data), 'time_sd': np.zeros(length_data), 'rounds_sd': np.zeros(length_data),'metric_sd': np.zeros(length_data)})
        
        h = 0
        for j in range(len(sparsityRange)) :
        
            for l in range(len(numberCPUs)) :
        
                # perform experiments
                out = [alg.FAST_OMP(features, target, model, sparsityRange[j], eps, tau, numberCPUs[l]) for i in range(N_samples)]
                out = np.array(out)
            
                # save data to file
                results.loc[j + h+l,'k']         = sparsityRange[j]
                results.loc[j + h+l,'n_cpus']     = numberCPUs[l]
                results.loc[j + h+l,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
                results.loc[j + h+l,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
                results.to_csv('FAST_OMP.csv', index = False)
                
            h += l

            
    # ----- run Random
    
    if Random :
        print('----- testing Random')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time_mn': np.zeros(length_data), 'rounds_mn': np.zeros(length_data),'metric_mn': np.zeros(length_data), 'time_sd': np.zeros(length_data), 'rounds_sd': np.zeros(length_data),'metric_sd': np.zeros(length_data)})
        
        h = 0
        for j in range(len(sparsityRange)) :
        
            for l in range(len(numberCPUs)) :
        
                # perform experiments
                out = [alg.Random(features, target, sparsityRange[j], model, numberCPUs[l]) for i in range(N_samples)]
                out = np.array(out)
            
                # save data to file
                results.loc[j + h+l,'k']         = sparsityRange[j]
                results.loc[j + h+l,'n_cpus']     = numberCPUs[l]
                results.loc[j + h+l,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
                results.loc[j + h+l,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
                results.loc[j + h+l,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
                results.loc[j + h+l,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
                results.to_csv('Random.csv', index = False)
                
            h += l
            
            
    # ----- run SDS_MA
    
    if SDS_MA :
    
        print('----- testing SDS_MA')
        results = pd.DataFrame(data = {'k': np.zeros(length_data).astype('int'), 'n_cpus': np.zeros(length_data).astype('int'), 'time': np.zeros(length_data), 'rounds': np.zeros(length_data),'rounds_ind': np.zeros(length_data),'metric': np.zeros(length_data)})
                
        h = 0
        for j in range(len(sparsityRange)) :

            for l in range(len(numberCPUs)) :
        
                # perform experiments
                out = alg.SDS_MA(features, target, sparsityRange[j], model, numberCPUs[l])
                out = np.array(out)
            
                # save data to file
                results.loc[j + h + l,'k'] = sparsityRange[j]
                results.loc[j + h + l,'n_cpus'] = numberCPUs[l]
                results.loc[j + h + l,'time']   = out[0]
                results.loc[j + h + l,'rounds'] = out[1]
                results.loc[j + h + l,'rounds_ind'] = out[2]
                results.loc[j + h + l,'metric'] = out[3]
                results.to_csv('SDS_MA.csv', index = False)

            h +=l
            
    return
    
    

def demo():

    '''
    Test algorithms with the run_experiment function.
    features -- the feature matrix for the regression
    target -- the observations for the regression
    model -- choose if 'logistic' or 'linear' regression
    eps -- parameter epsilon for FAST_OMP
    tau -- parameter m/M for the FAST_OMP
    N_samples -- number of runs for the randomized algorithms
    sparsityRange -- range for the parameter k for a set of experiments
    numberCPUs -- number of CPUs used to run the algorithms
    SDS_OMP -- if True, test this algorithm
    FAST_OMP -- if True, test this algorithm
    SDS_MA -- if True, test this algorithm
    Random -- if True, test this algorithm
    '''

    # define features and target for the experiments
    print('----- importing dataset')
    df = pd.read_csv('../data/healthstudy.csv', index_col=0, parse_dates=False)
    df = pd.DataFrame(df)
    target = df.iloc[:,0]
    features = df.iloc[:, range(1, df.shape[1])]

    # choose if logistic or linear regression
    model = 'logistic'

    # set range for the experiments
    sparsityRange = np.array([1,3,5,6])
    numberCPUs    = np.array([1,2])

    # choose algorithms to be tested
    SDS_MA   = True
    SDS_OMP  = True
    FAST_OMP = True
    Random   = True

    # define parameters for FAST_OMP
    eps = 0.9
    tau = 0.01

    # number of samples per evaluation
    N_samples = 3

    # run experiment
    run_experiment(features, target, model, eps, tau, N_samples, sparsityRange, numberCPUs, SDS_OMP, FAST_OMP, SDS_MA, Random)


def main():
    demo()
         
    
if __name__=="__main__":
    main()
