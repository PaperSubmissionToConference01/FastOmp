import sklearn.metrics as mt
import math
import numpy as np
import pandas as pd

from sklearn import dummy
from sklearn.linear_model import *



def oracle(features_np, target_np, S, model, algo) :

    '''
    Train the model and outputs metric for a set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    S -- index for the features used for model construction
    model -- choose if the regression is linear or logistic
    algo -- specify the output, based on the algorithm used for optimization
    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float log_like -- the normalized log-likelihood for the trained model
    float score -- the R^2 score for the trained linear model
    '''

    # preprocess current solution
    S = np.unique(S[S >= 0])
    
    # logistic model
    if model == 'logistic' :
    
        if algo == 'FAST_OMP' or algo == 'SDS_OMP' :
            grad, log_like = Logistic_Regression(features_np, target_np, S, gradient = True)
            return grad, log_like
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Random' :
            log_like = Logistic_Regression(features_np, target_np, S, gradient = False)
            return log_like
            
    # linear model
    if model == 'linear' :
    
        if algo == 'FAST_OMP' or algo == 'SDS_OMP' :
            grad, score = Linear_Regression(features_np, target_np, S, gradient = True)
            return grad, score
        
        if algo == 'SDS_MA' or algo == 'DASH' or algo == 'Random' :
            score = Linear_Regression(features_np, target_np, S, gradient = False)
            return score



# ------------------------------------------------------------------------------------------
#  logistic regression
# ------------------------------------------------------------------------------------------

def Logistic_Regression(features, target, dims, gradient = True):

    '''
    Logistic regression for a given set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    OMP -- if set to TRUE the function returns grad
    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float log_like -- the normalizde log-likelihood for the trained model
    '''

    # preprocess features
    features = pd.DataFrame(features)
    
    # predict dummy probabilities and dummy predictions
    dummy_model = dummy.DummyClassifier(strategy='uniform').fit(features, target)
    dummy_predict_prob = np.array(dummy_model.predict_proba(features))
    if gradient : dummy_predictions = dummy_model.predict(features)

    # predict probabilities and dummy predictions
    if not (features.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        
        # get model, predict probabilities, and predictions
        model = LogisticRegression(max_iter = 10000).fit(sparse_features , target)
        predict_prob = np.array(model.predict_proba(sparse_features))
        if gradient : predictions = model.predict(sparse_features)
        
    else :
    
        # get model, predict probabilities, and predictions
        model = dummy_model
        predict_prob = dummy_predict_prob
        if gradient : predictions = dummy_predictions

    # conpute gradient of log likelihood
    if gradient :
        log_like = (-mt.log_loss(target, predict_prob) + mt.log_loss(target, dummy_predict_prob)) * len(target)
        grad = np.dot(features.T, target - predictions)
        return grad, log_like
      
    # do not conpute gradient of log likelihood
    else :
        log_like = (-mt.log_loss(target, predict_prob) + mt.log_loss(target, dummy_predict_prob)) * len(target)
        return log_like



# ------------------------------------------------------------------------------------------
#  linear regression
# ------------------------------------------------------------------------------------------

def Linear_Regression(features, target, dims, gradient = True):

    '''
    Linear regression for a given set of features
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    OMP -- if set to TRUE the function returns grad
    OUTPUTS:
    float grad -- the garadient of the coefficient of determination
    float score -- the R^2 score for the trained model
    '''

    # preprocess features and target
    features = pd.DataFrame(features)
    target = np.array(target).reshape(target.shape[0], -1)
    
    if not (features.iloc[:,dims]).empty :
    
        # define sparse features
        sparse_features = np.array(features.iloc[:,dims])
        if sparse_features.ndim == 1 : sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)

        # get model, predict probabilities, and predictions
        model = LinearRegression().fit(sparse_features , target)
        score = model.score(sparse_features , target)
        if gradient : predict = model.predict(sparse_features)
        
    else :
    
        # predict probabilities, and predictions
        score = 0
        if gradient : predict = (np.ones((features.shape[0])) * 0.5).reshape(features.shape[0], -1)

    # compute gradient of log likelihood
    if gradient :
        grad = np.dot(features.T, target - predict)
        return grad, score
     
    # do not compute gradient of log likelihood
    else : return score



# function to evaluate additional constraints
def constraint(S) : return True
