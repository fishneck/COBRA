import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from scipy.special import softmax

class TemperatureScaling():
#adopted from https://github.com/jackzhu727/deep-probability-estimation/blob/main/calibration_tools/postprocessing.py
    def __init__(self, temp=1, maxiter=50, solver="BFGS"):
        """
        Initialize class
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict_proba(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    # Find the temperature
    def fit(self, logtis, true):
        """
        Trains the model and finds optimal temperature
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
        Returns:
            the results of optimizer after minimizing is finished.
        """
        k = np.unique(true).shape[0]
        # true = true.flatten() # Flatten y_val
        true = np.eye(k)[true.astype(int)]
        opt = minimize(self._loss_fun, x0=1, args=(logtis, true), options={'maxiter': self.maxiter}, method=self.solver)
        self.temp = opt.x[0]

        return opt

    def predict_proba(self, logits, temp=None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return softmax(logits / self.temp)
        else:
            return softmax(logits / temp)

        
        
        
class MatrixScaling():
    def __init__(self, reg_C_list=[0.1, 1,10,100],logit_input=True, **kwargs):
        self.reg_C_list = reg_C_list
        parameters = { 'C':reg_C_list}
        self.clf = GridSearchCV(LogisticRegression(random_state=0,n_jobs=-1,max_iter=500), parameters)
        self.logit_input=logit_input #input is logit, not probabilities
        
    
    def fit(self, X, y,  *args, **kwargs):
        _X = X.copy()
        if self.logit_input:
            _X = softmax(X, axis=0)
        print(f'fitting calibrator with {_X.shape[0]} samples')
        self.clf.fit(_X,y)
        
    @property
    def coef_(self):
        return self.clf.best_estimator_.coef_

    @property
    def intercept_(self):
        return self.clf.best_estimator_.intercept_

    def predict_proba(self, S):
        if self.logit_input:
            S = softmax(S, axis=0)
        pred = self.clf.best_estimator_.predict_proba(S)
        
        return pred

    def predict(self, S):
        if self.logit_input:
            S = softmax(S, axis=0)
        pred = self.clf.best_estimator_.predict(S)
        
        return pred

    
class DirichletCalibrator():
    def __init__(self, reg_C_list=[0.1, 1,10,100],logit_input=True, eps=1e-22, **kwargs):
        self.reg_C_list = reg_C_list
        parameters = { 'C':reg_C_list}
        self.clf = GridSearchCV(LogisticRegression(random_state=0,n_jobs=-1,max_iter=500), parameters)
        self.logit_input=logit_input #input is logit, not probabilities
        self.eps = eps
    
    def fit(self, X, y, *args, **kwargs):
        _X = X.copy()
        if self.logit_input:
            _X = softmax(X, axis=0)
        _X = np.log(_X+self.eps)
        print(f'fitting calibrator with {_X.shape[0]} samples')
        self.clf.fit(_X,y)
        
    @property
    def coef_(self):
        return self.clf.best_estimator_.coef_

    @property
    def intercept_(self):
        return self.clf.best_estimator_.intercept_

    def predict_proba(self, S):
        if self.logit_input:
            S = softmax(S, axis=0)
        S = np.log(S+self.eps)
        pred = self.clf.best_estimator_.predict_proba(S)
        
        return pred

    def predict(self, S):
        if self.logit_input:
            S = softmax(S, axis=0)
        pred = self.clf.best_estimator_.predict(S)
        
        return pred

    