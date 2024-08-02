import numpy as np
from sklearn.metrics import classification_report

class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def compute_weights(self,X,y):
        # Getting the transpose of X
        X_T = X.T
        weights = np.linalg.inv(X_T @ X) @ X_T @ y
        return weights


    def fit(self,X,y):
        self.weights = self.compute_weights(X,y)
        self.z = X @ self.weights           # z = wx . Here x includes the bias in the matrix itself

    def predict(self):
        y_probability = self.sigmoid(self.z)
        return [1 if y >=0.5 else 0 for y in y_probability]
    
    def evaluate(self, y):
        ypred = self.predict()
        return classification_report(y , ypred)
    

class GradientDescent(LogisticRegression):
    def __init__(self,iterations,lr):
        self.iterations = iterations
        self.alpha = lr

    def gradients(self, X, y, w, b):
        self.z = np.dot(X[:,:-1],w)  +  b          # z = wx + b
        y_pred = self.sigmoid(self.z)
        djdw = (1/X.shape[0]) * np.dot(X[:,:-1].T,(y_pred - y))                 # (1/m) * (f_wb - y)x
        djdb = (1/X.shape[0]) * np.sum((y_pred - y))
        return djdw, djdb

    def fit(self,X,y):
        self.weights = np.zeros(X.shape[1] - 1)      # matrix of 0s with shape n-1 to remove the last columns for the intercept
        self.bias = 0

        for _ in range(self.iterations):
            djdw , djdb = self.gradients(X, y, self.weights, self.bias)
            self.weights = self.weights - self.alpha * djdw
            self.bias = self.bias - self.alpha * djdb
        
        # Fitting the model with final weights, final bias
        self.z = np.dot(X[:,:-1],self.weights)  +  self.bias          # z = wx + b

    def predict(self):
        y_probability = self.sigmoid(self.z)
        return [1 if y >=0.5 else 0 for y in y_probability]
