import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self,data):
        self.data = data
        self.data_enc = None


    def preprocess(self):
        self.data.dropna(inplace = True)
        self.data.drop_duplicates(inplace = True)
        self.data.reset_index(inplace = True, drop=True)
        self.data.drop(columns='UserID',inplace = True)
        
        # Encoding Categorical Features
        self.data_enc = pd.concat([self.data,pd.get_dummies(self.data['CourseCategory'],dtype='int',prefix='CourseCategory_')],axis=1)
        self.data_enc.drop(columns = 'CourseCategory',inplace = True)

        # Setting Target and Independent Features
        Y = self.data_enc['CourseCompletion']
        X = self.data_enc.drop(columns='CourseCompletion')
        Features = X.columns.tolist()

        X_ones = np.ones((X.shape[0],1))            # creating a columns of Bias having value of 1
        X = np.array(X)
        X = np.concatenate((X,X_ones),axis=1)           # Adding that to the end of the feature matrix
        
        return np.array(X) , np.array(Y), Features
