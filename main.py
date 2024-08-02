import pandas as pd
from dataloader import load_data
from algorithm import LogisticRegression, GradientDescent

from preprocessing import Preprocessor


def main():
    # loading data
    data_obj = load_data('online_course_engagement_data.csv')
    data = data_obj.load()

    # Preprocessing the data
    preprocessor_obj = Preprocessor(data)
    X, y, features = preprocessor_obj.preprocess()

    inpt = int(input('1. Simple Logistic Regression \n2. Gradient Descent Logistic Regression. \nChoose \t : '))

    if inpt == 1:
        print('SIMPLE LOGISTIC REGRESSION')
        model = LogisticRegression()
    else:
        print('GRADIENT BOOSTING')
        model = GradientDescent(7000,0.0001)
    
    model.fit(X,y)             # This calculates the weight matrix using np.linalg.inv. This gets linear predictions -> y = X W
    # predictions = model.predict()
    # Evaluating model
    report = model.evaluate(y)
    print(report)


if __name__ == '__main__':
    main()