from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

'''
1.evaluation function
'''


def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


'''
2.cross validation to optimal params function
'''


def fit_model(X,y):
    cv = KFold(n_splits=8, shuffle=True) #cross validator
    reg= DecisionTreeRegressor()                #regressor
    params={'max_depth':np.arange(1,11)}        #parameters to opitaml
    scoring_fnc=make_scorer(performance_metric)  #scoring function
    grid=GridSearchCV(estimator=reg,param_grid=params,scoring=scoring_fnc,cv=cv,)#grid search
    grid.fit(X,y)                               #training to select best estimator

    return grid.best_estimator_


'''
3.main
'''
if __name__=='__main__':
    '''
    1.loading dataset
    '''

    data = pd.read_csv('bj_housing.csv')
    prices = data['Value']
    features = data.drop('Value', axis=1)
    print "Beijing housing dataset has {} data points with {} variables each.".format(*features.shape)

    '''
    2.split dataset
    '''
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

    '''
    3.select best estimator
    '''
    optimal_reg = fit_model(X_train, y_train)
    print "Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth'])

    '''
    4.scores for test data
    '''
    y_predic = optimal_reg.predict(X_test)

    r2 = performance_metric(y_test, y_predic)

    print r2