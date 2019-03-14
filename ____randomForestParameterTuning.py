'''
Created on 26 Feb 2019

@author: gche0022
'''

import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, cohen_kappa_score


def parameter_tuning_random_forest(path):
    # Read data
    dtrain = pandas.read_csv(path + "train_1.0")
    ddev = pandas.read_csv(path + "dev_1.0")
    dtest = pandas.read_csv(path + "test_1.0")
    
    dtrain_predictors = [x for x in dtrain.columns if x not in ['label']]
    ddev_predictors = [x for x in ddev.columns if x not in ['label']]
    dtest_predictors = [x for x in dtest.columns if x not in ['label']]
    
    scoring = {'AUC': make_scorer(roc_auc_score),
               'Cohen\'s Kappa': make_scorer(cohen_kappa_score),
               'F1':make_scorer(f1_score)}
    
    clf=RandomForestClassifier(random_state=27, class_weight='balanced')
    param_grid = {'n_estimators': range(25,201,25),
                  'max_features': ['auto', 'sqrt', 'log2', None],
                  'max_depth': range(3,10,1)
                  }
    
    gsearch = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,
                           scoring=scoring,
                           refit='Cohen\'s Kappa', return_train_score=True, verbose=10)
    gsearch.fit(dtrain[dtrain_predictors], dtrain['label'])
    
    print(gsearch.best_params_)
    print('')
    
    alg = gsearch.best_estimator_
    
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[dtrain_predictors])
    dtrain_predprob = alg.predict_proba(dtrain[dtrain_predictors])[:,1]
    
    # Dev prediction
    dev_predictors = [x for x in ddev.columns if x not in ['label']]
    ddev_predictions = alg.predict(ddev[dev_predictors])
    ddev_predprob = alg.predict_proba(ddev[dev_predictors])[:,1]
    
    # Test prediction
    test_predictors = [x for x in dtest.columns if x not in ['label']]
    dtest_predictions = alg.predict(dtest[test_predictors])
    dtest_predprob = alg.predict_proba(dtest[test_predictors])[:,1]
      
    # Print model report:
    print "\nModel Report"
    print("AUC    (Train/Test):\t%f\t%f\t%f" % (roc_auc_score(dtrain['label'], dtrain_predprob), roc_auc_score(ddev['label'], ddev_predprob), roc_auc_score(dtest['label'], dtest_predprob)))
    print("F1     (Train/Test):\t%f\t%f\t%f" % (f1_score(dtrain['label'], dtrain_predictions), f1_score(ddev['label'], ddev_predictions), f1_score(dtest['label'], dtest_predictions)))
    print("Kappa  (Train/Test):\t%f\t%f\t%f" % (cohen_kappa_score(dtrain['label'], dtrain_predictions), cohen_kappa_score(ddev['label'], ddev_predictions), cohen_kappa_score(dtest['label'], dtest_predictions)))


def build_random_forest(path):
    # Read data
    train = pandas.read_csv(path + "train_1.0")
    dev = pandas.read_csv(path + "dev_1.0")
    test = pandas.read_csv(path + "test_1.0")
    
    train_predictors = [x for x in train.columns if x not in ['label']]
    dev_predictors = [x for x in dev.columns if x not in ['label']]
    test_predictors = [x for x in test.columns if x not in ['label']]
    
    clf = RandomForestClassifier(random_state=42,class_weight='balanced',
                                 max_features='log2',
                                 n_estimators=100,
                                 max_depth=5)
    
    clf.fit(train[train_predictors], train['label'])
    
    predictions = clf.predict_proba(train[train_predictors])
    predictions = [ele[1] for ele in predictions]
    
    # print("There is a total of %d testing records." % test['label'].size)
    print("Random Forest AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f" % (roc_auc_score(train['label'], predictions),
                                                                    f1_score(train['label'],  [round(ele) for ele in predictions]),
                                                                    cohen_kappa_score(train['label'], [round(ele) for ele in predictions])))
    
    predictions = clf.predict_proba(dev[dev_predictors])
    predictions = [ele[1] for ele in predictions]
    
    # print("There is a total of %d testing records." % test['label'].size)
    print("Random Forest AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f" % (roc_auc_score(dev['label'], predictions),
                                                                    f1_score(dev['label'],  [round(ele) for ele in predictions]),
                                                                    cohen_kappa_score(dev['label'], [round(ele) for ele in predictions])))
    predictions = clf.predict_proba(test[dev_predictors])
    predictions = [ele[1] for ele in predictions]
    
    # print("There is a total of %d testing records." % test['label'].size)
    print("Random Forest AUC/F1 Score/Kappa (Test):\t%f\t%f\t%f" % (roc_auc_score(test['label'], predictions),
                                                                    f1_score(test['label'],  [round(ele) for ele in predictions]),
                                                                    cohen_kappa_score(test['label'], [round(ele) for ele in predictions])))


def main():
    
    path = './data/'
    parameter_tuning_random_forest(path)
    # build_random_forest(path)


if __name__ == '__main__':
    main()
