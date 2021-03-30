import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def base_model(X_train, y_train):
    clf = LogisticRegression(C = 1, random_state = 42, class_weight = "balanced")
    clf.fit(X_train, y_train)    
    return clf

def custom_model(X_train, y_train): 
    svc_cls = LinearSVC(C=.6, class_weight='balanced', random_state=42)
    svc_cls.fit(X_train, y_train)
    return svc_cls

def grid_search(X_train, y_train):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [.1, .3, .5, .7,.9, 1, 10, 20, 50, 100, 1000]},               
                    {'kernel': ['linear'], 'class_weight':['balanced'],'C': [.1, .3, .5, .7,.9, 1, 10, 20, 50, 100, 1000]}]

    scores = ['recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set :")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    return clf
