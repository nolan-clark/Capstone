# Fit predictors to models and generate performance metrics

import pandas as pd
import torch
from sklearn.model_selection import train_test_split, cross_val_score


import shap

from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


import xgboost

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def heat_it(clf,X_test,y_test):
    
    predicted = clf.predict(X_test)
    actual    = y_test

    cm = confusion_matrix(actual,predicted)
    
    #finding precision and recall
    accuracy = accuracy_score(actual,predicted)
    precision = precision_score(actual,predicted)
    recall = recall_score(actual,predicted)
    F1_score = f1_score(actual,predicted)
    
    
    collect= {'Accuracy'  : accuracy,
              'Precision' : precision,
              'Recall'    : recall,
              'F1_score'  : F1_score
        }
    

    return collect
    
def add_roc(clf, model_name, X_test, y_test):
    

    y_score =clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc=round(roc_auc_score(y_test, y_score),4)
    
    return plt.plot(fpr,tpr, label=f'{model_name}, AUC={auc}')

def add_dummy(clf, X_test, y_test):
    
    
    y_score =clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc=round(roc_auc_score(y_test, y_score),4)
    
    return plt.plot(fpr,tpr, label=f'Dummy Classifier, AUC={auc}', linestyle='--', color='black')


def fit_and_score(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


    collection= { 'Accuracy'  : [],
              'Precision' : [],
              'Recall'    : [],
              'F1_score'  : []
            }

    # Support Vector Machine
    svmc = svm.SVC(probability=True, random_state=42)
    svmc.fit(X_train, y_train)
    res = heat_it(svmc,X_test,y_test)
    for value in res:
        collection[value].append(res[value])
    
    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    res = heat_it(gnb,X_test,y_test)
    for value in res:
        collection[value].append(res[value])
    
    # Multi-Layer Perceptron
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    res = heat_it(mlp,X_test,y_test)
    for value in res:
        collection[value].append(res[value])
    
    # Decision Tree
    dtc = tree.DecisionTreeClassifier(random_state=42)
    dtc.fit(X_train,y_train)
    res = heat_it(dtc,X_test,y_test)
    for value in res:
        collection[value].append(res[value])
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train,y_train)
    res = heat_it(rf,X_test,y_test)
    for value in res:
        collection[value].append(res[value])
    
    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    res = heat_it(knn, X_test, y_test)
    for value in res:
        collection[value].append(res[value])
    
    # XGBoost
    # Use "hist" for constructing the trees, with early stopping enabled.
    xgb = xgboost.XGBClassifier(tree_method="hist", early_stopping_rounds=20, 
                                random_state=42, objective='binary:logistic')
    # Fit the model, test sets are used for early stopping.
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    res = heat_it(xgb, X_test, y_test)
    for value in res:
        collection[value].append(res[value])
        
    
    # Logistic Regression
    logR = LogisticRegression(random_state=42)
    logR.fit(X_train, y_train)
    res = heat_it(logR, X_test, y_test)
    for value in res:
        collection[value].append(res[value])
    
    # Dummy Classifier
    dummy_clf= DummyClassifier(random_state = 42, strategy='most_frequent')
    dummy_clf.fit(X_train,y_train)
    
    ds1=pd.DataFrame(collection)
    
    # label index
    models=[
            'Support Vector Machine',
            'Naive Bayes',
            'Multi-Layer Perceptron',
            'Decision Tree',
            'Random Forest',
            'KNN',
            'XGBoost',
            'Logistic Regression'
            ]
    ds1.index = models
    plt.figure(figsize=(10,6))


    add_roc(mlp, 'Multi-Layer Perceptron',X_test, y_test)
    add_roc(xgb,'XGBoost',X_test, y_test)
    add_roc(rf, 'Random Forest',X_test, y_test)
    add_roc(logR, 'Logistic Regression',X_test, y_test)
    add_roc(gnb, 'Naive Bayes',X_test, y_test)
    add_roc(svmc, 'Support Vector Machine',X_test, y_test)
    add_roc(knn, 'K-Nearest Neighbor',X_test, y_test)
    add_roc(dtc, 'Decision Tree',X_test, y_test)
    add_dummy(dummy_clf,X_test, y_test)
    
    
    plt.legend()
    plt.show()
    
    explainer=shap.Explainer(xgb)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)

    return ds1


# Generate Confusion Matrix with Performance metrics

def confusion_scores(clf,X_test,y_test):
    
    predicted = clf.predict(X_test)
    actual    = y_test

    cm = confusion_matrix(actual,predicted, normalize='true')
    
    # Visualize confusion matrix
    sns.heatmap(cm,
                annot=True,
                fmt='.2%',
                xticklabels=['HUMAN','LLM'],
                yticklabels=['HUMAN','LLM'],
                vmin=0, 
                vmax=1,
                cmap='Blues'
               )
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()
    #finding precision and recall
    accuracy = accuracy_score(actual,predicted)
    print("Accuracy   :", accuracy)
    precision = precision_score(actual,predicted)
    print("Precision :", precision)
    recall = recall_score(actual,predicted)
    print("Recall    :", recall)
    F1_score = f1_score(actual,predicted)
    print("F1-score  :", F1_score)







