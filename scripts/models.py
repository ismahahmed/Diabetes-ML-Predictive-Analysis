import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from scipy.spatial import distance

# Cleaning data and returning panads df
def clean_df(df):
    df['BloodPressure'].replace(0, pd.NA, inplace=True)
    df['Glucose'].replace(0, pd.NA, inplace=True)
    df['Insulin'].replace(0, pd.NA, inplace=True)
    df['BMI'].replace(0, pd.NA, inplace=True)
    df = df.dropna().reset_index(drop=True)
    return(df)

def get_training_test_sets(df, cols):
    y = df['Outcome']
    X = df[cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=4, stratify=y)
    return(X_train, X_test, y_train, y_test)

def cm(y_test, y_pred, filename):
    '''
    returns cm, TP, FN, FP, and TN
    '''
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

    plotcm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
    plotcm.plot()
    plt.savefig(f'figures/{filename}.png')

    return(cm, cm[0][0], cm[0][1], cm[1][0], cm[1][1])

# Model 1: Knn
def knn(X_train, y_train, X_test, k):
    # scaling the features since we are using Euclidean distance
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train) # fit and scales
    X_test_sc = scaler.transform(X_test) # do not re fit on test data

    # given count neighbors, train the classifier with k = neighborns
    classifier = KNeighborsClassifier(n_neighbors=k) 
    classifier.fit(X_train_sc,y_train)   

    # array for predictions
    predictions =  classifier.predict(X_test_sc)
    return(predictions)

def get_best_k(X_train, y_train, X_test, y_test):
    k_accuracy = {1:0, 2:0, 4:0, 8:0, 9:0, 10:0, 12:0}

    for k in [1, 2, 4, 8, 10, 9, 12]:
       pred = knn(X_train, y_train, X_test, k)
       k_accuracy[k] = accuracy_score(y_test, pred)
    
    max_key = max(k_accuracy, key=k_accuracy.get)
    
    return max_key

# Model 2: Logistic Regression
def logistic_regression(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train) # fit and scales. optional for log regression
    X_test_sc = scaler.transform(X_test) # do not re fit on test data

    classifier = LogisticRegression()
    classifier.fit(X_train_sc, y_train)

    predictions = classifier.predict(X_test_sc) # ndarry of predictions
    return predictions

# Model 3: Naive Bayesian
def nb_model(X_train, X_test, y_train):
    model = GaussianNB() 
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return(y_pred)

# Model 4: Decision Tree
def decisiontree(X_train, X_test, y_train):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=4)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    featurenames = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    class_name = ['0', '1']
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(model, feature_names= featurenames,class_names= class_name,filled=True) # https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
    fig.savefig("figures/decision_tree.png")

    return( y_pred)

# Model 5: SVM
def SVM(X_train, X_test, y_train, k):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test) 

    if k == 'poly':
        svm_classifier = svm.SVC(kernel=k, degree=3)
    else:
        svm_classifier = svm.SVC(kernel= k)
    svm_classifier.fit(X_train_sc, y_train)

    y_pred =  svm_classifier.predict(X_test_sc)

    return(y_pred)


def main():
    df = pd.read_csv('data/diabetes.csv')
    df = clean_df(df)
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    X_train, X_test, y_train, y_test = get_training_test_sets(df, cols)

    # knn
    knn_k = get_best_k(X_train, y_train, X_test, y_test)
    knn_pred = knn(X_train, y_train, X_test, knn_k)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_cm, knn_tp, knn_fp, knn_fn, knn_tn = cm(y_test, knn_pred, 'confusion matrix: knn')
    knn_tpr = knn_tp/(knn_tp+knn_fn)
    knn_tnr = knn_tn/(knn_tn+knn_fp)

    # logistic regression
    log_reg_pred = logistic_regression(X_train, y_train, X_test)
    log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
    log_cm, log_tp, log_fp, log_fn, log_tn = cm(y_test, log_reg_pred, 'confusion matrix: logistic regression')
    log_tpr = log_tp/(log_tp+log_fn)
    log_tnr = log_tn/(log_tn+log_fp)

    # Naive Bayes
    nb_pred = nb_model(X_train, X_test, y_train)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_cm, nb_tp, nb_fp, nb_fn, nb_tn = cm(y_test, nb_pred, 'confusion matrix: Naive Bayesian')
    nb_tpr = nb_tp/(nb_tp+nb_fn)
    nb_tnr = nb_tn/(nb_tn+nb_fp)

    # Decision Tree
    dt_pred = decisiontree(X_train, X_test, y_train)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_cm, dt_tp, dt_fp, dt_fn, dt_tn = cm(y_test, dt_pred, 'confusion matrix: Decision Tree')
    dt_tpr = dt_tp/(dt_tp+dt_fn)
    dt_tnr = dt_tn/(dt_tn+dt_fp)

    # SVM (linear)
    svm_linear_pred = SVM(X_train, X_test, y_train, 'linear')
    svm_linear_accuracy = accuracy_score(y_test, svm_linear_pred)
    svm_lin_cm, svm_lin_tp, svm_lin_fp, svm_lin_fn, svm_lin_tn = cm(y_test, svm_linear_pred, 'confusion matrix: SVM (Linear)')
    svm_lin_tpr = svm_lin_tp/(svm_lin_tp+svm_lin_fn)
    svm_lin_tnr = svm_lin_tn/(svm_lin_tn+svm_lin_fp)
    
    # SVM (gaussian)
    svm_rbf_pred = SVM(X_train, X_test, y_train, 'rbf')
    svm_rbf_accuracy = accuracy_score(y_test, svm_rbf_pred)
    svm_rbf_cm, svm_rbf_tp, svm_rbf_fp, svm_rbf_fn, svm_rbf_tn = cm(y_test, svm_rbf_pred, 'confusion matrix: SVM (Gaissian)')
    svm_rbf_tpr = svm_rbf_tp/(svm_rbf_tp+svm_rbf_fn)
    svm_rbf_tnr = svm_rbf_tn/(svm_rbf_tn+svm_rbf_fp)

    # SVM (poly)
    svm_poly_pred = SVM(X_train, X_test, y_train, 'poly')
    svm_poly_accuracy = accuracy_score(y_test, svm_poly_pred)
    svm_poly_cm, svm_poly_tp, svm_poly_fp, svm_poly_fn, svm_poly_tn = cm(y_test, svm_poly_pred, 'confusion matrix: SVM (poly)')
    svm_poly_tpr = svm_poly_tp/(svm_poly_tp+svm_poly_fn)
    svm_poly_tnr = svm_poly_tn/(svm_poly_tn+svm_poly_fp)

    table_dict = {'model': ['k-NN', 'Logistic Regression', 'Naive Baysian', 'Decision Tree', 'SVM (linear)', 'SVM (gaussian)', 'SVM (poly)'],
                 'TP' : [knn_tp, log_tp, nb_tp, dt_tp, svm_lin_tp, svm_rbf_tp, svm_poly_tp],
                 'FP' : [knn_fp, log_fp, nb_fp, dt_fp, svm_lin_fp, svm_rbf_fp, svm_poly_fp],
                 'TN' : [knn_tn, log_tn, nb_tn, dt_tn, svm_lin_tn, svm_rbf_tn, svm_poly_tn],
                 'FN' : [knn_fn, log_fn, nb_fn, dt_fn, svm_lin_fn, svm_rbf_fn, svm_poly_fn],
                 'TPR' : [knn_tpr, log_tpr, nb_tpr, dt_tpr, svm_lin_tpr, svm_rbf_tpr, svm_poly_tpr],
                 'TNR' : [knn_tnr, log_tnr, nb_tnr, dt_tnr, svm_lin_tnr, svm_rbf_tnr, svm_poly_tnr],
                 'Accuracy' : [knn_accuracy, log_reg_accuracy, nb_accuracy, dt_accuracy, svm_linear_accuracy, svm_rbf_accuracy, svm_poly_accuracy]
                 }
    
    summary_df = pd.DataFrame(table_dict)
    return(summary_df)

if __name__ == "__main__":
    main()