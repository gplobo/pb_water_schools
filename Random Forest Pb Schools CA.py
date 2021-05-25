#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages used in all code
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.utils import resample
from random import seed
from random import randint
from sklearn.metrics import roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from numpy import argmax
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import os
import seaborn as sns
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier

##############################################################################################
#resample_features
#resamples data with Pb violations to match complient data 
##############################################################################################


def resample_features(features, labels, multiplier):
    zeros = np.where(labels == 0)
    ones = np.where(labels == 1)
    features_zero = features[zeros]
    features_one = features[ones]
    features_one = resample(features_one, replace=True,n_samples=len(features_zero)*multiplier, 
                            random_state=42)
    features_zero = np.insert(features_zero, 0, 0, axis=1)
    features_one = np.insert(features_one, 0, 1, axis=1)
    features_final = np.append(features_zero, features_one, axis=0)
    return features_final

##############################################################################################
#Random number generator
#returns a randomly generated array with the same length as the train labels by using the test 
#labels to generate a prob distribution
##############################################################################################


def random_number_gen(train_labels, test_labels):
    seed(2)
    prob_Pb = round(sum(train_labels)/len(train_labels),2)*100
    random_gen=[]
    for i in range(test_labels.shape[0]):
        random_num = randint(0, 100)
        if random_num<=prob_Pb:
            result = 1
        else:
            result = 0
        random_gen.append(result)
    return random_gen

##############################################################################################
#Drop column score
#calculates OOB score or ROC AUC score as features are dropped out of the model
##############################################################################################
def drop_col_feat_imp(model, X_train, y_train, X_test, y_test, mode):
    model_clone = model
    importances = []
    if mode == 'OOB':
        model_clone.fit(X_train, y_train)
        benchmark_score = model_clone.oob_score_
        for col in range(X_train.shape[1]):
            X_train_drop = np.delete(X_train, col, 1)
            X_test_drop = np.delete(X_test, col, 1)
            model_clone.fit(X_train_drop, y_train)
            drop_score = model_clone.oob_score_
            importances.append(benchmark_score - drop_score)

        importances_df = importances
    
    elif mode == 'AUC':
        model_clone.fit(X_train, y_train)
        predicted_proba = model_clone.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicted_proba[:,1])
        roc_auc = auc(fpr, tpr)
        benchmark_score = roc_auc
        for col in range(X_train.shape[1]):
            X_train_drop = np.delete(X_train, col, 1)
            X_test_drop = np.delete(X_test, col, 1)
            model_clone.fit(X_train_drop, y_train)
            predicted_proba = model_clone.predict_proba(X_test_drop)
            fpr, tpr, thresholds = roc_curve(y_test, predicted_proba[:,1])
            roc_auc = auc(fpr, tpr)
            drop_score = roc_auc
            importances.append(benchmark_score - drop_score)

        importances_df = importances
    
    return importances_df, benchmark_score

##############################################################################################
#Preprocess data
#preprocessess data for use in random forest by transformin categorical data and getting rid of text
#option to normalize and to chose Pb violation threshold 
#normalize if normalize = True. Threshold may be 5, 10 or 15
##############################################################################################
def preprocess_data(df, Pb_threshold, normalize):
    #Obtain features and labels from imported file
    if Pb_threshold == 5:
        labels = df['ProbCategory_5_ppb']
    elif Pb_threshold == 10:
        labels = df['ProbCategory_10_ppb']
    elif Pb_threshold == 15:
        labels = df['ProbCategory_15_ppb']
    features = df
    
    #drop unnecessary columns
    #just_dummies = pd.get_dummies(features['WaterSystemName'])
    features.drop('ProbCategory_15_ppb', inplace = True, axis = 1)
    features.drop('ProbCategory_10_ppb', inplace = True, axis = 1)
    features.drop('ProbCategory_5_ppb', inplace = True, axis = 1)
    features.drop('RESULT', inplace = True, axis = 1)
    features.drop('DISINFECTANT', inplace = True, axis = 1)
    features.drop(features.iloc[:, 0:6], inplace = True, axis = 1) 
    #features = pd.concat([features, just_dummies], axis=1, sort=False)

    # use Imputer to fill in missing feature data with the mean of the column
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    missing_values= imputer.fit(features)
    features = imputer.transform(features)

    #normalize data (optional)
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(features)
        features = x_scaled
    print(features.shape)
    return features, labels


# In[4]:


##############################################################################################
#Random forest model - predicts whether Pb leaching is likely to occur in a school based on 
#water quality and socio-economic data
##############################################################################################

#read excel file with features (inputs) and labels (outputs)
Pb_data = pd.read_excel('CA water+social+spatial with Bay Area.xlsx')

#preprocess data
features, labels = preprocess_data(Pb_data, 15, True)
# Using Skicit-learn to split data into training and testing sets

train_features, test_features, train_labels, test_labels= train_test_split(features, 
    labels, test_size = 0.3, random_state = 42)

train_features = resample_features(train_features, train_labels,2)
train_labels = train_features[:,0]
train_features = np.delete(train_features, 0, axis=1)

test_features = resample_features(test_features, test_labels,2)
test_labels = test_features[:,0]
test_features = np.delete(test_features, 0, axis=1)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Implement random forest on train data
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42, criterion='entropy', oob_score=True, class_weight={0:1,1:100})
#rf = RandomForestClassifier()
rf.fit(train_features, train_labels)
threshold = 0.5
predicted_proba = rf.predict_proba(test_features)
#predict using the test data
#rf_prediction = rf.predict(test_features)
rf_prediction = (predicted_proba [:,1] >= threshold).astype('int')


# In[5]:


#plot ROC curve
# get false and true positive rates
fpr, tpr, thresholds = roc_curve(test_labels, predicted_proba[:,1])
# get area under the curve
roc_auc = auc(fpr, tpr)
# PLOT ROC curve
plt.figure(dpi=150)
plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
plt.title('ROC Curve for RF classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()

#plot precision-recall curve

# get precision and recall values
precision, recall, thresholds = precision_recall_curve(test_labels, predicted_proba[:,1])
# average precision score
avg_precision = average_precision_score(test_labels, predicted_proba[:,1])
# precision auc
pr_auc = auc(recall, precision)
#chose best threshold based on f score
zeros = np.where(precision == 0)
precision = np.delete(precision, zeros)
recall = np.delete(recall, zeros)
thresholds= np.delete(thresholds, zeros)
fscore = (2 * precision * recall) / (precision + recall)
fscore = fscore[np.logical_not(np.isnan(fscore))]
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# plot
plt.figure(dpi=150)
plt.plot(recall, precision, lw=1, color='blue', label=f'AP={avg_precision:.3f}; AUC={pr_auc:.3f}')
#plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
plt.fill_between(recall, precision, -1, facecolor='lightblue', alpha=0.5)
plt.title('PR Curve for RF classifier')
plt.xlabel('Recall (TPR)')
plt.ylabel('Precision')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()


# In[6]:


##############################################################################################
#Data analysis
##############################################################################################
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
rfc_cv_score = cross_val_score(rf, features, labels, cv=cv, scoring='roc_auc')

#OOB score
rf_clone = clone(rf)
oob_features, oob_labels = shuffle(features,labels)
rf_clone.fit(oob_features, oob_labels)
OOB_score = rf_clone.oob_score_

#calculate random predictions using a random number generator to compare to RF data
random_prediction = random_number_gen(train_labels, test_labels)

print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')

print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print('\n')

print("=== Out of Bag Error ===")
print("OOB Score - Random Forest: ", OOB_score)
print('\n')

print("=== Data ===")
print('1:', sum(test_labels),'0:',len(test_labels) - sum(test_labels))
print('\n')

print("=== Confusion Matrix - Random Forest ===")
print(confusion_matrix(test_labels, rf_prediction, labels=[1,0]))
print('\n')

print("=== Confusion Matrix - Random Number Generator ===")
print(confusion_matrix(test_labels, random_prediction, labels=[1,0]))
print('\n')

print("=== Classification Report - Random Forest ===")
print(classification_report(test_labels, rf_prediction, labels=[1,0]))
print('\n')

print("=== Classification Report - Random Number Generator ===")
print(classification_report(test_labels, random_prediction, labels=[1,0]))
print('\n')


# In[7]:


#Figure 3
##############################################################################################
#ROC-AUC and PR-AUC for various splits
#obtains the ROC-AUC and PR-AUC values and charts for 100 runs of RF
##############################################################################################

ROCs = []
PRs = []
for state in range(500,5000,50):
    
    train_features, test_features, train_labels, test_labels= train_test_split(features, 
        labels, test_size = 0.2, random_state = state)

    train_features = resample_features(train_features, train_labels,2)
    train_labels = train_features[:,0]
    train_features = np.delete(train_features, 0, axis=1)

    test_features = resample_features(test_features, test_labels,2)
    test_labels = test_features[:,0]
    test_features = np.delete(test_features, 0, axis=1)
    
    rf_optimum = RandomForestClassifier()
    
    rf_optimum.fit(train_features, train_labels)
    probas = rf_optimum.predict_proba(test_features)
    
    #ROC curve
    # get false and true positive rates
    fpr, tpr, thresholds = roc_curve(test_labels, probas[:,1])
    # get area under the curve
    roc_auc = auc(fpr, tpr)
    ROCs.append(roc_auc)
    
    #PR curve
    precision, recall, thresholds = precision_recall_curve(test_labels, probas[:,1])
    # average precision score
    avg_precision = average_precision_score(test_labels, probas[:,1])
    # precision auc
    pr_auc = auc(recall, precision)
    PRs.append(pr_auc)
    
fig = plt.figure(figsize =(10, 7)) 
  
# Creating plot 
ROC_PR_data = [ROCs, PRs]
plt.boxplot(ROC_PR_data, labels = ['ROC AUC', 'PR AUC']) 
  
# show plot 
plt.show() 


# In[5]:


# create data for figure 3
ROC_PR_data = np.transpose(ROC_PR_data)
Fig_3_df = pd.DataFrame(ROC_PR_data, columns = ['ROC AUC','PR AUC'])
Fig_3_df.to_excel('/Users/gabriel/Box Sync/UC Berkeley/Lead in drinking water/Manuscript/Figures/Fig_3_CA_raw.xlsx')


# In[38]:


##############################################################################################
#Feature importance analysis
#
##############################################################################################
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_head = Pb_data.columns
feature_head= feature_head[indices]
# Print the feature ranking
print("Feature ranking:")

# Plot the impurity-based feature importances of the forest
plt.figure(dpi=150)
plt.title("Feature importances")
plt.bar(range(train_features.shape[1]), importances[indices],
        color="r")
plt.xticks(range(train_features.shape[1]), feature_head, rotation='vertical')
plt.xlim([-1, train_features.shape[1]])
plt.show()

Feature_imp_data = [feature_head,importances[indices]]
Feature_imp_data = np.transpose(Feature_imp_data)
Fig_5_df = pd.DataFrame(Feature_imp_data, columns = ['Feature','Importance'])
Fig_5_df.to_excel('/Users/gabriel/Box Sync/UC Berkeley/Lead in drinking water/Manuscript/Figures/Fig_5_CA_raw.xlsx')


# In[39]:


# LOGISTIC REGRESSION FEATURE IMPORTANCE
# Instantiate, fit and obtain accuracy score
logit_model = LogisticRegression()
logit_model = logit_model.fit(train_features, train_labels)

print(logit_model.coef_)
print(Pb_data.columns)
# Examine the coefficients
importances = logit_model.coef_
importances = np.array(importances)
importances= importances.flatten()
feature_head = Pb_data.columns
feature_head= feature_head[indices]
# Print the feature ranking
print("Feature ranking:")

# Plot the impurity-based feature importances of the forest
plt.figure(dpi=150)
plt.title("Feature importances")
plt.bar(range(train_features.shape[1]), importances[indices],
        color="r")
plt.xticks(range(train_features.shape[1]), feature_head, rotation='vertical')
plt.xlim([-1, train_features.shape[1]])
plt.show()


# In[ ]:


feature_head_drop = Pb_data.columns
mins_index = []
mins = []
drop_list = []
benchmarks = []
##train_features_drop = train_features
train_features_drop = features
##train_labels_drop = train_labels
train_labels_drop = labels
test_features_drop = test_features
test_labels_drop = test_labels
rf_clone = clone(rf)

for col in range(train_features.shape[1]-1):
    importances, benchmark_score = drop_col_feat_imp(rf_clone, train_features_drop, 
                                                     train_labels_drop, test_features_drop, 
                                                     test_labels_drop, 'OOB')
    print(benchmark_score)
    min_pos = np.argmin(importances)
    mins_index.append(min_pos)
    mins.append(np.amin)
    benchmarks.append(benchmark_score)
    drop_list.append(feature_head_drop[min_pos])
    train_features_drop = np.delete(train_features_drop, min_pos, 1)
    test_features_drop = np.delete(test_features_drop, min_pos, 1)
    feature_head_drop = np.delete(feature_head_drop, min_pos, 0)
last_feature = feature_head_drop
drop_list.append(last_feature[0])
benchmarks.append(0)

last_feature = feature_head_drop[0]
drop_list.append(last_feature)
benchmarks.append(0)

plt.figure(dpi=150)
plt.title("Feature importances - OOB")
plt.bar(drop_list, benchmarks, color="r")
plt.xticks(range(train_features.shape[1]), drop_list, rotation='vertical')
plt.xlim([-1, train_features.shape[1]])
plt.show()


# In[ ]:


feature_head_drop = Pb_data.columns
mins_index = []
mins = []
drop_list = []
benchmarks = []
train_features_drop = train_features
train_labels_drop = train_labels
test_features_drop = test_features
test_labels_drop = test_labels
rf_clone = clone(rf)

for col in range(train_features.shape[1]-1):
    importances, benchmark_score = drop_col_feat_imp(rf_clone, train_features_drop, train_labels_drop, 
                                                     test_features_drop, test_labels_drop, 'AUC')
    print(benchmark_score)
    min_pos = np.argmin(importances)
    mins_index.append(min_pos)
    mins.append(np.amin)
    benchmarks.append(benchmark_score)
    drop_list.append(feature_head_drop[min_pos])
    train_features_drop = np.delete(train_features_drop, min_pos, 1)
    test_features_drop = np.delete(test_features_drop, min_pos, 1)
    feature_head_drop = np.delete(feature_head_drop, min_pos, 0)
last_feature = feature_head_drop
drop_list.append(last_feature[0])
benchmarks.append(0)

last_feature = feature_head_drop[0]
drop_list.append(last_feature)
benchmarks.append(0)

plt.figure(dpi=150)
plt.title("Feature importances - AUC")
plt.bar(drop_list, benchmarks, color="r")
plt.xticks(range(train_features.shape[1]), drop_list, rotation='vertical')
plt.xlim([-1, train_features.shape[1]])
plt.show()


# In[ ]:


##############################################################################################
#model selection analysis
#obtains cross validation scores (AUC) for 6 machine learning models
##############################################################################################

x_train = train_features
y_train = train_labels
x_test = test_features
y_test = test_labels
cross_val_features, cross_val_labels = shuffle(features,labels)


# LOGISTIC REGRESSION
# Instantiate, fit and obtain accuracy score
logit_model = LogisticRegression()
logit_model = logit_model.fit(x_train, y_train)
logit_model.score(x_train, y_train)

# Predictions on the test dataset
predicted = pd.DataFrame(logit_model.predict(x_test))

# Probabilities on the test dataset
probs = pd.DataFrame(logit_model.predict_proba(x_test))

# Store metrics
logit_accuracy = metrics.accuracy_score(y_test, predicted)
logit_roc_auc = metrics.roc_auc_score(y_test, probs[1])
logit_confus_matrix = metrics.confusion_matrix(y_test, predicted)
logit_classification_report = metrics.classification_report(y_test, predicted)
logit_precision = metrics.precision_score(y_test, predicted, pos_label=1)
logit_recall = metrics.recall_score(y_test, predicted, pos_label=1)
logit_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('Logistic confusion matrix:')
print(logit_confus_matrix)

# Evaluate the model using 10-fold cross-validation
logit_cv_scores = cross_val_score(LogisticRegression(), cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
logit_cv_mean = np.mean(logit_cv_scores)
logit_cv_std = np.std(logit_cv_scores)

# DECISION TREE (pruned to depth of 3)

# Instantiate with a max depth of 3
tree_model = tree.DecisionTreeClassifier(max_depth=3)
# Fit a decision tree
tree_model = tree_model.fit(x_train, y_train)
# Training accuracy
tree_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(tree_model.predict(x_test))
probs = pd.DataFrame(tree_model.predict_proba(x_test))

# Store metrics
tree_accuracy = metrics.accuracy_score(y_test, predicted)
tree_roc_auc = metrics.roc_auc_score(y_test, probs[1])
tree_confus_matrix = metrics.confusion_matrix(y_test, predicted)
tree_classification_report = metrics.classification_report(y_test, predicted)
tree_precision = metrics.precision_score(y_test, predicted, pos_label=1)
tree_recall = metrics.recall_score(y_test, predicted, pos_label=1)
tree_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('Tree confusion matrix:')
print(tree_confus_matrix)

# Evaluate the model using 10-fold cross-validation
tree_cv_scores = cross_val_score(tree.DecisionTreeClassifier(max_depth=3),
                                cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
tree_cv_mean = np.mean(tree_cv_scores)
tree_cv_std = np.std(tree_cv_scores)

# RANDOM FOREST
# Instantiate
rf = RandomForestClassifier()
# Fit
rf_model = rf.fit(x_train, y_train)
# Training accuracy
rf_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(rf_model.predict(x_test))
probs = pd.DataFrame(rf_model.predict_proba(x_test))

# Store metrics
rf_accuracy = metrics.accuracy_score(y_test, predicted)
rf_roc_auc = metrics.roc_auc_score(y_test, probs[1])
rf_confus_matrix = metrics.confusion_matrix(y_test, predicted)
rf_classification_report = metrics.classification_report(y_test, predicted)
rf_precision = metrics.precision_score(y_test, predicted, pos_label=1)
rf_recall = metrics.recall_score(y_test, predicted, pos_label=1)
rf_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('Random forest confusion matrix:')
print(rf_confus_matrix)

# Evaluate the model using 10-fold cross-validation
rf_cv_scores = cross_val_score(RandomForestClassifier(), cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
rf_cv_mean = np.mean(rf_cv_scores)
rf_cv_std = np.std(rf_cv_scores)

# SUPPORT VECTOR MACHINE
# Instantiate
svm_model = SVC(probability=True)
# Fit
svm_model = svm_model.fit(x_train, y_train)
# Accuracy
svm_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(svm_model.predict(x_test))
probs = pd.DataFrame(svm_model.predict_proba(x_test))

# Store metrics
svm_accuracy = metrics.accuracy_score(y_test, predicted)
svm_roc_auc = metrics.roc_auc_score(y_test, probs[1])
svm_confus_matrix = metrics.confusion_matrix(y_test, predicted)
svm_classification_report = metrics.classification_report(y_test, predicted)
svm_precision = metrics.precision_score(y_test, predicted, pos_label=1)
svm_recall = metrics.recall_score(y_test, predicted, pos_label=1)
svm_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('SVM confusion matrix:')
print(svm_confus_matrix)

# Evaluate the model using 10-fold cross-validation
svm_cv_scores = cross_val_score(SVC(probability=True), cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
svm_cv_mean = np.mean(svm_cv_scores)
svm_cv_std = np.std(svm_cv_scores)

# KNN
# Instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
# Fit the model
knn_model.fit(x_train, y_train)
# Accuracy
knn_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(knn_model.predict(x_test))
probs = pd.DataFrame(knn_model.predict_proba(x_test))

# Store metrics
knn_accuracy = metrics.accuracy_score(y_test, predicted)
knn_roc_auc = metrics.roc_auc_score(y_test, probs[1])
knn_confus_matrix = metrics.confusion_matrix(y_test, predicted)
knn_classification_report = metrics.classification_report(y_test, predicted)
knn_precision = metrics.precision_score(y_test, predicted, pos_label=1)
knn_recall = metrics.recall_score(y_test, predicted, pos_label=1)
knn_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('knn confusion matrix')
print(knn_confus_matrix)

# Evaluate the model using 10-fold cross-validation
knn_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
knn_cv_mean = np.mean(knn_cv_scores)
knn_cv_std = np.std(knn_cv_scores)

# TWO CLASS BAYES
# Instantiate
bayes_model = GaussianNB()
# Fit the model
bayes_model.fit(x_train, y_train)
# Accuracy
bayes_model.score(x_train, y_train)

# Predictions/probs on the test dataset
predicted = pd.DataFrame(bayes_model.predict(x_test))
probs = pd.DataFrame(bayes_model.predict_proba(x_test))

# Store metrics
bayes_accuracy = metrics.accuracy_score(y_test, predicted)
bayes_roc_auc = metrics.roc_auc_score(y_test, probs[1])
bayes_confus_matrix = metrics.confusion_matrix(y_test, predicted)
bayes_classification_report = metrics.classification_report(y_test, predicted)
bayes_precision = metrics.precision_score(y_test, predicted, pos_label=1)
bayes_recall = metrics.recall_score(y_test, predicted, pos_label=1)
bayes_f1 = metrics.f1_score(y_test, predicted, pos_label=1)

print('bayes matrix')
print(bayes_confus_matrix)

# Evaluate the model using 10-fold cross-validation
bayes_cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), cross_val_features, cross_val_labels, scoring='roc_auc', cv=10)
bayes_cv_mean = np.mean(bayes_cv_scores)
bayes_cv_std = np.std(bayes_cv_scores)

# EVALUATE
# Model comparison
models = pd.DataFrame({
    'Model'    : ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'kNN', 'Bayes'],
    'Accuracy' : [logit_accuracy, tree_accuracy, rf_accuracy, svm_accuracy, knn_accuracy, bayes_accuracy],
    'Precision': [logit_precision, tree_precision, rf_precision, svm_precision, knn_precision, bayes_precision],
    'recall'   : [logit_recall, tree_recall, rf_recall, svm_recall, knn_recall, bayes_recall],
    'F1'       : [logit_f1, tree_f1, rf_f1, svm_f1, knn_f1, bayes_f1],
    'cv_precision' : [logit_cv_mean, tree_cv_mean, rf_cv_mean, svm_cv_mean, knn_cv_mean, bayes_cv_mean],
    'ROC AUC'       : [logit_roc_auc, tree_roc_auc, rf_roc_auc, svm_roc_auc, knn_roc_auc, bayes_roc_auc],
    'Cross Validation':[logit_cv_mean, tree_cv_mean, rf_cv_mean, svm_cv_mean, knn_cv_mean, bayes_cv_mean],
    'CV std':[logit_cv_std, tree_cv_std, rf_cv_std, svm_cv_std, knn_cv_std, bayes_cv_std]
    })
models.sort_values(by='Precision', ascending=False)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(features, labels)


# In[ ]:


rf_opt = RandomForestClassifier(n_estimators = 400, random_state = 42, min_samples_split = 2, criterion='entropy', 
                       min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, 
                       bootstrap = True, oob_score=True)
rf_opt.fit(features,labels)
print(rf_opt.oob_score_)


# In[15]:


# Generate OOBs as a function of number of variables
from sklearn.utils import shuffle
features_shuffle, labels_shuffle = shuffle(features, labels, random_state=42)
OOBs = []
for i in range(10,features_shuffle.shape[0],10):
    if i <= features_shuffle.shape[0]:
        rf_opt = RandomForestClassifier(n_estimators = 400, random_state = 42, min_samples_split = 2, criterion='entropy', 
                       min_samples_leaf = 4, max_features = 'sqrt', max_depth = 10, 
                       bootstrap = True, oob_score=True)
        rf_opt.fit(features[:i,:], labels[:i])
        oob = rf_opt.oob_score_
        OOBs.append(oob)


# In[ ]:


# PLOT OOB curve and save data
plt.figure(dpi=150)
x = range(10,features_shuffle.shape[0],10)
OOBs = OOBs
plt.plot(x, OOBs, lw=1, color='red')
#plt.title('Out of bag accuracy for RF classifier')
plt.xlabel('Number of data points')
plt.ylabel('Out of bag accuracy')
plt.xlim([10, 4500])
plt.ylim([0, 1])
#plt.legend()
plt.show()
# create data for figure 3
OOB_data = [x, OOBs]
OOB_data = np.transpose(OOB_data)
Fig_3_df = pd.DataFrame(OOB_data, columns = ['Number of data points','OOB'])
Fig_3_df.to_excel('/Users/gabriel/Box Sync/UC Berkeley/Lead in drinking water/Manuscript/Figures/Fig_4_CA_raw.xlsx')


# In[ ]:





# In[45]:



#CSMR DEPENDENCE PLOT
dependence_data = pd.read_excel('CA partial dependence plot.xlsx', sheet_name = 'CSMR')

#Obtain features and labels from imported file

features_d = dependence_data
features_d.drop('ProbCategory_15_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_10_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_5_ppb', inplace = True, axis = 1)
features_d.drop('RESULT', inplace = True, axis = 1)
features_d.drop('DISINFECTANT', inplace = True, axis = 1)
features_d.drop(features_d.iloc[:, 0:6], inplace = True, axis = 1) 

#predict output using rf
predicted_proba = rf.predict_proba(features_d)
plt.figure(dpi=150)
plt.title("Partial dependence plot - CSMR")
X = np.array(features_d['CSMR'])
Y = np.array(predicted_proba[:,1])
plt.plot(X, Y, lw=1, color='red')
plt.xlabel('CSMR')
plt.ylabel('Probability')
#plt.xlim([0, 7])
#plt.ylim([0, 1])

plt.show()

#DISTANCE TO NEAREST SCHOOL DEPENDENCE PLOT
dependence_data = pd.read_excel('CA partial dependence plot.xlsx', sheet_name = 'Nearest school')

#Obtain features and labels from imported file

features_d = dependence_data
features_d.drop('ProbCategory_15_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_10_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_5_ppb', inplace = True, axis = 1)
features_d.drop('RESULT', inplace = True, axis = 1)
features_d.drop('DISINFECTANT', inplace = True, axis = 1)
features_d.drop(features_d.iloc[:, 0:6], inplace = True, axis = 1) 

#predict output using rf
predicted_proba = rf.predict_proba(features_d)
plt.figure(dpi=150)
plt.title("Partial dependence plot - Nearest school")
X = np.array(features_d['Neares_school_Pb'])
Y = np.array(predicted_proba[:,1])
plt.plot(X, Y, lw=1, color='red')
plt.xlabel('Neares school above MCL (m)')
plt.ylabel('Probability')
#plt.xlim([6000, 10000])
#plt.ylim([0, 1])

plt.show()


#POVERTY
dependence_data = pd.read_excel('CA partial dependence plot.xlsx', sheet_name = 'Poverty')

#Obtain features and labels from imported file

features_d = dependence_data
features_d.drop('ProbCategory_15_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_10_ppb', inplace = True, axis = 1)
features_d.drop('ProbCategory_5_ppb', inplace = True, axis = 1)
features_d.drop('RESULT', inplace = True, axis = 1)
features_d.drop('DISINFECTANT', inplace = True, axis = 1)
features_d.drop(features_d.iloc[:, 0:6], inplace = True, axis = 1) 

#predict output using rf
predicted_proba = rf.predict_proba(features_d)
plt.figure(dpi=150)
plt.title("Partial dependence plot - poverty")
X = np.array(features_d['Percent_Black'])
Y = np.array(predicted_proba[:,1])
plt.plot(X, Y, lw=1, color='red')
plt.xlabel('Percentage below poverty level')
plt.ylabel('Probability')
#plt.xlim([0, 50])
#plt.ylim([0, 1])

plt.show()


# In[6]:


names = ['all','water_social', 'social_spatial', 'spatial_water']
AUC_variables = []
for sheetname in names:

#read excel file with features (inputs) and labels (outputs)
    Pb_data = pd.read_excel('CA separate variables.xlsx', sheet_name = sheetname )

    #Obtain features and labels from imported file
    labels_s = Pb_data['ProbCategory_5_ppb']
    features_s = Pb_data
    features_s.drop('ProbCategory_15_ppb', inplace = True, axis = 1)
    features_s.drop('ProbCategory_10_ppb', inplace = True, axis = 1)
    features_s.drop('ProbCategory_5_ppb', inplace = True, axis = 1)
    features_s.drop('RESULT', inplace = True, axis = 1)
    features_s.drop('DISINFECTANT', inplace = True, axis = 1)
    features_s.drop(Pb_data.iloc[:, 0:6], inplace = True, axis = 1) 

    # use Imputer to fill in missing feature data with the mean of the column
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    missing_values= imputer.fit(features_s)
    features_s = imputer.transform(features_s)
    
    random_states = range(20,200,5)
    AUCs = []
    for r in random_states:
        # Using Skicit-learn to split data into training and testing sets
        train_features_s, test_features_s, train_labels_s, test_labels_s= train_test_split(features_s, 
            labels_s, test_size = 0.3, random_state = r)

        #resample
        train_features_s = resample_features(train_features_s, train_labels_s,2)
        train_labels_s = train_features_s[:,0]
        train_features_s = np.delete(train_features_s, 0, axis=1)

        test_features_s = resample_features(test_features_s, test_labels_s,2)
        test_labels_s = test_features_s[:,0]
        test_features_s = np.delete(test_features_s, 0, axis=1)

        rf = RandomForestClassifier()
        rf.fit(train_features_s, train_labels_s)
        predicted_proba_s = rf.predict_proba(test_features_s)
        rf_roc_auc_1 = metrics.roc_auc_score(test_labels_s, predicted_proba_s[:,1])
        AUCs.append(rf_roc_auc_1)

    print(sheetname, ':', np.average(AUCs), np.std(AUCs))
    AUC_variables.append(AUCs)


# In[8]:


#Saves and plots data for Figure 6
plt.boxplot(AUC_variables, labels = names) 
# show plot 
plt.show() 

# create data for figure 6
ROC_data = np.transpose(AUC_variables)
Fig_6_df = pd.DataFrame(ROC_data, columns = names)
Fig_6_df.to_excel('/Users/gabriel/Box Sync/UC Berkeley/Lead in drinking water/Manuscript/Figures/Fig_6_CA_raw.xlsx')


# In[5]:


AUC_variables


# In[42]:


##############################################################################################
#Bay Area predictions - predicts whether Pb leaching is likely to occur in the Bay based on 
#water quality and socio-economic data
##############################################################################################

#read excel file with features (inputs) and labels (outputs)
Pb_data = pd.read_excel('CA water+social+spatial with Bay Area.xlsx', sheet_name = 'Train data')
Bay_data = pd.read_excel('CA water+social+spatial with Bay Area.xlsx', sheet_name = 'Bay data')


#preprocess data
features, labels = preprocess_data(Pb_data, 5, True)
test_features, test_labels = preprocess_data(Bay_data, 5, True)
# Using Skicit-learn to split data into training and testing sets

features = resample_features(features, labels,2)
col_list = (features.append([df2,df3])).columns.tolist()
labels = features[:,0]
features = np.delete(features, 0, axis=1)

print('Features Shape:', features.shape)
print('Labels Shape:', labels.shape)

# Implement random forest on train data
rf = RandomForestClassifier(n_estimators = 500, random_state = 42, criterion='entropy', oob_score=True, class_weight={0:1,1:100})

rf.fit(features, labels)

predicted_proba = rf.predict_proba(test_features)
rf_prediction_df = pd.DataFrame(data = predicted_proba, columns = ['Prob no Pb', 'Prob Pb'])
rf_prediction_df.to_excel('Bay Area predictions_RF.xlsx')


# In[18]:


##############################################################################################
#Bay Area predictions - predicts whether Pb leaching is likely to occur in the Bay based on 
#water quality and socio-economic data
##############################################################################################

#read excel file with features (inputs) and labels (outputs)
Pb_data = pd.read_excel('CA water+social+spatial with Bay Area.xlsx', sheet_name = 'Train data')
Bay_data = pd.read_excel('CA water+social+spatial with Bay Area.xlsx', sheet_name = 'Bay data')


#preprocess data
features, labels = preprocess_data(Pb_data, 5, True)
test_features, test_labels = preprocess_data(Bay_data, 5, True)
# Using Skicit-learn to split data into training and testing sets

features = resample_features(features, labels,2)
labels = features[:,0]
features = np.delete(features, 0, axis=1)

print('Features Shape:', features.shape)
print('Labels Shape:', labels.shape)

# Implement random forest on train data
rf = RandomForestClassifier(n_estimators = 500, random_state = 42, criterion='entropy', oob_score=True, class_weight={0:1,1:100})

rf.fit(features, labels)

predicted_proba = rf.predict_proba(test_features)
rf_prediction_df = pd.DataFrame(data = predicted_proba, columns = ['Prob no Pb', 'Prob Pb'])
rf_prediction_df.to_excel('Bay Area predictions_RF_Bay_only.xlsx')


# In[32]:


##############################################################################################
#Run model for all the data and calculate average Pb probability per school
##############################################################################################

#read excel file with features (inputs) and labels (outputs)
Pb_data_prob = pd.read_excel('CA water+social+spatial with Bay Area + students.xlsx')
Pb_data_students = pd.read_excel('CA water+social+spatial with Bay Area + students.xlsx', sheet_name = 'CA with schools')

#preprocess data
features_total, labels_total = preprocess_data(Pb_data_prob, 15, True)
# Using Skicit-learn to split data into training and testing sets
student_count = Pb_data_students['Estimate__']

# Implement trained random forest on original data
rf.fit(features_total, labels_total)
threshold = 0.5
predicted_proba_total = rf.predict_proba(features_total)[:,1]

d = {'SchoolName': Pb_data_students['SchoolName'], 'Num students': student_count, 'Prob':predicted_proba_total}
df_prob = pd.DataFrame(data=d)
df_prob = df_prob.groupby(['SchoolName']).mean()

average_prob = np.sum(df_prob['Num students']*df_prob['Prob'])/np.sum(df_prob['Num students'])
print('Average Pb probability: ', average_prob)


# In[31]:


average_prob = np.sum(df_prob['Num students']*df_prob['Prob'])/np.sum(df_prob['Num students'])


# In[22]:


student_count


# In[23]:


predicted_proba_total


# In[ ]:




