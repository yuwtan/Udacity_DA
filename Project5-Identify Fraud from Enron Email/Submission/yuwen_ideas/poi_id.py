#!/usr/bin/python

import sys
import pickle
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Remove outliers
data_dict.pop('TOTAL', 0);
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0);
data_dict['GLISAN JR BEN F']['shared_receipt_with_poi'] = \
    data_dict['GLISAN JR BEN F']['to_messages'];
my_dataset = data_dict


### Create new features
for person in my_dataset:

    # read dataset information
    data_point = my_dataset[person]

    from_poi_to_this_person = float(data_point['from_poi_to_this_person'])
    shared_receipt_with_poi = float(data_point['shared_receipt_with_poi'])
    to_messages = float(data_point['to_messages'])

    from_this_person_to_poi = float(data_point['from_this_person_to_poi'])
    from_messages = float(data_point['from_messages'])

    total_stock_value = float(data_point['total_stock_value'])
    total_payments = float(data_point['total_payments'])

    # create new features
    if math.isnan(from_poi_to_this_person) \
            or math.isnan(to_messages) \
            or to_messages == 0:
        ratio_from_poi_to_this_person = 0
    else:
        ratio_from_poi_to_this_person = from_poi_to_this_person / to_messages

    if math.isnan(from_this_person_to_poi) \
            or math.isnan(from_messages) \
            or from_messages == 0:
        ratio_from_this_person_to_poi = 0
    else:
        ratio_from_this_person_to_poi = from_this_person_to_poi / from_messages

    if math.isnan(shared_receipt_with_poi) \
            or math.isnan(to_messages) \
            or to_messages == 0:
        ratio_shared_receipt_with_poi = 0
    else:
        ratio_shared_receipt_with_poi = shared_receipt_with_poi / to_messages

    if math.isnan(total_stock_value) \
            or math.isnan(total_payments) \
            or total_payments == 0:
        ratio_stock_to_payments = 0
    else:
        ratio_stock_to_payments = total_stock_value / total_payments

    # save new features
    data_point['ratio_from_poi_to_this_person'] = ratio_from_poi_to_this_person
    data_point['ratio_from_this_person_to_poi'] = ratio_from_this_person_to_poi
    data_point['ratio_shared_receipt_with_poi'] = ratio_shared_receipt_with_poi
    data_point['ratio_stock_to_payments'] = ratio_stock_to_payments


### Feature selection from SelectKBest
features_list_all = ['poi',
                     'salary', 'deferral_payments',
                     'total_payments', 'loan_advances',
                     'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value',
                     'expenses', 'exercised_stock_options',
                     'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees',
                     'to_messages', 'from_poi_to_this_person',
                     'from_messages', 'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'ratio_from_poi_to_this_person',
                     'ratio_from_this_person_to_poi',
                     'ratio_shared_receipt_with_poi',
                     'ratio_stock_to_payments'
                     ]
data = featureFormat(my_dataset, features_list_all, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
selector = SelectKBest(f_classif)
selector.fit(features, labels)
indices = np.argsort(selector.scores_)[::-1]
for f in range(len(features[0])):
    print("%2d) %-*s %f" %
          (f+1, 30, features_list_all[indices[f]+1], selector.scores_[indices[f]]))


### Select what features to use
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'ratio_from_this_person_to_poi',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'ratio_shared_receipt_with_poi',
                 'total_payments'
                 ]


### Define a varity of classifiers
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()

from sklearn.svm import SVC
param_svm = {'kernel':('linear', 'rbf'), 'C':[1, 10, 1000]}
svr = SVC()
clf_svm = GridSearchCV(svr, param_svm)

from sklearn.linear_model import LogisticRegression
param_lr = {'penalty':('l1', 'l2'), 'C':[1, 10, 100]}
lr = LogisticRegression()
clf_lr = GridSearchCV(lr, param_lr)

from sklearn.tree import DecisionTreeClassifier
param_dt = {'max_depth':[4, 6, 8]}
dt = DecisionTreeClassifier()
clf_dt = GridSearchCV(dt, param_dt)

### Further define scaler and PCA
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1000, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.decomposition import PCA

### Define function to evaluate each classifier
def test_clf(features, labels, clf, PCA_k):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in sss.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        pca = PCA(n_components=PCA_k, whiten=True).fit(features_train)
        features_train = pca.transform(features_train)
        features_test = pca.transform(features_test)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1

    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    return total_predictions, accuracy, precision, recall, f1, f2


### Check if the new features created are helping to get better results
data = featureFormat(my_dataset, features_list[0:8], sort_keys = True)
labels, features = targetFeatureSplit(data)
print features_list[0:8]
print test_clf(features, labels, clf_nb, 4)

feature_list_temp =  features_list[0:5] + features_list[6:9]
print feature_list_temp
data = featureFormat(my_dataset, feature_list_temp, sort_keys = True)
labels, features = targetFeatureSplit(data)
print test_clf(features, labels, clf_nb, 4)


### Evaluate different classifiers
# print test_clf(features, labels, clf_nb, 3)
# print test_clf(features, labels, clf_lr, 3)
# print test_clf(features, labels, clf_svm, 3)
# print test_clf(features, labels, clf_dt, 3)
#
# print test_clf(features, labels, clf_nb, 6)
# print test_clf(features, labels, clf_lr, 6)
# print test_clf(features, labels, clf_svm, 6)
# print test_clf(features, labels, clf_dt, 6)
#
# print test_clf(features, labels, clf_nb, 9)
# print test_clf(features, labels, clf_lr, 9)
# print test_clf(features, labels, clf_svm, 9)
# print test_clf(features, labels, clf_dt, 9)


### Try to find the best parameters for Naive Bayes
for KBest_k in range(5, 11):
    for PCA_k in range(1, KBest_k-2):
        data = featureFormat(my_dataset, features_list[0:KBest_k-1], sort_keys = True)
        labels, features = targetFeatureSplit(data)
        total_predictions, accuracy, precision, recall, f1, f2 = \
            test_clf(features, labels, clf_nb, PCA_k)
        if precision > 0.3 and recall > 0.3:
            print "Naive Bayes: KBest set to ", KBest_k-1, "; PCA set to", PCA_k
            print test_clf(features, labels, clf_nb, PCA_k)


### Define the final classifier we will use
from sklearn.pipeline import Pipeline
estimators = [('scaler', MinMaxScaler()),
              ('reduce_dim', PCA(n_components=4, whiten=True)),
              ('clf', GaussianNB())]
clf = Pipeline(estimators)
features_list = features_list[0:8]


### Dump the classifier, dataset, and features_list so anyone can check the results
dump_classifier_and_data(clf, my_dataset, features_list)