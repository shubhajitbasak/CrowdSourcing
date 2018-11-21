# Authors: Niall Guerin
# Part 1: Script Function
# 1. Read the training sample (gold_sample.csv)
# 2. Extract the classification label values for training set, extract the LSA topic feature values for training
# 3. Configure decision tree and train on training sample
# 4. Parse in test.csv and drop id and convert pos/neg values to binary 0/1 for ground truth labelling
# 5. Test decision tree against test.csv
# 6. Construct confusion matrix, print accuracy, f-score, prediction probabilities, plot ROC curve.
# 7. Write accuracy, f-score, prediction probabilities to text file.
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
# import graphviz

# read the training sample from file
train = pd.read_csv("../data/gold_sample.csv", skiprows=0)
print("*********** Test ****************")
# print(head(train))

# check NaN values in the data frame
# Web References:
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.isnull.html
# https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
isNaN = train.isnull().any().any()
print("NaN Check Status (goal state is False) : ", isNaN)
# fix the NaN to 0 if isNaN is true: or use mean value as alternative?
if isNaN:
    train = train.fillna(0)
isNaN = train.isnull().any().any()
print("NaN Re-Check Status (goal state is False): ", isNaN)

# configure decision tree training features:
# omit two columns for feature training: id, classification labels as irrelevant for model training
# feature selection should comprise of LSA feature columns topic0:topic1999
feature_col_count = 1201
# convert this structure to a list so we can work with scikit learn functions later in decision tree
features = list(train.columns[1:feature_col_count])

# configure the classification labels from the data frame: use convention of y = classifier label values, x = feature values
y_label = train['class']
x_features = train[features]

# configure decision tree: scikitlearn tree will determine max depth range best itself. values of 15 only set after
# trial and error testing close, within, and beyond these value ranges following advice in guidelines for use in
# scikit learn decision tree documentation.
# Web Reference:
# https://scikit-learn.org/stable/modules/tree.html -> section 1.10.5. Tips on practical use
dt = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, class_weight=None, presort=False)

# train the decision tree and create decision tree learning model based on x and y values defined previously
dt = dt.fit(x_features, y_label)

# labels for data output results
pred_label = "predicted class:"
true_label = "actual class:"

# list of predictions versus ground truth values: required for confusion matrix and testing on test.csv
pred_value_list = []
true_value_list = []

# run on test.csv
test = pd.read_csv("../data/test.csv", skiprows=0)

# handle conversion on the classification label column data type so it is binary
test['class'] = test['class'].map({'pos': 1, 'neg': 0})

# preprocess the test.csv file
test_set = test.iloc[:,1:feature_col_count]

# get test features: use slice value range to determine how many records we want
test_df = test_set.loc[0:5427]
# get test ground truth values
true_value_list = test['class']

# get predictions
pred_value_list = dt.predict(test_df)

# generate decision tree prediction probabilities
y_predict_probabilities = dt.predict_proba(test_df)
# print(y_predict_probabilities)

# calculate accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
accuracy_score_label = "Accuracy Score Measure is: "
accuracy = accuracy_score(true_value_list, pred_value_list)
print(accuracy_score_label, accuracy)

# calculate f-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
f_score_label = "F-Score Measure is: "
f_score = fbeta_score(true_value_list, pred_value_list, average='macro', beta=0.5)
print(f_score_label, f_score)

# print confusion matrix
confusion_matrix_label = "Confusion Matrix: "
print(confusion_matrix_label)
cm = confusion_matrix(true_value_list, pred_value_list)
print(confusion_matrix(true_value_list, pred_value_list))
print(classification_report(true_value_list, pred_value_list))

# get confusion matrix values explicitly and use for reporting table in conjunction with cm visualization
true_negative = cm[0][0]
print("True Negative", true_negative)
false_negative = cm[1][0]
print("False Negative", false_negative)
true_positive = cm[1][1]
print("True Positive", true_positive)
false_positive = cm[0][1]
print("False Positive", false_positive)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# get auroc value and will need it for plotting the curve graph
roc_auc_score = roc_auc_score(true_value_list, pred_value_list)
print("AUROC:", roc_auc_score)

# plot roc curve graph per guidelines in web reference samples
# Web References:
# http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# fpr, tpr = scikit learn standard format for false postive rate, true positive rate
fpr, tpr, _ = roc_curve(true_value_list, pred_value_list)
roc_auc = auc(fpr, tpr)

# create the roc curve plot: use benalexkeen.com blog as main reference, customize and using scikit learn samples
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# create output file
prob_formatted_results = []
test_result_file = []
test_result_file.append("*** Decision Tree Classifier Measurements ***\n")
test_result_file.append("*** Accuracy: ***")
test_result_file.append(accuracy)
test_result_file.append("\n")
test_result_file.append("*** F-Score: ***")
test_result_file.append(f_score)
test_result_file.append("\n")
test_result_file.append("*** Probability of Class Predictions: ***")
# improve readability in output file, so for every probability record, just append record so reviewer can read easily
for p in y_predict_probabilities:
    test_result_file.append(p)

# write train_gold.txt metrics to file
with open('../results/train_gold.txt', 'w') as f:
    for record in test_result_file:
        f.write(str(record) + '\n')

# visualize the decision tree
# dot_data = tree.export_graphviz(dt, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("part_1_decision_tree")

