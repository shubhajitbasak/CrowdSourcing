# Authors: Shubhajit Basak
# Part 3: Script Function
# 1. Read the training sample (mturk_sample.csv)
# 2. Filter out the movie sentence ids so we join on those ids with mturk.csv: creates mturk_sample.csv.
# 3. Implement a majority vote hard vote function that counts majority of labels in input: pos/neg from mturk_
# sample.csv.
# 4. Implement dawid-skene method based on 1979 research paper and case study slide deck pp.30-45.
# 5. Configure decision tree and train on training sample i.e. the aggregated data output (1K rows)
# from mturk_sample.csv input (4k-6k rows) using dawid-skene method to generate the training set based on worker
# quality.
# 6. Parse in test.csv and drop id and convert pos/neg values to binary 0/1 for ground truth labelling
# 7. Test decision tree against test.csv
# 8. Construct confusion matrix, print accuracy, f-score, prediction probabilities, plot ROC curve.
# 9. Write accuracy, f-score, prediction probabilities to text file.
from sklearn.preprocessing import normalize
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

def proc_init(df):
    # Initialise to create polarity matrix with majority vote
    # Create a empty dictionary to store confusion matrix

    # polarity columns based on table in slide deck pp.30-45

    # Create Data Frame to store Polarity Matrix
    pol_cols = ['id', 'pos', 'neg']
    df_pol = pd.DataFrame(columns=pol_cols)
    df_pol = df_pol.fillna(0) # initialise with zeros
    # Loop Through the data and update the data frame with the majority vote
    for i in range(0, len(df)):
        inputId = df.iloc[i,:]["id"]
        pos = 0
        neg = 0
        lst = list(df.iloc[i,1:])

        # Take the max to get the majority vote
        major = max(lst, key=lst.count)
        if(major == 1):
            pos = 1
        else:
            neg = 1

        # Update Polarity Matrix with thev majority vote count
        df_pol = df_pol.append({'id': inputId, 'pos': pos,'neg': neg}, ignore_index=True)

    # Create and initialise the dictionary for storing confision matrix
    d_w={}
    for w in df.columns[1:]:
        d_w[w] = np.zeros((2, 2))

    return (df_pol, d_w)


# normalize the python collection using imports from scikit learn normalization functions
# Web References:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
def normalise_dict(d_w):
    for key, value in d_w.items():
        d_w[key] = normalize(d_w[key], axis=1, norm='l1')
    return(d_w)


# function to update worker polarity values
def update_polarity_table(df, d_w,df_pol):
    for i in range(0,len(df)):
        item = df.iloc[i,:]

        pos = 0
        neg = 0
        # loop through all the workers 
        for w in df.columns[1:]:

            if(item[w] == 1):
                pos = pos + d_w[w][1, 1]
                neg = neg + d_w[w][0, 1]


            elif(item[w] == 0):
                pos = pos + d_w[w][1, 0]
                neg = neg + d_w[w][0, 0]

        # Code Review Insertion: removed .set_value function call
        # as this is deprecated in pandas API and was generating
        # warnings in console during testing.
        # Web Reference: https://stackoverflow.com/questions/28757389/pandas-loc-vs-iloc-vs-ix-vs-at-vs-iat
        # Update polarity values
        df_pol.at[df.index[i], 'pos'] = pos
        df_pol.at[df.index[i], 'neg'] = neg

    # Normalise the data in the polarity table
    df_pol.iloc[:, 1:] = df_pol.iloc[:, 1:].div(df_pol.values[:, 1:].sum(axis=1), axis=0)
    return df_pol


# update collection of confusion matrix
def update_confusion_matrix(df, df_pol, d_w):
    # Read for each worker from sample
    for w in df.columns[1:]:
        wData = df[['id', w]] # get the worker data
        # Update Confusion matrix for the same worker
        for i in range(0, len(wData)):
            # Parse all the items for that worker for its vote
            if(wData.iloc[i,1] == 1):
                if(df_pol.iloc[i,:]["pos"] > df_pol.iloc[i,:]["neg"]):
                    d_w[w][1,1] = 1
                else:
                    d_w[w][0,1] = df_pol.iloc[i,:]["neg"]
            elif(wData.iloc[i,1] == 0):
                if(df_pol.iloc[i,:]["pos"] > df_pol.iloc[i,:]["neg"]):
                    d_w[w][1,0]=df_pol.iloc[i,:]["pos"]
                else:
                    d_w[w][0,0]=1
    # Normalise Confusion Matrix dictionary
    d_w = normalise_dict(d_w)
    return d_w

# get classification whether positive or negative
# will be used in lambda expression
def get_class(pos, neg):
    if(pos > neg):
        return 'pos'
    else:
        return 'neg'


df_mturk = pd.read_csv("../data/mturk_sample.csv")
df_mturk.head()

df_mturk_updt = df_mturk
# drop unrequired columns
df_mturk_updt = df_mturk_updt.drop(df_mturk_updt.loc[:,'TOPIC0':'TOPIC1199'].head(0).columns, axis=1)
df_mturk_updt.head()

# Reshape the data frame so that it is organised by worker/annotator
# Web References:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
df_mturk_updt = df_mturk_updt.pivot(index='id', columns='annotator', values='class')

# Reset index after reshaping to make it easier to work with dataframe
# Web References:
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
df_mturk_updt = df_mturk_updt.reset_index()

# Set binary classification label to numeric 0 or 1
df_mturk_updt = df_mturk_updt.replace(to_replace=['neg', 'pos'], value=[0, 1])
#df_mturk_updt.head()

# run the david skene method on the actual dataset 
df = df_mturk_updt
df_pol,d_w = proc_init(df)
for i in range(0,5):
    d_w = update_confusion_matrix(df, df_pol, d_w)
    df_pol = update_polarity_table(df, d_w,df_pol)

# Update the class value as per the majority value
df_pol['Label'] = df_pol.apply(lambda x: get_class(x['pos'], x['neg']), axis=1)

# Reset the index of the dataframe
train_data = df_mturk.groupby(['id']).first().reset_index()

# Take from the 2nd to row before the last row to get the features
train_data = train_data.iloc[:, :-1]
train_data = train_data.iloc[:, 2:]
# train_data.head()
# Extrach Labels from polarity table
train_label = df_pol.iloc[:,-1:]

# parse in test set
test_df = pd.read_csv("../data/test.csv")
test_df_feature = test_df.iloc[:, :-1]
test_df_feature = test_df_feature.iloc[:, 1:]
test_df_label = test_df.iloc[:,-1]

# configure decision tree: parameters must be consistent with part 1 and part 2 parameter values for comparison
# test evaluation of the decision tree model
classifier = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, class_weight=None, presort=False)

# train the decision tree on the training set and classification labels
classifier.fit(train_data, train_label)

# get predictions
y_pred = classifier.predict(test_df_feature)

# cm = confusion_matrix(test_df_label, y_pred)  
classreport = classification_report(test_df_label, y_pred)
accuracy = accuracy_score(test_df_label, y_pred)
f_score = fbeta_score(test_df_label, y_pred, average='macro', beta=0.5)

# calculate accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
accuracy_score_label = "Accuracy Score Measure is: "
accuracy = accuracy_score(test_df_label, y_pred)
print(accuracy_score_label, accuracy)

# calculate f-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
f_score_label = "F-Score Measure is: "
f_score = fbeta_score(test_df_label, y_pred, average='macro', beta=0.5)
print(f_score_label, f_score)

# get confusion matrix values explicitly and use for reporting table in conjunction with cm visualization
confusion_matrix_label = "Confusion Matrix: "
print(confusion_matrix_label)
cm = confusion_matrix(test_df_label, y_pred)
print(confusion_matrix(test_df_label, y_pred))
print(classification_report(test_df_label, y_pred))

# get confusion matrix values directly
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

# get prediction probabilities:
# below is utility code only to fix errors that were occuring due to mix of string/numerics when trying to plot
# ROC curve using methods from part 1 and part 2 scripts. Standard functions were complaining about the mix of
# of string/numerics when calling scikit learn roc_curve() function with unchanged values above. Values are same
# so just converting pos/neg to 1,0 respectively for both variables to allow same plots for reporting.
test_df_label = test_df_label.replace({'pos': 1, 'neg': 0})
y_pred = np.where(y_pred == 'pos', 1, 0)

# generate decision tree prediction probabilities: roll this back to part 1 and part 2 scripts as one-liner
y_predict_probabilities = classifier.predict_proba(test_df_feature)

# get auroc score and will need it for plotting the curve graph
roc_auc_score = roc_auc_score(test_df_label, y_pred)
print("AUROC:", roc_auc_score)

# plot roc curve graph per guidelines in web reference samples
# Web References:
# http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
fpr, tpr, _ = roc_curve(test_df_label, y_pred)
roc_auc = auc(fpr, tpr)

# plot roc curve graph per guidelines in web reference samples
# Web References:
# http://benalexkeen.com/scoring-classifier-models-using-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
fpr, tpr, _ = roc_curve(test_df_label, y_pred)
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
test_result_file.append("*** Decision Tree Classifier Measurements ***\n\n")
test_result_file.append("*** Accuracy: ***")
test_result_file.append(accuracy)
test_result_file.append("\n\n")
test_result_file.append("*** F-Score: ***")
test_result_file.append(f_score)
test_result_file.append("\n\n")
test_result_file.append("*** Probability of Class Predictions: ***")
# improve readability in output file, so for every probability record, just append record so reviewer can read easily
for p in y_predict_probabilities:
    test_result_file.append(p)

# write train_mv.txt metrics to file
with open('../results/train_ds.txt', 'w') as f:
	for record in test_result_file:
		f.write(str(record) + '\n')

# visualize the decision tree
# dot_data = tree.export_graphviz(classifier, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("part_3_decision_tree")
