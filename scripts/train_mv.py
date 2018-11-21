# Authors: Niall Guerin, Shubhajit Basak
# Part 2: Script Function
# 1. Read the training sample (gold_sample.csv)
# 2. Filter out the movie sentence ids so we join on those ids with mturk.csv: creates mturk_sample.csv.
# 3. Implement a majority vote hard vote function that counts majority of labels in input: pos/neg from mturk_sample.csv.
# 4. Configure decision tree and train on training sample i.e. the aggregated data output (1K rows)
# from mturk_sample.csv input (4k-6k rows).
# 5. Parse in test.csv and drop id and convert pos/neg values to binary 0/1 for ground truth labelling
# 6. Test decision tree against test.csv
# 7. Construct confusion matrix, print accuracy, f-score, prediction probabilities, plot ROC curve.
# 8. Write accuracy, f-score, prediction probabilities to text file.
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
# import graphviz

# Code Start - Shubhajit Basak ---- # 

# Read sample file
df_mturk = pd.read_csv("../data/mturk_sample.csv")
# df_mturk.head()

# take a backup to work on
df_mturk_updt = df_mturk
# drop unrequired columns
df_mturk_updt = df_mturk_updt.drop(df_mturk_updt.loc[:,'TOPIC0':'TOPIC1199'].head(0).columns, axis=1)
# df_mturk_updt.head()

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

# get a backup of original dataset
df = df_mturk_updt


def get_majority(df):
    # calculate the majority vote

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
        # Update Matrix with thev majority vote count
        df_pol = df_pol.append({'id': inputId, 'pos': pos,'neg': neg}, ignore_index=True)

    return (df_pol)

# Get Training Labels
train_label = get_majority(df)
# Take the Positive Labels
train_label = train_label[["pos"]]
# Group By and Reset Index
train_data = df_mturk.groupby(['id']).first().reset_index()

# Take from the 2nd to row before the last row to get the features
train_data = train_data.iloc[:, :-1]
train_data = train_data.iloc[:, 2:]

# Create Y Label and X Features for Classifier Input
y_label = train_label.iloc[:,0]
x_features = train_data

feature_col_count = 1201

# Code End - Shubhajit Basak ---- # 

# Code Start - Niall Guerin ---- # 

# configure decision tree: scikitlearn tree will determine max depth range best itself. values of 15 only set after
# trial and error testing close, within, and beyond these value ranges following advice in guidelines for use in
# scikit learn decision tree documentation.
#  Web Reference:
# https://scikit-learn.org/stable/modules/tree.html -> section 1.10.5. Tips on practical use
dt = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, class_weight=None, presort=False)

# train the decision tree and create decision tree learning model: use this dt model on test.csv in final step
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
test_set = test.iloc[:, 1:feature_col_count]

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
with open('../results/train_mv.txt', 'w') as f:
    for record in test_result_file:
        f.write(str(record) + '\n')

# visualize the decision tree
# dot_data = tree.export_graphviz(dt, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("part_2_decision_tree")

# # helper function for reporting: get worker to movie sentence id average count
# 
#
# # get worker to movie sentence id all reviewers counts to return total_counts to caller
# def get_worker_movie_metrics(data_record):
# 	# analyze the records and get majority classifier and return classification majority vote result subset from input list
# 	positive_reviews = data_record.loc[data_record['class'] == 1]
# 	negative_reviews = data_record.loc[data_record['class'] == 0]
#
# 	# get the counts of positive versus negative reviews for the movie id as all represent different workers
# 	count_pos = len(positive_reviews)
# 	count_neg = len(negative_reviews)
#
# 	# obtain the totals for this movie sentence id
# 	total_count = count_pos+count_neg
#
# 	return total_count
#
#
# # get worker to movie average and print to console
# def get_worker_to_movie_average(agg_worker_movie_list):
# 	worker_movie_average_count = []
# 	for idx, item in enumerate(agg_worker_movie_list):
# 		measure_worker_movie = get_worker_movie_metrics(item)
# 		worker_movie_average_count.append(measure_worker_movie)
# 		average_workers = np.mean(worker_movie_average_count)
# 	return average_workers
#
#
# # print worker to movie average for reporting
# print("Worker to Movie Sentence ID Average Count:")
# print(get_worker_to_movie_average(movie_worker_list_agg))


# Code End - Niall Guerin ---- # 