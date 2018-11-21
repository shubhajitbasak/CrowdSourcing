# Authors: Niall Guerin
# Part 1: Script Function
# 1. Read the gold.csv expert rated movie sentence id file.
# 2. Convert the pos/neg string values to binary numeric 0 and 1 values.
# 3. Take a random sample of 1000 records.
# 4. Write the resulting sample to file system.
import pandas as pd

# load full gold.csv dataset from file into pandas dataframe

gold_dataset = pd.read_csv("../data/gold.csv", skiprows=0)

# convert the pos and neg classification labels: can be represented as binary 0 or 1 values
gold_dataset['class'] = gold_dataset['class'].map({'pos': 1, 'neg': 0})

# get random sample using pandas dataframe built-in sample function: n = 1000
# set random_state = 1 for final submission to ensure values are consistent in report write-up, sets seed
# Web Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
sample_size = 1000
training_sample = gold_dataset.sample(sample_size, random_state=1)

# write the sample to file, set index = False to avoid leading comma on row 1 based on unit tests
training_sample.to_csv("../data/gold_sample.csv", encoding='utf-8', index=False)

# end main script: the functions below this point are helper functions for dataset analysis and reporting only

# # check for label bias in the random sample i.e. we want balance of pos and neg labels in 1000 row sample
# def check_random_sample_is_balanced(input):
# 	pos_sample = input.loc[input['class'] == 1]
# 	neg_sample = input.loc[input['class'] == 0]
# 	# counts will not be identical but also should not be skewed to one class whether pos or neg
# 	print(len(pos_sample))
# 	print(len(neg_sample))
# 	return
#
# # get the max value from the feature labels only (used by decision tree) from dataframe cols/rows
# def check_feature_max_value(input):
# 	value_range_check = input.iloc[:, 1:1201]
# 	print(value_range_check)
# 	maxFeatureValue = value_range_check.values.max()
# 	print(maxFeatureValue)
# 	return
#
# # get the min value from the feature labels only (used by decision tree) from dataframe cols/rows
# def check_feature_min_value(input):
# 	value_range_check = input.iloc[:, 1:1201]
# 	print(value_range_check)
# 	maxFeatureValue = value_range_check.values.min()
# 	print(maxFeatureValue)
# 	return
#
# # helper statement for reporting only: comment out in final build
# # check sample balance (pos/neg)
# check_random_sample_is_balanced(training_sample)
# # check max and min LSA topic feature value ranges
# check_feature_max_value(training_sample)
# check_feature_min_value(training_sample)
