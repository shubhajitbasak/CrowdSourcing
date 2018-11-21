# Authors: Niall Guerin
# Part 2: Script Function
# 1. Using gold_sample.csv from step 1, correlate the ids with the mturk.csv
# 2. Create merged/joined dataset that aggregates the workers and correlated movie sentence ids.
# 3. Write the aggregated mturk_sample.csv to file system. This will be the input to majority vote function.

# import pandas for data set processing and filtering
import pandas as pd

# read the gold_sample.csv to obtain the movie sentence id column values to be used to correlate with mturk.csv id values
gold_sample_id_list = pd.read_csv("../data/gold_sample.csv", skiprows=0)

# filter out the movie sentence ids only
filter_id_list = gold_sample_id_list.iloc[:, 0]

# sort the filtered movie sentence id values so we facilitate faster index search and debugging if required
sorted_filter_id_list = filter_id_list.sort_values()

# load the mturk.csv file: this contains the worker data (annotator column)
mturk_sample = pd.read_csv("../data/mturk.csv", skiprows=0)

# result count value of this read should be between 4K and 6K records: 4K < result < 6K records.
# join on movie sentence id from both lists (mturk.csv, gold_sample.csv) - see pp.58 case studies slide deck.
joined_mturk_sample = mturk_sample.loc[mturk_sample['id'].isin(sorted_filter_id_list)]

# write the mturk_sample to file: this will be input for the next step for majority vote function
# set index=False to avoid leading comma observed in unit tests
joined_mturk_sample.to_csv("../data/mturk_sample.csv", encoding='utf-8', index=False)

# end main script: the functions below this point are helper functions for dataset analysis and reporting only

# # get the max value from the feature labels only (used by decision tree) from dataframe cols/rows
# def check_feature_max_value(input):
# 	value_range_check = input.iloc[:, 2:1202]
# 	print(value_range_check)
# 	maxFeatureValue = value_range_check.values.max()
# 	print(maxFeatureValue)
# 	return
#
# # get the min value from the feature labels only (used by decision tree) from dataframe cols/rows
# def check_feature_min_value(input):
# 	value_range_check = input.iloc[:, 2:1202]
# 	print(value_range_check)
# 	maxFeatureValue = value_range_check.values.min()
# 	print(maxFeatureValue)
# 	return
#
# # helper statement for reporting only: comment out in final build
# # check max and min LSA topic feature value ranges
# check_feature_max_value(joined_mturk_sample)
# check_feature_min_value(joined_mturk_sample)
