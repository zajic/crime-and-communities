import pandas as pd
import os
import numpy as np
import pickle

#increase the column width to print long attribute descriptions
pd.set_option('max_colwidth', 100)
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))

'''
#######################################
# Load dataset                        #
#######################################
'''
#parse weka header
header_file = open(os.path.join(DATA_DIR, "unnormalized_header.txt"), "r")
header = []
for line in header_file:
    header.append(line.split(" ")[1])

data = pd.read_csv(os.path.join(DATA_DIR, "crime_data_unnormalized.txt"), sep = ",", header = None, names = header)

data.info()

#load attribute descriptions, keys in the dictionary are dataframe header entries
header_description = open(os.path.join(DATA_DIR, "header_description.txt"), "r")
attribute_descr = {}
index = 0
for line in header_description:
    line = line.split(": ")
    attribute_descr[header[index]] = line[1].strip()
    index+=1


'''
##########################################
# View missing values for each attribute #
##########################################
'''
#replace ? with NaN to mark missing values
data = data.replace("?",np.NaN)

def find_missing_values(data):
#calculate count and percentage of missing values per column
    missing_values = data.isnull().sum()
    missing_values = pd.DataFrame(missing_values.loc[missing_values != 0], columns = ["count"])
    missing_values['percentage'] = round(missing_values['count']/data.shape[0],2)
    missing_values = missing_values.sort_values(by = 'count', ascending = False)
    return missing_values

missing_values = find_missing_values(data)
print("\n\n")
print("Missing values")
print(missing_values)

#get description of each attribute with missing values
attr_with_missing_values = missing_values.index.tolist()
#append the description to the dataframe
missing_values['description'] = pd.Series([attribute_descr[key] for key in attr_with_missing_values]).values

print("\n\n")
print('Dropping columns with 50% and more missing values:\n')
columns_with_too_many_missing = missing_values[missing_values["percentage"] > 0.5]
print(missing_values[missing_values["percentage"] > 0.5].loc[:,"description"])
data = data.drop(columns_with_too_many_missing.index.values, axis = 1)

#TODO: drop fold and other columns

print("\n\n")
print("Remaining missing values")
missing_values = find_missing_values(data)
print(missing_values)

# delete one line with a missing value
data = data[data['otherPerCap'].notnull()]

##########################################
# Split data to X and y                  #
##########################################

#independent variables are the last 18
y_labels = header[-18:]
y = data[y_labels]

print("\n\n")
print("Dependent variables:\n")
print(y.columns.values)

X = data.drop(y_labels, axis = 1)

print("\n\n")
print("Independent variables:\n")

for name in X.columns.values:
    print(name + " : " + attribute_descr[name])