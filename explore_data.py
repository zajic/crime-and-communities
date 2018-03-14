import pandas as pd
import os
import numpy as np
import pickle

#increase the column width to print long attribute descriptions
pd.set_option('max_colwidth', 100)
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))

'''
Load dataset
'''
#parse weka header
header_file = open(os.path.join(DATA_DIR, "unnormalized_header.txt"), "r")
header = []
for line in header_file:
    header.append(line.split(" ")[1])

data = pd.read_csv(os.path.join(DATA_DIR, "crime_data_unnormalized.txt"), sep = ",", header = None, names = header)

data.info()

#split to X and y
y_labels = header[-18:]
y = data[y_labels]

print("Dependent variables:")
print(y.columns.values)

X = data.drop(y_labels, axis = 1)

#load attribute descriptions, keys in the dictionary are dataframe header entries
header_description = open(os.path.join(DATA_DIR, "header_description.txt"), "r")
attribute_descr = {}
index = 0
for line in header_description:
    line = line.split(": ")
    attribute_descr[header[index]] = line[1].strip()
    index+=1

print("Independent variables:")
for name in X.columns.values:
    print(name + " : " + attribute_descr[name])

'''
View missing values for each attribute
'''
#replace ? with NaN to mark missing values
data = data.replace("?",np.NaN)

#calculate count and percentage of missing values per column
missing_values = data.isnull().sum()
missing_values = pd.DataFrame(missing_values.loc[missing_values != 0], columns = ["count"])
missing_values['percentage'] = round(missing_values['count']/X.shape[0],2)
missing_values = missing_values.sort_values(by = 'count', ascending = False)

print("Missing values")
print(missing_values)

#get description of each attribute with missing values
attr_with_missing_values = missing_values.index.tolist()
missing_values['description'] = pd.Series([attribute_descr[key] for key in attr_with_missing_values]).values

print("Attributes with 50% and more missing values:")
print(missing_values[missing_values["percentage"] > 0.5,"description"])

print('Dropping columns with no predictive value and/or too many missing values.')
