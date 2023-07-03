import json
import re
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def extract_TranactionType_columns(df):
    # Create empty lists for each extracted column
    estate_type_german = []
    distribution_type_german = []
    estate_type = []
    distribution_type = []

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Extract the JSON structure as a string from the "TranactionType" column
        json_string = row['TranactionType']
        
        # Convert the JSON string to a Python dictionary
        transaction_dict = json.loads(json_string)
        
        # Extract the values from the dictionary
        estate_type_german.append(transaction_dict['EstateTypeGerman'])
        distribution_type_german.append(transaction_dict['DistributionTypeGerman'])
        estate_type.append(transaction_dict['EstateType'])
        distribution_type.append(transaction_dict['DistributionType'])

    # Add the extracted columns to the DataFrame
    df['EstateTypeGerman'] = estate_type_german
    df['DistributionTypeGerman'] = distribution_type_german
    df['EstateType'] = estate_type
    df['DistributionType'] = distribution_type

    # Drop the original column
    df.drop('TranactionType', axis=1, inplace=True)

    return df


def binarize_columns(df, column_name):

    # replace whitespaces with underscores within entries and put everything in lowercase
    df[column_name] = df[column_name].apply(lambda x: [re.sub(r'\s', '_', feature.lower()) for feature in re.findall(r'"([^"]*)"', x)])

    # use MultiLabelBinarizer to binarize the column
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df[column_name])
    df_binary = pd.DataFrame(binary_matrix, columns=mlb.classes_)

    # Drop the original column and concatenate the binary columns to the existing DataFrame
    df.drop(column_name, axis=1, inplace=True)
    df = pd.concat([df, df_binary], axis=1)

    return df
