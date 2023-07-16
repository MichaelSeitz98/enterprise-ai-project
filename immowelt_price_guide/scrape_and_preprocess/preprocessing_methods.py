import json
import re
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import ast
import numpy as np

def preprocess_data(df):

    df = extract_zipcode(df)
    df = extract_EstateDistribution_columns(df)
    df = compute_rooms_if_missing(df)
    df = binarize_columns(df, 'Object_features')
    df = drop_columns(df)
    return df

def compute_rooms_if_missing(df):

    df["LivingSpace"] = pd.to_numeric(df["LivingSpace"], errors="coerce")
    df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")

    df.dropna(subset=["LivingSpace"], inplace=True)

    df["Rooms"].fillna((df["LivingSpace"] / 15).round().astype(int), inplace=True)
    
    df.reset_index(drop=True, inplace=True)

    return df

def drop_columns(df):
    columns_to_drop = ['Url', 'Object_currency', 'Title', 'Price', 'MediaItems', 'BasicInfo', 'ContactData']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(existing_columns, axis=1, inplace=True)
    return df


def extract_EstateDistribution_columns(df):
    # Create empty lists for each extracted column
    estate_type = []
    distribution_type = []

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Extract the JSON structure as a string from the "TranactionType" column
        json_string = row['TranactionType']
        
        # Convert the JSON string to a Python dictionary
        transaction_dict = json.loads(json_string)
        
        # Extract the values from the dictionary, if present
        estate_type_val = transaction_dict.get('EstateType')
        distribution_type_val = transaction_dict.get('DistributionType')

        # Append the values to the respective lists, or add NaN if not found
        estate_type.append(estate_type_val if estate_type_val is not None else np.nan)
        distribution_type.append(distribution_type_val if distribution_type_val is not None else np.nan)

    # Add the extracted columns to the DataFrame
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

def check_concat_xlsx(file1, file2, output_file):
    # Read the Excel files
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Check if column structures match
    if df1.columns.tolist() != df2.columns.tolist():
        print("Column structures do not match.")
        return

    # Concatenate the data frames
    concatenated = pd.concat([df1, df2])

    # Write the concatenated data frame to a new Excel file
    concatenated.to_excel(output_file, index=False)
    print(f"Concatenated data saved to {output_file}")


def extract_zipcode(df):
    # Convert string values in 'Address' column to dictionaries
    df['Address'] = df['Address'].apply(lambda x: {} if pd.isnull(x) else json.loads(x))
    
    # Extract 'ZipCode' from 'Address' column and convert it to string
    df['ZipCode'] = df['Address'].apply(lambda x: str(x.get('ZipCode', None)))
    
    # Drop the old 'Address' column
    df.drop('Address', axis=1, inplace=True)
    
    return df

def rename_columns_with_umlauts(df):
    # Replace 'ü' with 'ue' in column names
    new_columns = df.columns.str.replace('ü', 'ue')

    # Replace 'ö' with 'oe' in the modified column names
    new_columns = new_columns.str.replace('ö', 'oe')

    # Replace 'ä' with 'ae' in the modified column names
    new_columns = new_columns.str.replace('ä', 'ae')

    # Rename the columns in the DataFrame
    df.columns = new_columns
    return df

