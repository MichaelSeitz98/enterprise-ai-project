import json

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