import pickle
import pandas as pd
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


def load_model():
    with open(f"{BASE_DIR}/ml_model-{__version__}.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def preprocess_data(data):
    expected_columns = [
        "LivingSpace", "Rooms", "altbau_(bis_1945)", "balkon", "barriefrei", "dachgeschoss", "einbaukueche", "neubau", "parkett", "stellplatz", 
        "bad/wc_getrennt", "personenaufzug", "garten", "garage", "renoviert", "terrasse", "wanne", "zentralheizung", "abstellraum", "ferne", 
        "fussbodenheizung", "gartennutzung", "kelleranteil", "ZipCode_97070", "ZipCode_97072", "ZipCode_97074", "ZipCode_97076", 
        "ZipCode_97078", "ZipCode_97080", "ZipCode_97082", "ZipCode_97084", "ZipCode_97204", "ZipCode_97209", "ZipCode_97218", 
        "ZipCode_97222", "ZipCode_97228", "ZipCode_97234", "ZipCode_97236", "ZipCode_97246", "ZipCode_97249", "ZipCode_97250", 
        "ZipCode_97261", "ZipCode_97270", "ZipCode_97288", "ZipCode_97297", "ZipCode_97299"
    ]

    encoded_data = {
    'LivingSpace': float(data['LivingSpace']),
    'Rooms': float(data['Rooms']),
    'altbau_(bis_1945)': int(data['altbau_(bis_1945)']),
    'balkon': int(data['balkon']),
    'barriefrei': int(data['barriefrei']),
    'dachgeschoss': int(data['dachgeschoss']),
    'einbaukueche': int(data['einbaukueche']),
    'neubau': int(data['neubau']),
    'parkett': int(data['parkett']),
    'stellplatz': int(data['stellplatz']),
    'bad/wc_getrennt': int(data['bad/wc_getrennt']),
    'personenaufzug': int(data['personenaufzug']),
    'garten': int(data['garten']),
    'garage': int(data['garage']),
    'renoviert': int(data['renoviert']),
    'terrasse': int(data['terrasse']),
    'wanne': int(data['wanne']),
    'zentralheizung': int(data['zentralheizung']),
    'abstellraum': int(data['abstellraum']),
    'ferne': int(data['ferne']),
    'fussbodenheizung': int(data['fussbodenheizung']),
    'gartennutzung': int(data['gartennutzung']),
    'kelleranteil': int(data['kelleranteil'])
}

    zip_code_key = 'ZipCode_' + data['ZipCode']
    for column in expected_columns:
        if column == zip_code_key:
            encoded_data[column] = 1
        else:
            encoded_data[column] = 0

    return encoded_data


def predict(data):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    data = preprocess_data(data)
    df = pd.DataFrame(data, index=[0])
    df['LivingSpace'] = df['LivingSpace'].astype(float)
    df['Rooms'] = df['Rooms'].astype(float)
    columns_to_convert = ["ZipCode_97070", "ZipCode_97072", "ZipCode_97074", "ZipCode_97076",
                      "ZipCode_97078", "ZipCode_97080", "ZipCode_97082", "ZipCode_97084",
                      "ZipCode_97204", "ZipCode_97209", "ZipCode_97218", "ZipCode_97222",
                      "ZipCode_97228", "ZipCode_97234", "ZipCode_97236", "ZipCode_97246",
                      "ZipCode_97249", "ZipCode_97250", "ZipCode_97261", "ZipCode_97270",
                      "ZipCode_97288", "ZipCode_97297", "ZipCode_97299"]

    # Cast specified columns to int32
    df[columns_to_convert] = df[columns_to_convert].astype('int32')
    result = model.predict(df)[0]
    return round(result, 2)
