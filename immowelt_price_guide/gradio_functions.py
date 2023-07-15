import gradio as gr
import pandas as pd
import mlflow
import numpy as np
import mlflow.pyfunc
import xgboost as xgb
import mlflow.xgboost
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from 


# nimmt die daten der abgespeicherten Prediction und wandelt es als datafrme um
def get_predictions(file):
    df = pd.read_excel(file)
    return df


# macht aus den Daten ein bar Plot welche models am besten sind
def bar_chart(file="results-selected-features-aug.xlsx"):
    df = get_predictions(file)
    plot = px.bar(
        df,
        x="tags.mlflow.runName",
        y="metrics.mae",
        title="Modellperformance",
        color="tags.mlflow.runName",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    return gr.update(value=plot, visible=True)


# lädt das beste model, das als pickle file abgespeichert ist
def get_model():
    with open("model.pkl", "rb") as file:
        model_pickle = pickle.load(file)
    return model_pickle


# lädt das model aus mlflow
def load_model(model_name, stage="production"):
    model_version = 1
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model


# speichert das model als pickle file ab
# def save_model(model_name):
#     model_pickle = load_model(model_name)
#      with open('model.pkl', 'wb') as file:
#          model_pickle = pickle.dump(file)
#      return model_pickle
# hier wird aus den gradio eingaben ein datframe erstellt und das Modell wird geladen und macht predictions anhand der eingaben
def trigger_actions(
    feature_squrmeter,
    feature_zip,
    feature_rooms,
    features_altbau,
    feature_balkon,
    feature_ba,
    feature_dachgeschoss,
    feature_einbaukueche,
    feature_neubau,
    feature_parkett,
    feature_stellplatz,
    feature_badwc_getrennt,
    feature_personenaufzug,
    feature_garten,
    feature_garage,
    feature_renoviert,
    feature_terrasse,
    feature_wanne,
    feature_zentralheizung,
    feature_abstellraum,
    feature_fernwaerme,
    feature_fussbodenheitzung,
    feature_gartenmitbenutzung,
    feature_kellerabteil,
    erklärung,
):
    model = get_model()
    data_list = (
        [feature_squrmeter]
        + [feature_rooms]
        + [features_altbau]
        + [feature_balkon]
        + [feature_ba]
        + [feature_dachgeschoss]
        + [feature_einbaukueche]
        + [feature_neubau]
        + [feature_parkett]
        + [feature_stellplatz]
        + [feature_badwc_getrennt]
        + [feature_personenaufzug]
        + [feature_garten]
        + [feature_garage]
        + [feature_renoviert]
        + [feature_terrasse]
        + [feature_wanne]
        + [feature_zentralheizung]
        + [feature_abstellraum]
        + [feature_fernwaerme]
        + [feature_fussbodenheitzung]
        + [feature_gartenmitbenutzung]
        + [feature_kellerabteil]
        + [int(0)] * 23
    )
    data = pd.DataFrame()
    data = pd.DataFrame(
        columns=[
            "LivingSpace",
            "Rooms",
            "altbau_(bis_1945)",
            "balkon",
            "barriefrei",
            "dachgeschoss",
            "einbaukueche",
            "neubau",
            "parkett",
            "stellplatz",
            "bad/wc_getrennt",
            "personenaufzug",
            "garten",
            "garage",
            "renoviert",
            "terrasse",
            "wanne",
            "zentralheizung",
            "abstellraum",
            "ferne",
            "fussbodenheizung",
            "gartennutzung",
            "kelleranteil",
            "ZipCode_97070",
            "ZipCode_97072",
            "ZipCode_97074",
            "ZipCode_97076",
            "ZipCode_97078",
            "ZipCode_97080",
            "ZipCode_97082",
            "ZipCode_97084",
            "ZipCode_97204",
            "ZipCode_97209",
            "ZipCode_97218",
            "ZipCode_97222",
            "ZipCode_97228",
            "ZipCode_97234",
            "ZipCode_97236",
            "ZipCode_97246",
            "ZipCode_97249",
            "ZipCode_97250",
            "ZipCode_97261",
            "ZipCode_97270",
            "ZipCode_97288",
            "ZipCode_97297",
            "ZipCode_97299",
        ]
    )
    data.loc[len(data)] = data_list
    for index, row in data.iterrows():
        for columns in data.columns:
            if feature_zip[:5] in columns:
                data.loc[index, columns] = int(1)
    data = data.replace(False, 0)
    data = data.replace(True, 1)
    data = data.astype({"ZipCode_97070": np.int32})
    data = data.astype({"ZipCode_97072": np.int32})
    data = data.astype({"ZipCode_97074": np.int32})
    data = data.astype({"ZipCode_97076": np.int32})
    data = data.astype({"ZipCode_97078": np.int32})
    data = data.astype({"ZipCode_97080": np.int32})
    data = data.astype({"ZipCode_97082": np.int32})
    data = data.astype({"ZipCode_97084": np.int32})
    data = data.astype({"ZipCode_97204": np.int32})
    data = data.astype({"ZipCode_97209": np.int32})
    data = data.astype({"ZipCode_97218": np.int32})
    data = data.astype({"ZipCode_97222": np.int32})
    data = data.astype({"ZipCode_97228": np.int32})
    data = data.astype({"ZipCode_97234": np.int32})
    data = data.astype({"ZipCode_97236": np.int32})
    data = data.astype({"ZipCode_97246": np.int32})
    data = data.astype({"ZipCode_97249": np.int32})
    data = data.astype({"ZipCode_97250": np.int32})
    data = data.astype({"ZipCode_97261": np.int32})
    data = data.astype({"ZipCode_97270": np.int32})
    data = data.astype({"ZipCode_97288": np.int32})
    data = data.astype({"ZipCode_97297": np.int32})
    data = data.astype({"ZipCode_97299": np.int32})

    html_response = ""
    fig_waterfall = None
    if not erklärung:
        data_model = data
        """
        Prints the required input of model_xgb.

        """
        preds = model.predict(data_model)
        preds = int(preds)
        print(preds)
        print(type(preds))
        html_response = f"""
            <html>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin-top: 200px;
                    }}
                    
                    h1 {{
                        color: #333;
                        font-size: 24px;
                        text-align: center;
                    }}
                    
                    p {{
                        color: #777;
                        font-size: 22px;
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>Der vorgeschlagene Preis für die Immobilie beträgt:</h1>
                <p> {preds} €</p>
            </body>
            </html>
        """
    elif erklärung:
        preds = model.predict(data)
        preds = int(preds)
        html_response = f"""
            <html>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin-top: 200px;
                    }}
                    
                    h1 {{
                        color: #333;
                        font-size: 24px;
                        text-align: center;
                    }}
                    
                    p {{
                        color: #777;
                        font-size: 22px;
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>Der vorgeschlagene Preis für die Immobilie beträgt:</h1>
                <p>{preds} €</p>
            </body>
            </html>
        """
    return html_response

