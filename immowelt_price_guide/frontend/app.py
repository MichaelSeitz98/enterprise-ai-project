import gradio as gr
import requests
import json

# Define the prediction function
def predict_housing_price(
    feature_squrmeter,
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
    feature_zip
):  

    # Prepare the data dictionary
    data = {
        "LivingSpace": feature_squrmeter,
        "Rooms": feature_rooms,
        "altbau_(bis_1945)": features_altbau,
        "balkon": feature_balkon,
        "barriefrei": feature_ba,
        "dachgeschoss": feature_dachgeschoss,
        "einbaukueche": feature_einbaukueche,
        "neubau": feature_neubau,
        "parkett": feature_parkett,
        "stellplatz": feature_stellplatz,
        "bad/wc_getrennt": feature_badwc_getrennt,
        "personenaufzug": feature_personenaufzug,
        "garten": feature_garten,
        "garage": feature_garage,
        "renoviert": feature_renoviert,
        "terrasse": feature_terrasse,
        "wanne": feature_wanne,
        "zentralheizung": feature_zentralheizung,
        "abstellraum": feature_abstellraum,
        "ferne": feature_fernwaerme,
        "fussbodenheizung": feature_fussbodenheitzung,
        "gartennutzung": feature_gartenmitbenutzung,
        "kelleranteil" : feature_kellerabteil,
        "ZipCode": feature_zip
        }


    print(data)

    # Send a POST request to the /predict endpoint
    response = requests.post("http://localhost:8080/predict", json=data)

    print(response)

    return json.loads(response.text)["Predicted price"]


# Define the Gradio interface
iface = gr.Interface(
    fn=predict_housing_price,
    inputs=[
        gr.inputs.Slider(minimum=0, maximum=1000, step=1, label="Living Space (sqm)"),
        gr.inputs.Slider(minimum=0, maximum=15, step=1, label="Number of Rooms"),
        gr.inputs.Checkbox(label="Historic Building (Pre-1945)"),
        gr.inputs.Checkbox(label="Balcony"),
        gr.inputs.Checkbox(label="Barrier-Free"),
        gr.inputs.Checkbox(label="Attic"),
        gr.inputs.Checkbox(label="Fitted Kitchen"),
        gr.inputs.Checkbox(label="New Building"),
        gr.inputs.Checkbox(label="Parquet Flooring"),
        gr.inputs.Checkbox(label="Parking Space"),
        gr.inputs.Checkbox(label="Separate Bath/Toilet"),
        gr.inputs.Checkbox(label="Elevator"),
        gr.inputs.Checkbox(label="Garden"),
        gr.inputs.Checkbox(label="Garage"),
        gr.inputs.Checkbox(label="Renovated"),
        gr.inputs.Checkbox(label="Terrace"),
        gr.inputs.Checkbox(label="Bathtub"),
        gr.inputs.Checkbox(label="Central Heating"),
        gr.inputs.Checkbox(label="Storage Room"),
        gr.inputs.Checkbox(label="Distance"),
        gr.inputs.Checkbox(label="Underfloor Heating"),
        gr.inputs.Checkbox(label="Shared Garden"),
        gr.inputs.Checkbox(label="Cellar Compartment"),
        gr.inputs.Dropdown(label="PLZ", choices=['97072: Sanderau', '97074: Frauenland'])
        #gr.inputs.Textbox(label="Zip Code"),
    ],
    outputs=gr.outputs.Textbox(label="Monthly rent in EUR"),
    title="Renting Price Prediction",
    description=""
)

# Run the Gradio interface
iface.launch(server_port=8000)