from fastapi import FastAPI
from pydantic import BaseModel
from ml_model import *
import json
import yaml



app = FastAPI()

class PredictionRequest(BaseModel):
    LivingSpace: float
    Rooms: float
    abstellraum: bool
    bad_wc_getrennt: bool
    barriefrei: bool
    dusche: bool
    elektro: bool
    erdwaerme: bool
    fenster: bool
    ferne: bool
    fliesen: bool
    frei: bool
    fussbodenheizung: bool
    gaestewc: bool
    garage: bool
    kable_sat_tv: bool
    kontrollierte_be_entlueftungsanlage: bool
    kunststofffenster: bool
    luftwp: bool
    parkett: bool
    personenaufzug: bool
    reinigung: bool
    rollstuhlgerecht: bool
    speisekammer: bool
    terrasse: bool
    wanne: bool
    zentralheizung: bool
    ZipCode: str

@app.post("/predict")
async def predict_housing_price(data: dict):
    result = predict(data)
    return {"Predicted price": result}

@app.get("/model-info")
async def model_info():
    with open("prod_model_details.txt", "r") as file:
        content = file.read()
    
    # Parse YAML content and convert it to JSON
    json_content = yaml.safe_load(content)
    
    return json_content

@app.get("/")
async def root():
    return {"Status": "Health check - API is running"}

