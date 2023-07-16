import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_model import *



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
    ConstructionYear: float
    EstateType_APARTMENT: bool
    DistributionType_RENT: bool
    ZipCode: str

@app.post("/predict")
async def predict_housing_price(data: dict):
    result = predict(data)
    return {"Predicted price": result}

