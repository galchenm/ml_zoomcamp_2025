import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    X = [lead.dict()]
    y_pred = model.predict_proba(X)[0, 1]
    return {"convert_probability": float(y_pred)}
