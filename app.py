import pandas as pd
import joblib 
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.responses import FileResponse

app = FastAPI()
model = joblib.load('fake_job_post_detector_model.pkl')

class job(BaseModel): #used for input validation
    
    #text columns = default value
    title : str
    company_profile : str = ""
    description : str = "" 
    requirements : str = ""

    #categorical columns
    employment_type : str = "Unknown"
    required_experience : str = "Unknown"
    required_education : str = "Unknown"

    #Numerical columns
    telecommuting : int
    has_company_logo : int
    has_question : int 

@app.get("/")
def home():
    return FileResponse('index2.html')

@app.post("/predict")
def predict(job : job):
    text = f"{job.title} {job.company_profile} {job.description} {job.requirements}"

    row = pd.DataFrame([{
        'corpus' : text,

        'employment_type' : job.employment_type,
        'required_experience' : job.required_experience,
        'required_education' : job.required_education,

        'telecommuting' : job.telecommuting,
        'has_company_logo' : job.has_company_logo,
        'has_questions' : job.has_question
    }])

    prediction = float(model.predict_proba(row)[0][1])
    return {'fraudulent': int (prediction > 0.5), "probability" : round(prediction, 2)}
