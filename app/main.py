import uvicorn
from fastapi import FastAPI, Request, Form
#to use templates
from fastapi.templating import Jinja2Templates
from mangum import Mangum  # <---------- import Mangum library

#to data handle
import pandas as pd
import tensorflow as tf

# ML import
import joblib,os,io
import numpy as np


#Models
lr = joblib.load("gas_model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded',model_columns)

    
#init app
app = FastAPI(title='Serverless Lambda FastAPI')

handler = Mangum(app=app) # <----------- wrap the API with Mangum

templates = Jinja2Templates(directory="templates")

# Routes
@app.get('/')
def index(request: Request):
    result = "Type a number"
    return templates.TemplateResponse("form.html", context={"request":request, 'result': result})

@app.post('/')
async def form_post(request: Request, press: float = Form(...), dp: str = Form(...), temp : float = Form(...),
                vel : float = Form(...),liqden : float = Form(...),wc : float = Form(...),choke : float = Form(...)):
    

    df = pd.DataFrame(columns = model_columns)
    df = df.append({'Pressure[Bar]': press, 'DP[Bar]': dp, 'Temperature[C]': temp, 'Velocity[m/s]':vel,
                    'LiqDen[kg/m3]':liqden, 'WaterCut[%]':wc, 'choke':choke}, ignore_index=True)
    
    print(df.head())
    #prediction = lr.predict(df)
    data_tensor =np.array(df.astype(np.float32))
    #data_tensor = tf. convert_to_tensor(df.values)
    print(data_tensor)
    prediction = lr.predict(data_tensor)
    #result = df
    result = str(prediction)
    result = [result.replace("[", "").replace("]", "")]
    


    return templates.TemplateResponse("form.html", context={"request":request,'text':"The predicted Gas production is: " ,'result': result[0], 'unit':" m3/D"})




if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)