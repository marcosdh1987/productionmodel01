import uvicorn
from fastapi import FastAPI, Request, Form
#to use templates
from fastapi.templating import Jinja2Templates


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
app = FastAPI()
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




# ML app
@app.get('/predict/')
async def predict(data):

    df = pd.DataFrame([x.split(',') for x in data.split('\n')])
    
    print(df)

   
    #print(query)
    prediction = lr.predict(df)
    prediction = str(prediction)
    prediction = [prediction.replace("[", "").replace("]", "")]

    if lr:
        try:

            return {"name":"Gas Flow Rate Predicted","value":prediction[0],"unit":"Sm3/D"}

        except:

            return {"name":"name"}
    else:
        print ('Train the model first')
        return ('No model here to use')



if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)