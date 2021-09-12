import uvicorn
from fastapi import FastAPI
import pandas as pd

# ML import
import joblib,os,io

# Vectorizer


#Models
lr = joblib.load("gas_model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded',model_columns)

    
#init app
app = FastAPI()

# Routes
@app.get('/')
async def index():
    return {'text':'welcome'}

@app.get('/items/{name}')
async def get_items(name):
    return {"name":name}

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