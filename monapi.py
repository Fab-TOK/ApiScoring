
from fastapi import FastAPI
import pandas as pd
import pickle

app= FastAPI()

@app.get("/")
async def bonjour():
    return { "Ok": "Hello world"}

@app.post("credit/")
async def calcul(age : int, revenu : int, habitation : str, activity: float, motif : str,  montprete : float, taux : float):
    model=pickle.load(open("modele.txt", "rb"))
    data = pd.DataFrame({'a': [age], 'b':[revenu], 'c':[habitation] , 'd':[activity] , 'e':[motif], 'f':[2], 'g':[montprete], 'h':[taux], 'i':[0.10]})
    code = {'education': 1,
           'medical' : 2,
           'entreprise' : 3,
           'personnel': 4,
           'consolidation': 5,
           'habitat': 6,
           'location' : 1,
           'hypothèque': 2,
           'propriétaire': 3,
           'autre': 4,
           'B': 2}
    for col in data.select_dtypes('object'):
        data.loc[:,col]= data[col].map(code)
    
    pred=model.predict(data)
    prob=model.predict_proba(data)
    
    return { "La prediction ": f"{pred[0]}", 
              "La propabilité ": f"{prob[0][1]}"}

