#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 16:02:46 2022

@author: cyril
"""

# 1. Import des librairies

import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib
import shap



# 2. Création de l'objet app

app = FastAPI()
model = joblib.load('best_model.joblib')
df = pd.read_csv("df.csv")
df.set_index('Unnamed: 0', inplace = True)


# Création d'une liste de d'ID de clients
list_id = df.index.tolist()


# 3. Index route, ouverture automatique sur http://127.0.0.1:8000

@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Route avec un seul paramètre, retourne le paramètre avec un massage
#    Localisé sur: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}


# 5. Fonction de prédiction, réalisation via les données au format
#   JSON and retourne the probabilité d'appartenance aux deux classes
#   ainsi qu'un avis

@app.post('/predict')
def predict_customer(id:int):

    if id in list_id:
        prediction = model.predict_proba([df.loc[id]])
        print(prediction[0,1])
        if(prediction[0,1]>0.5):
            return{"proba" : prediction[0,1], "avis" : "Avis défavorable"}
        else:
            return{"proba" : prediction[0,1], "avis" : "Avis favorable"}


# 5. Fonction de prédiction, réalisation via les données au format
#   JSON and retourne the probabilité d'appartenance aux deux classes
#   ainsi qu'un avis

#@app.post('/featimp')
#def predict_customer(id:int):

    #if id in list_id:
            #return{"proba" : prediction[0,1], "avis" : "Avis favorable"}

#if __name__ == '__main__':
    #uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
