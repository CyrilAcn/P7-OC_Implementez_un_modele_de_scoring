#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 16:02:46 2022

@author: cyril
"""

# 1. Library imports

import uvicorn
from fastapi import FastAPI


# 2. Create the app object

app = FastAPI()

# Création d'une liste de d'ID de clients
list_id = df.index.tolist()


# 3. Index route, opens automatically on http://127.0.0.1:8000

@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
