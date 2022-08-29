#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:52:18 2022

@author: cyril
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import requests
import shap
import pickle
import altair as alt


# Définition des path
path ='http://127.0.0.1:8000/'
APP_PRED = 'predict'
APP_FEAT = 'featimp'

# Liste des ID de clients
df = pd.read_csv("df.csv")
df.set_index('Unnamed: 0', inplace = True)

# Liste des prédictions de probas
pred_prob = pd.read_csv("prob_pred_pos.csv")

# Création d'un df pour le scatterplot avec les prédictions de probas
pred_prob_sample = pred_prob.head(100)
scatterplot = pd.read_csv("df.csv")
scatterplot = scatterplot.join(pred_prob_sample['Taux de risque'])
scatterplot.set_index('Unnamed: 0', inplace = True)

# Création d'une liste de d'ID de clients
list_id = df.index.tolist()

with open('explainer', 'rb') as f1:
   exp = pickle.load(f1)
with open('shap_values', 'rb') as f2:
   shap_val = pickle.load(f2)

shap_values = exp.shap_values(df)



# Requests permet d'envoyer des requêtes HTTP en Python.
# Elle permet donc d'utiliser des API.
def request_prediction(model_pred, data):

    data_json = {'id': data}
    # Requête sur le seveur Fast api
    response = requests.post(path + APP_PRED, params=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, 
                                                       response.text))

    return response.json()


# Requests permet d'envoyer des requêtes HTTP en Python.
# Elle permet donc d'utiliser des API.
#def request_featimp(model_pred, data):

    #data_json = {'id': data}
    #Requête sur le seveur Fast api
    #response = requests.post(path + APP_FEAT, params=data_json)

    #if response.status_code != 200:
        #raise Exception(
            #"Request failed with status {}, {}".format(response.status_code, 
                                                       #response.text))

    #return response.json()


def main():
    customer_id = st.sidebar.selectbox(
        "Entrez le numéro d'identification du client",
        list_id)
    # On récupère le numéro de l'index correspondant au client sélectionné.
    pos = (df.index.get_loc(customer_id))
    predict_btn = st.sidebar.button('Prédire')
    

    
    with st.container():
        st.title('Client')

        col1, col2 = st.columns(2)

        with col1:
            st.text('Jauge de risque estimé')
            #predict_btn = st.button('Prédire')
            if predict_btn:
                pred = None
                pred = request_prediction(APP_PRED, customer_id)    
                chart_data = pd.DataFrame( [[1-pred.get("proba"),
                                             pred.get("proba")]])
                st.bar_chart(chart_data)
                st.write(
                    'Le risque calculé est de {:.2f}'.format(pred.get("proba"))) 

        with col2:
            st.text('Variables importantes pour la prédiction')
            if predict_btn:
                #feat = None
                #feat = request_featimp(APP_FEAT, customer_id)
                shap.plots._waterfall.waterfall_legacy(exp.expected_value[1],
                                                       shap_values[1][pos,:], 
                                                       feature_names=df.columns, 
                                                       max_display=10)
                plt.pyplot.savefig('waterfall.png')
                st.write()
                #st.write(
                #        'Retour: {}'.format(feat.get("feature")))
                st.image('/tmp/waterfall.png')
    
        if predict_btn:
            if pred.get("avis")=="Avis favorable":
                st.success(
                'Décision concernant le crédit : {}'.format(pred.get("avis"))) 
            else:
                st.error(
                'Décision concernant le crédit : {}'.format(pred.get("avis"))) 
    
    st.text("__________________________________________________________________________________")

    with st.container():
        st.title('Portefeuille') 
        
        col3, col4 = st.columns(2)

        with col3:
            st.text("Distribution des risques portefeuille")
            st.bar_chart(pred_prob["Taux de risque"].value_counts(ascending=True),width=150,height=450)
            
        with col4:
            st.text("Features importance globales")
            st.image('/tmp/imp_val.png')

    choix2=df.columns.values.tolist()
    var_sel2 = st.selectbox("Choisissez une variable pour voir sa relation avec la classe des clients", choix2)

    c = alt.Chart(scatterplot).mark_circle().encode(
         x=var_sel2, y='Taux de risque')
    st.altair_chart(c, use_container_width=True)            


if __name__ == '__main__':
    main()
