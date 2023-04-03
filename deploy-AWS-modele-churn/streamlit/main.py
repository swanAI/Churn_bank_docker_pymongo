import streamlit as st
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import requests
import joblib
import pickle
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer 


##############################################################
# Fonction shap plot html 
###############################################################
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
##################################################################

############################################################################
#Cette fonction fetch(session, url) permet d'envoyer une requête GET 
#à l'URL spécifiée en utilisant une session HTTP fournie en paramètre.
#############################################################################
def fetch(session, url):

    try:
      result = session.get(url)
      return result.json()
    except Exception:
        return {}

session = requests.Session()
######################################################################


#########################################################################
#La fonction prend en argument l'identifiant d'un client et utilise 
#la fonction fetch(session, url) pour envoyer une requête GET à API 
#FasApi
######################################################################

def client_prediction(id):  
  # Récupérer prédiction du modéle pour un client donner
  response = fetch(session, f"http://fastapi:8008/predict/{id}")
  if response:
    return response
  else:
    return "Error"

######################################################################



###############################################################
# fonction plot prédiction local avec shap pour un client donné
##############################################################

def explain_plot(id: int, pred: int):
    
    # afficher plot init java script
    shap.initjs()
    # mettre 'customer_id' en index et lui donner un id client 
    df_shap_local = X_testprod.set_index(['customer_id']).loc[[id]]
   
    # Création d'un explainer pour le modèle de prédiction
    explainer = shap.TreeExplainer(pipe_prod.named_steps['LGBM'])
    # étape tranformation des données avec le preprocessing
    X_test_transformed = pipe_prod[:-2].transform(df_shap_local)
    # récupérer les features aprés la transformation
    features_names = pipe_prod['preprocessor_trees'].get_feature_names_out()
    # Calcul des valeurs SHAP pour les données de test transformées
    shap_values = explainer.shap_values(X_test_transformed)
    # Affichage du graphique SHAP pour la classe cible spécifiée
    if pred == 1:
       p = st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:-1], X_test_transformed[0,:-1]))#, feature_names=features_names))
    else:
       p = st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][0,:-1], X_test_transformed[0,:-1]))#, feature_names=features_names))
    return p

######################################################################


# load la pipeline LGBM entrainée
pipe_prod = joblib.load('Pipe_prod.pkl')
# importer les données de production test
X_testprod = pd.read_csv('X_testprod.csv', index_col=[0])
# Création liste des customer_id pour st.sidebar.selectbox
list_client_prod = X_testprod["customer_id"].tolist() 

# logo sidebar 
st.sidebar.image("image_churn.png", use_column_width=True)

# Liste clients id sidebar 
client_id = st.sidebar.selectbox("Client Id list",list_client_prod)
client_id = int(client_id)




# GET prédiction et probabilité d'appartenance à une classe pour un client id 
prediction = client_prediction(client_id)

pred = prediction["prediction"]

probability_value_0 = round(prediction["probability_0"] * 100,2)
probability_value_1 = round(prediction["probability_1"] * 100,2)


#Création tabs (onglets)
tab1, tab2 = st.tabs(["Prédiction de désabonnement", "AgGrid"])

# définir AgGrid pour tabs1
with tab1:
   # Titre 
   st.title("**Prédiction de désabonnement**")
   st.markdown("un modèle d'apprentissage automatique capable de prédire si un client de la banque va partir ou rester. Pour cela, le modèle utilisera un score d'attrition pour évaluer la probabilité que le client ferme son compte bancaire et quitte la banque. Ainsi, le modèle pourra aider la banque à prendre des décisions éclairées pour retenir ses clients et améliorer leur satisfaction.")
   st.header(f'*Prédiction du modéle pour le client {client_id}*')
   # bloc conditionnel pour class si 1 sinon 0 (gauge proba)
   if pred == 1: 
    st.error('Le client quitte la banque')
    option_1 = {
          "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
          "series": [
              {
                  "name": "Pressure",
                  "type": "gauge",
                  "axisLine": {
                      "lineStyle": {
                          "width": 10,
                      },
                  },
                  "progress": {"show": "true", "width": 10},
                  "detail": {"valueAnimation": "true", "formatter": "{value}"},
                  "data": [{"value": probability_value_1, "name": "Probabilité %"}],
              }
          ],
      }
      
    st_echarts(options=option_1, width="100%", key=0)
    st.header(f'*Les données qui ont le plus influencé le calcul de la prédiction pour le client {client_id}*')
    # afficher shap explain pour client qui quitte la banque 
    try:
      explain_plot(client_id, pred)
    except Exception as e:
      st.write(f"Exception : {e}")
   # sinon le client reste et affichage gauge probabilité 
   else:
    st.success('Le client reste dans la banque')
    option = {
          "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
          "series": [
              {
                  "name": "Pressure",
                  "type": "gauge",
                  "axisLine": {
                      "lineStyle": {
                          "width": 10,
                        },
                    },
                    "progress": {"show": "true", "width": 10},
                    "detail": {"valueAnimation": "true", "formatter": "{value}"},
                    "data": [{"value": probability_value_0, "name": "Probabilité %"}],
               }
           ],
       }
   
    st_echarts(options=option, width="100%", key=0)
 
    st.header(f'*Les données qui ont le plus influencé le calcul de la prédiction pour le client {client_id}*')
    
    # afficher shap explain pour client qui quitte la banque 
    try:
        explain_plot(client_id, pred)
    except Exception as e:
        st.write(f"Exception : {e}")
   
   st.header("*Les variables les plus significatives par ordre décroissant et qui ont un pouvoir prédictif élevé.*")
   st.image("Features_importance.png", use_column_width=True)

# définir AgGrid pour tabs2
with tab2:
   st.title("**Les informations sur les clients**")
   st.markdown("Voici la table des clients pour l'analyse")
   
   # Configuration de la grille
   gb = GridOptionsBuilder.from_dataframe(X_testprod)
   gb.configure_pagination()
   gb.configure_side_bar()
   gb.configure_selection(selection_mode="multiple", use_checkbox=True)
   gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
   gridOptions = gb.build()

   # Afficher la grille
   data = AgGrid(X_testprod, 
                 gridOptions=gridOptions, 
                 enable_enterprise_modules=True, 
                 allow_unsafe_jscode=True, 
                 update_mode=GridUpdateMode.SELECTION_CHANGED)
   # Vérifier si des colonnes ont été sélectionnées
   selected_columns = data.get('selected_columns', [])
   #st.write(selected_data.columns)
   if len(selected_columns) > 0:
       # Filtrer les données pour n'inclure que les colonnes sélectionnées
       selected_data = df[selected_columns]
       st.write(selected_data.columns)
       # Identifier les types de données pour chaque colonne
       column_types = selected_data.dtypes

       # Sélectionner les colonnes quantitatives
       quantitative_columns = list(column_types[column_types != 'object'].index)

       # Sélectionner les colonnes qualitatives
       qualitative_columns = list(column_types[column_types == 'object'].index)
      
       # Vérifier si des lignes ont été sélectionnées
       selected_rows = data["selected_rows"]
       if len(selected_rows) > 0:
           # Filtrer les données pour n'inclure que les lignes sélectionnées
           selected_data = selected_data.loc[selected_rows.index]
        
           # Vérifier si les variables sont compatibles avec les graphiques
           if len(quantitative_columns) > 0:
               # Afficher l'histogramme des variables quantitatives
               for col in quantitative_columns:
                   fig = px.histogram(selected_data, x=col, nbins=20)
                   st.plotly_chart(fig)
        
           if len(qualitative_columns) > 0:
               # Afficher le pie chart des variables qualitatives
               for col in qualitative_columns:
                   fig = px.pie(selected_data, values=col, names=col)
                   st.plotly_chart(fig)

           # Vérifier si les variables sont compatibles avec les graphiques bivariés
           if len(quantitative_columns) > 1:
               # Afficher le scatterplot des variables quantitatives
               fig = px.scatter(selected_data, x=quantitative_columns[0], y=quantitative_columns[1])
               st.plotly_chart(fig)
            
               # Afficher le boxplot des variables quantitatives
               fig = px.box(selected_data, x=quantitative_columns[0], y=quantitative_columns[1])
               st.plotly_chart(fig)
            
           if len(qualitative_columns) > 1:
               st.error('Impossible d\'afficher des graphiques bivariés pour des variables qualitatives')
       else:
           st.warning('Sélectionnez au moins une ligne pour afficher des graphiques')
   else:
       st.warning('Sélectionnez au moins une colonne pour afficher des graphiques')

