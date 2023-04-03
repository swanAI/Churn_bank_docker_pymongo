# 1. Library imports
import pandas as pd 
import uvicorn
from fastapi import FastAPI, HTTPException
import joblib


# Creer l'objet app
app = FastAPI(title="App Churn prédictions for Bank Customers",
    description=""" Churn prédictions et la probabilité qu’un client reste ou part de la banque"""
    )


@app.get("/")
async def root():
    return {"message": "Hello World"}



#importer dataframe des données clients tests et les vraies étiquettes
X_testprod = pd.read_csv('X_testprod.csv', index_col=[0])

# Création list des clients 
clients_id = X_testprod["customer_id"].tolist() 
# mettre customer_id en index
X_testprod_request  = X_testprod.set_index('customer_id')

#y_testprod = pd.read_csv('y_testprod.csv', index_col=[0])

# Création list des clients 


# fonction predict
@app.get('/predict/{id}')
async def fonction_predict_LGBM(id: int):

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    
    else:
        
        
        pipe_prod = joblib.load('Pipe_prod.pkl')
    
        #y_test_real = y_testprod.set_index('customer_id').loc[[id]]['churn'].values[0]
        # Sélecionner un id client 
        values_id_client = X_testprod_request.loc[[id]]
             
        # predict proba sur le client id sélectionné 
        prob_preds = pipe_prod.predict_proba(values_id_client)
        
        #Définir le best threshold 
        threshold = 0.43 # definir threshold ici
        y_test_prob = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
        
      
        # Comparer la prédiction modèle avec la valeur réelle correspondante
        #if y_test_real == y_test_prob[0]:
            #result = "correct"
        #else:
            #result = "incorrect"
  
  
        return {"prediction": y_test_prob[0],
          "probability_0" : prob_preds[0][0],
          "probability_1" : prob_preds[0][1],}
          #"result": result}
    

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
    #uvicorn.run("hello_world_fastapi:app")
   # uvicorn.run(app, host='127.0.0.1', port=8000)





