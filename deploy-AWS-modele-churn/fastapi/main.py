# Importer les bibliothèques nécessaires
from fastapi import FastAPI, HTTPException, Response
import joblib
import pandas as pd
from prometheus_client import Gauge, start_http_server, generate_latest, REGISTRY
from sklearn.metrics import f1_score, recall_score, precision_score

# Créer l'objet FastAPI
app = FastAPI(title="App Churn prédictions for Bank Customers",
    description=""" Churn prédictions et la probabilité qu’un client reste ou part de la banque"""
    )

# Exposer les métriques via Prometheus sur le port 9090
start_http_server(9090)

# Charger les données de test et les étiquettes réelles
X_testprod = pd.read_csv('X_testprod.csv', index_col=[0])
y_testprod = pd.read_csv('y_testprod.csv', index_col=[0])

# Créer une jauge Prometheus pour chaque métrique
f1_score_gauge = Gauge('f1_score', 'F1 score metric', ['customer_id'])
recall_score_gauge = Gauge('recall_score', 'Recall score metric', ['customer_id'])
precision_score_gauge = Gauge('precision_score', 'Precision score metric', ['customer_id'])

# Définir la fonction de prédiction avec des métriques de performance
@app.get('/predict/{id}')
async def fonction_predict_LGBM(id: int):

    # Vérifier si l'ID du client est valide
    if id not in X_testprod["customer_id"].tolist():
        raise HTTPException(status_code=404, detail="client's id not found")
    
    else:
        # Charger le pipeline de prétraitement et le modèle de prédiction
        pipe_prod = joblib.load('Pipe_prod.pkl')

        # Prétraiter les données du client avec le pipeline
        values_id_client = X_testprod.set_index('customer_id').loc[[id]]

        # Faire la prédiction de la probabilité de churn pour ce client
        prob_preds = pipe_prod.predict_proba(values_id_client)

        # Sélectionner le seuil de classification optimal
        threshold = 0.43 # définir le seuil ici
        y_test_prob = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]

        # Calculer les métriques d'évaluation
        y_test_real = y_testprod.set_index('customer_id').loc[[id]]['churn'].values[0]
        f1 = f1_score([y_test_real], y_test_prob, average='macro', zero_division=1)
        recall = recall_score([y_test_real], y_test_prob, average='macro', zero_division=1)
        precision = precision_score([y_test_real], y_test_prob, average='macro', zero_division=1)

        # Enregistrer les métriques dans Prometheus
        f1_score_gauge.labels(customer_id=id).set(f1)
        recall_score_gauge.labels(customer_id=id).set(recall)
        precision_score_gauge.labels(customer_id=id).set(precision)

        # Retourner la prédiction et les probabilités associées
        return {"prediction": y_test_prob[0],
                "probability_0" : prob_preds[0][0],
                "probability_1" : prob_preds[0][1],
                }

# Endpoint pour les métriques 
@app.get("/metrics")
async def endpoint_metrics():
    return Response(generate_latest(REGISTRY))
