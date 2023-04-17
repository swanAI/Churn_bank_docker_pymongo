# Churn_bank_docker_pymongo
Le projet consiste en la création d'un pipeline de données pour extraire les données de churn d'une base de données MongoDB. Ensuite, une analyse des données a été effectuée pour comprendre les facteurs qui influencent l'attrition des clients d'une banque.

Le but ultime est de prédire l'attrition des clients en utilisant un modèle d'apprentissage automatique et de centraliser toutes les expériences avec MLflow.

Une fois le modèle entraîné et évalué, le pipeline a été déployé sur AWS EC2 à l'aide d'une API FastAPI et d'un frontend Streamlit pour construire une interface utilisateur destinée à la relation client bancaire. Cette interface permet de visualiser les prédictions du modèle et d'améliorer la connaissance des clients afin de pouvoir anticiper les clients susceptibles de quitter la banque.

Enfin, un système de surveillance et de collecte de métriques de performance du modèle a été mis en place à l'aide de Prometheus et une interface de monitoring a été créée avec Grafana pour surveiller et analyser les performances du modèle en temps réel.

# Déploiement du modèle sur cloud provider AWS avec instance EC2

Le lien pour accéder à l’application streamlit
http://ec2-3-84-195-206.compute-1.amazonaws.com:8501

Le lien pour accéder à FastAPi 
http://ec2-3-84-195-206.compute-1.amazonaws.com:8008/docs



