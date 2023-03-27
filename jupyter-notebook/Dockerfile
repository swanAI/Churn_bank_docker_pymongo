# Extends jupyter/datascience-notebook
FROM jupyter/scipy-notebook

# Install  
RUN pip install pymongo
RUN pip install colored
RUN pip install termcolor
RUN pip install lightgbm
RUN pip install xgboost
RUN pip install imblearn
RUN pip install optuna 
RUN pip install mlflow
RUN pip install lime 
RUN pip install shap
# Install jupyter_contrib_nbextensions with pip
RUN pip install jupyter_contrib_nbextensions


# Enable NbExtensions Configurator interface
RUN jupyter nbextensions_configurator enable --user








