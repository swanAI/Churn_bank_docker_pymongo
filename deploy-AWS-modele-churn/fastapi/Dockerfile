#pull official image docker python 
FROM python:3.9-slim

# Définir le répertoire de travail pour toute les instructions (RUN, COPY, CMD)
WORKDIR /fastapi

RUN pip install --upgrade pip==22.1

# Copier les dépendances dans le répertoire de travail
COPY requirements.txt /fastapi

#installer les dépendances 
RUN pip install -r requirements.txt

#Copier project dans répertoire
COPY . /fastapi

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1