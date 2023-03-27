
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from termcolor import colored
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import timeit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report



class MLFlowRandomizedSearchCV:
    def __init__(self, pipeline, name_pipeline, X_train, y_train, X_test, y_test, scoring, costs, experiment_name, uri, artifact_location, source_df, source_stockage):
        """
        Constructeur pour la classe MLFlowRandomizedSearchCV.
        
        Args:
        - pipeline (estimator): Pipeline contenant l'ensemble des étapes à effectuer sur les données.
        - name_pipeline (str): Nom du pipeline.
        - X_train (dataframe): dataframe contenant les données d'entraînement pour les features.
        - y_train (dataframe): Dataframe contenant les données d'entraînement pour les labels.
        - X_test (dataframe): Dataframe contenant les données de test pour les features.
        - y_test (dataframe): Dataframe contenant les données de test pour les labels.
        - scoring (myscorer): Métrique personnalisée utiliser pour évaluer les performances du modèle.
        - costs (function): Fonction personnalisée 
        - experiment_name (str): Nom de l'expérimentation.
        - uri (str): URI pour accéder à la plateforme de tracking.
        - artifact_location (str): Emplacement pour stocker les artefacts.
        - source_df (str): Nom du dataframe source.
        - source_stockage (str): Type de stockage pour les données.
        """
        self.pipeline = pipeline
        self.name_pipeline = name_pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring = scoring
        self.costs = costs
        self.experiment_name = experiment_name
        self.uri = uri
        self.artifact_location = artifact_location
        self.source_df = source_df
        self.source_stockage = source_stockage
        
        
        
    
    def run_experiment(self, params):
        """
        Méthode pour effectuer la random search CV sur la pipeline et Enregistrer les log_metric, log_param, log_model sur le registre de MLFlow
        
        Args:
        - params (dict): Dictionnaire contenant les hyperparamètres à tester.
        
        Returns:
        - best_model (estimator): Meilleur pipeline trouvé grâce à la recherche.
        """
        
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)

        mlflow_client = MlflowClient(tracking_uri=self.uri)
        experiment = mlflow_client.get_experiment_by_name(self.experiment_name)
        # Si expérimentation n'existe pas, création d'une nouvelle expérimentation 
        # Si expérimentation existe Run sur celle-là 
        if experiment is None:
            experiment_id = mlflow_client.create_experiment(self.experiment_name, artifact_location=self.artifact_location)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(run_name=f"{self.name_pipeline}RandomsearchCV", experiment_id=experiment_id):
            # définir les seuils de probabilité à tester
            thresholds = np.arange(0, 1, 0.1)
            start_time_model = timeit.default_timer()
            Random_grid = RandomizedSearchCV(self.pipeline, param_distributions=params, cv=5, scoring=self.scoring, random_state=77)
            pipe_optimized = Random_grid.fit(self.X_train, self.y_train)
            end_time_model = round(timeit.default_timer() - start_time_model, 3)
    
            # sélectionner le meilleur modèle trouvé
            best_model = pipe_optimized.best_estimator_
    
            # évaluer les performances du modèle sur les données de test
            y_prob = best_model.predict_proba(self.X_test)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_prob)
            auc = metrics.auc(fpr, tpr)
            mlflow.log_metric("auc", auc)
    
            # trouver le meilleur seuil et le stocker dans les paramètres
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            mlflow.log_param("best_threshold", best_threshold)
    
            myscorer_mean = pipe_optimized.best_score_
            mlflow.log_metric("myscorer score CV", myscorer_mean)
            mlflow.log_metric("score eval test data", pipe_optimized.score(self.X_test, self.y_test))

            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
                mlflow.log_param("Le temps pour la randomsearhCV", end_time_model)
                mlflow.log_param("Les meilleurs paramètres", pipe_optimized.best_params_)
                mlflow.set_tag(self.name_pipeline, self.experiment_name)
                mlflow.set_tag("La source de stockage", self.source_stockage)
                mlflow.set_tag("df_source", self.source_df)
                mlflow.set_tag("X_train", self.X_train)
                mlflow.set_tag("y_train", self.y_train)
                mlflow.set_tag("y_test", self.y_test)
                mlflow.set_tag("X_test", self.X_test)
                mlflow.sklearn.log_model(best_model, f"{self.name_pipeline}_randomsearhCV")
            # Print and log best parameters and metrics
            print(colored("Les résultats de la randomsearhCV".upper(), 'blue'))
            print()
            print(colored("Les meilleurs paramètres pour la recherche des paramètres:", 'blue'), pipe_optimized.best_params_)
            print()
            # afficher le meilleur score et le meilleur seuil
            
            print(colored("Le best_score randomsearhCV :", 'blue'), pipe_optimized.best_score_)
            print(colored("Le temps pour la randomsearchCV: ", 'blue'), end_time_model)
    
        # End the MLflow run
        mlflow.end_run()
        return best_model
    
    def plot_learning_curve(self,best_model):
        
        """
        Trace une courbe d'apprentissage pour le meilleur modèle donné, en utilisant la méthode learning_curve() de scikit-learn.
    
        Args:
        - self (object): Instance de la classe  MLFlowRandomizedSearchCV contenant les données et les paramètres.
        - best_model (object): Meilleur pipeline pour lequel la courbe d'apprentissage sera tracée.
        
        Returns:
        - plot learning curve 
        """
        
        train_sizes, train_scores, test_scores = learning_curve(estimator=best_model, X=self.X_train, y=self.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring=self.scoring)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        plt.title(f"Learning curve pour {self.name_pipeline}")
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training fontion cout métier')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation fontion cout métier')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid(True)
        plt.xlabel('Number of training samples')
        plt.ylabel('fonction_cout métier')
        plt.legend(loc='lower right')
        plt.show()
        
    
    def eval_plot_curve(self, best_model):
        """
        Évalue et trace la courbe ROC et la courbe précision-rappel pour le meilleur modèle donné.
    
        Args:
        - self (object): Instance de la classe MLFlowRandomizedSearchCV contenant les données et les paramètres.
        - best_model (object): Meilleur pipeli pour lequel la courbe ROC et la courbe précision-rappel seront tracées.
    
        Returns:
        - plot courbe ROC et la courbe précision-rappel
        """
        best_model.fit(self.X_train, self.y_train)
        y_prob = best_model.predict_proba(self.X_test)[:,1]
        y_pred = best_model.predict(self.X_test)
      
        fpr, tpr, _ = metrics.roc_curve(self.y_test, y_prob)
    
        auc = metrics.roc_auc_score(self.y_test, y_prob)
      
        #create ROC curve
        plt.figure(figsize=(12,6))
        plt.subplot(1, 2, 1)
        plt.title('ROC Curve', fontsize=18)
        plt.plot(fpr,tpr,label="AUC="+str(np.round(auc,4)))
        plt.plot(np.linspace(0,1),np.linspace(0,1),linestyle='--',color='blue',label='lgbm classifier')
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xlabel('False Positive Rate',fontsize=14)
        plt.legend(loc=4)
      
        # find optimal threshold using custom metric
        #thresholds = np.arange(0, 1.01, 0.01)
        thresholds = np.arange(0, 1, 0.01)
        scores = []
        for threshold in thresholds:
            y_pred_thresh = np.where(y_prob > threshold, 1, 0)
            score = self.costs(self.y_test, y_pred_thresh)
            scores.append(score)
            
        # convert scores to numpy array
        scores = np.array(scores)
            
        # get index of optimal threshold
        optimal_threshold_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_threshold_idx]
        print('Optimal Threshold:', optimal_threshold)
            
        #calculate precision and recall using optimal threshold
        y_pred_optimal_thresh = np.where(y_prob > optimal_threshold, 1, 0)
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob, pos_label=1)
        optimal_precision = precision[optimal_threshold_idx]
        optimal_recall = recall[optimal_threshold_idx]
        print('Optimal Precision:', optimal_precision)
        print('Optimal Recall:', optimal_recall)
    
        #create precision recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue')
        plt.plot(optimal_recall, optimal_precision, 'o', markersize=10, color='red')
        
        ##add axis labels to plot
        plt.title('Precision-Recall Curve', fontsize=18)
        plt.ylabel('Precision', fontsize=14)
        plt.xlabel('Recall',fontsize=14)
        ##display plot
        plt.tight_layout()
        plt.show()
 
    
    
    def eval_best_threshold(self, best_model):
        
        """ Évalue la meilleure valeur seuil pour un modèle donné.

        Args:
            self (object): Instance de la classe MLFlowRandomizedSearchCV
            best_model (object): Meilleur modèle sélectionné lors de l'optimisation.
    
        Returns:
            plot threshold
            
        """
        thresholds = np.arange(0, 1, 0.01)

        # initialiser les scores
        best_score = 0
        best_threshold = 0
        scores = []   
        # itérer sur les seuils de probabilité et calculer le score pour chaque seuil
        for threshold in thresholds:
            y_pred = (best_model.predict_proba(self.X_test)[:, 1] >= threshold).astype(int)
            score = self.costs(self.y_test, y_pred)
            scores.append(score)# **
            # mettre à jour le meilleur score et le meilleur seuil si le score actuel est meilleur
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print("Best score pour seuil de probabilité optimal:", best_score)
        print("Best seuil de probabilité optimal:", best_threshold)      
      
        scores = np.asarray(scores)
        ix = np.argmax(scores)
    
        best_threshold, best_score = (thresholds[ix], round(scores[ix], 3))

     
        plt.subplots(1, figsize=(6, 5))
        plt.plot(thresholds, scores, lw=1)
        plt.axvline(best_threshold, ls="--", lw=1, c="r")
        plt.title("Threshold")
        plt.xlabel("proba threshold")
        plt.ylabel("Score performed")
        plt.show()
          
    
    def eval_confusion_matrix_threshold(self, best_model, threshold):
        
        """ Évalue la matrice de confusion pour un modèle et un seuil donnés.

        Args:
            self (object): Instance de la class MLFlowRandomizedSearchCV
            best_model (object): Meilleur modèle sélectionné lors de l'optimisation.
            threshold (float): Valeur seuil pour la classification binaire.
    
        Returns:
            plot Matrice confusion
        """
        
        # Get predictions and classification report for test set
        print(colored(f"Evaluation de la {self.name_pipeline} sur des données de test avec les bests paramètres".upper(), 'red'))
        print()
        #print(self.pipe_optimized.best_params_)
        #print(best_model.get_params())
        print()
        #print(colored("Score eval test data :",'red'),best_model.score(self.X_test, self.y_test))
        print()
        print()
        
        y_prob = best_model.predict_proba(self.X_test)
        y_pred = best_model.predict(self.X_test)
        
        threshold = threshold # definir threshold ici
        # Nouveau ypred_test avec threshold définis 
        ypred_tresh = [1 if y_prob[i][1]> threshold else 0 for i in range(len(y_prob))]
        
        cf_matrix = confusion_matrix(self.y_test, ypred_tresh)
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes()
        plt.title(f"Matrice de confusion pour {self.name_pipeline}", size=20, y=1.1)
        group_names = ["True nég\n(client reste et modèle prédit reste)","False pos\n(ERROR 1 perte opportunitée pour la banque)\n(client reste et modèle prédit désabonnement)","False nég\n(ERROR 2 perte argent pour la banque))\n(client désabonne et modèle prédit reste)","True pos\n(client désabonne et modèle prédit désabonnement)",]
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}"    for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels,fmt="", cmap="Reds", xticklabels=["Reste (0 = négatif)", "Désabonnement (1 = positif)"],yticklabels=["Reste (0 = négatif)", "Désabonnement (1 = positif)"],)
        plt.ylabel("Classe réelle", fontsize=14)
        plt.xlabel("Classe prédite", fontsize=14)
        plt.show()
        print()
        print(classification_report(self.y_test, ypred_tresh))
        print()
    
    def __repr__(self):
        """
        Renvoie une représentation sous forme de chaîne des variables d'instance de l'objet.

        Retour:
        str : représentation sous forme de chaîne des variables d'instance de l'objet.
        """
        
        return str(self.__dict__)

        
   