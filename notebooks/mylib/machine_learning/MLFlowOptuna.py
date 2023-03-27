
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.io as pio
import plotly.express as px
from termcolor import colored
import optuna
import mlflow
from mlflow.tracking import MlflowClient
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report


class MLFlowOptuna:
    def __init__(self, pipeline, name_pipelines, param_grid, X_train, y_train, X_test, y_test, scoring, costs, experiment_name, artifact_location, uri, source_df, source_stockage):
        
        """
        Initialise une instance de la class MLFlowOptuna pour l'optimisation d'une pipeline ML avec Optuna et MLFlow.

        Args:
            pipeline (object): objet Pipeline de scikit-learn à optimiser
            name_pipeline (str): nom de la pipeline à optimiser
            param_grid (dict): dictionnaire contenant les grilles de recherche pour chaque hyperparamètre
            X_train (dataframe): dataframe des features pour l'ensemble d'entraînement
            y_train (dataframe): dataframe de la variable cible pour l'ensemble d'entraînement
            X_test (dataframe): dataframe des features pour l'ensemble de validation
            y_test (dataframe): dataframe de la variable cible pour l'ensemble de validation
            scoring (myscorer): métrique personnalisée à optimiser
            costs (fonction): fonction personnalisée 
            experiment_name (str): nom de l'expérience MLflow
            artifact_location (str): chemin relatif du répertoire de stockage des artefacts
            uri (str): URI du serveur de suivi MLflow
            source_df (str): nom de la source des données d'entrée
            source_stockage (str): nom de la source de stockage des données

        """
        self.pipeline = pipeline
        self.name_pipeline = name_pipelines
        self.param_grid = param_grid
        self.X_train =  X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring = scoring 
        self.costs = costs
        self.artifact_location = artifact_location
        self.experiment_name = experiment_name
        self.uri = uri
        self.source_df = source_df
        self.source_stockage = source_stockage
        
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        
        self.mlflow_client = MlflowClient(tracking_uri=self.uri)
        if not self.mlflow_client.get_experiment_by_name(self.experiment_name):
            self.mlflow_client.create_experiment(self.experiment_name, artifact_location=self.artifact_location)
    
   
    
    def objective(self,trial):
        
        """Fonction qui exécute un entrainement sur un pipeline en utilisant une combinaison d'hyperparamètres proposée par Optuna.
    
        Args:
            self (object): L'objet qui contient le pipeline, les données d'entraînement, le scoring, le param_grid et les informations de logging pour MLflow.
            trial (optuna.trial.Trial): L'objet trial d'Optuna qui propose les hyperparamètres à essayer pour l'entraînement.
        
        Returns:
            float: La moyenne des scores de validation croisée pour la combinaison d'hyperparamètres proposée.
        """
        with mlflow.start_run(run_name="Pipeline LGBM Optimisation Optuna 1"):
            params = {}
            for param_name, param_values in self.param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1], step=param_values[2])
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], step=param_values[2])
                elif isinstance(param_values[0], str):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)

            self.pipeline.set_params(**params)

            score = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=5, scoring=self.scoring)
            myscorer_mean = score.mean()
        
            for key, value in params.items():
                mlflow.log_param(key, value)
        
            mlflow.log_metric("cost_function_métier", myscorer_mean)
            mlflow.set_tag("La source de stockage", self.source_stockage)
            mlflow.set_tag("df_source", self.source_df)
            mlflow.set_tag("LGBM", self.experiment_name)

    
            return score.mean()

   
 
    
    def run_study(self, n_trials):
        """Fonction qui exécute une étude Optuna pour optimiser un pipeline et renvoie le meilleur pipeline entraîné.
    
        Args:
            self (object): L'objet qui contient le pipeline, les données d'entraînement, le scoring, le param_grid, le nom de l'expérience et les informations de logging pour MLflow.
            n_trials (int): Le nombre d'essais d'hyperparamètres proposés pour l'étude.
        
        Returns:
            object: La meilleur pipeline entraîné avec les hyperparamètres optimaux proposés par Optuna.
        """
        
        mlflc = MLflowCallback(tracking_uri=self.uri, metric_name="cost_function_métier")

        study = optuna.create_study(direction="maximize", study_name=self.experiment_name)
        study.optimize(self.objective, n_trials= n_trials, show_progress_bar = True)
        #return study.best_trial.params
        best_pipe_optuna = self.pipeline.set_params(**study.best_trial.params)

        print(study.best_trial)
        
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        
        print('\tValue: {}'.format(trial.value))
        print('\tParams: ')
        for key, value in trial.params.items():
            print('\t\t{}: {}'.format(key, value))
        
        return best_pipe_optuna, study
        
    def plot_params(self, study, params_names):
        """
        Visualisez l'espace de recherche des hyperparamètres et leur importance en fonction de l'étude d'optimisation.
    
        args:
       
        - study : `optuna.study.Study`
            Un objet d'étude Optuna qui contient les résultats d'optimisation des hyperparamètres.
        - params_names (Liste[str]) : Une liste de noms des hyperparamètres à visualiser.
        
        Returns:
        --------
        plot optuna params
        """
        # Créer la figure pour la visualisation
        fig = optuna.visualization.plot_parallel_coordinate(study, params=params_names)
        fig.show()
        
        
        # Visualiser les hyperparamétres les plus important
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()

    
    def plot_learning_curve(self,best_pipe_optuna):
        """
        Trace une courbe d'apprentissage pour le meilleur modèle donné, en utilisant la méthode learning_curve() de scikit-learn.
    
        Args:
        - self (object): Instance de la classe  MLFlowOptunba contenant les données et les paramètres.
        - best_pipe_optuna (object): Meilleur pipeline pour lequel la courbe d'apprentissage sera tracée.
        
        Returns:
        - plot learning curve 
        """
        
        train_sizes, train_scores, test_scores = learning_curve(estimator=best_pipe_optuna, X=self.X_train, y=self.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring=self.scoring)
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
        
    def eval_plot_curve(self, best_pipe_optuna):
        
        """
        Évalue et trace la courbe ROC et la courbe précision-rappel pour le meilleur modèle donné.
    
        Args:
        - self (object): Instance de la classe MLFlowOptuna contenant les données et les paramètres.
        - best_pipe_optuna (object): Meilleur pipeline pour lequel la courbe ROC et la courbe précision-rappel seront tracées.
    
        Returns:
        - plot courbe ROC et la courbe précision-rappel
        """
        
        best_pipe_optuna.fit(self.X_train, self.y_train)
        y_prob = best_pipe_optuna.predict_proba(self.X_test)[:,1]
        y_pred = best_pipe_optuna.predict(self.X_test)
      
        fpr, tpr, _ = metrics.roc_curve(self.y_test, y_prob)
    
        auc = metrics.roc_auc_score(self.y_test, y_prob)
      
        #create ROC curve
        plt.figure(figsize=(12,6))
        plt.subplot(1, 2, 1)
        plt.title('ROC Curve', fontsize=18)
        plt.plot(fpr,tpr,label="AUC="+str(np.round(auc,4)))
        plt.plot(np.linspace(0,1),np.linspace(0,1),linestyle='--',color='blue',label=self.name_pipeline)
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
        
    def eval_best_threshold(self, best_pipe_optuna):
        
        """ Évalue la meilleure valeur seuil pour un modèle donné.

        Args:
            self (object): Instance de la classe MLFlowOptuna 
            best_pipe_optuna (object): Meilleur pipeline sélectionné lors de l'optimisation.
    
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
            y_pred = (best_pipe_optuna.predict_proba(self.X_test)[:, 1] >= threshold).astype(int)
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
          
    
    def eval_confusion_matrix_threshold(self, best_pipe_optuna, threshold):
        
        """ Évalue la matrice de confusion pour un modèle et un seuil donnés.

        Args:
            self (object): Instance de la class MLFlowOptuna
            best_pipe_optuna (object): Meilleur pipeline sélectionné lors de l'optimisation.
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
        
        y_prob = best_pipe_optuna.predict_proba(self.X_test)
        y_pred = best_pipe_optuna.predict(self.X_test)
        
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

                