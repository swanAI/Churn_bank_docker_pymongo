import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('ggplot')
from termcolor import colored
from IPython.core.display import display
from sklearn import set_config
set_config(display='diagram')
from sklearn.model_selection import  StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve



class PipelineBaselineOneMetric:
    
    def __init__(self, X_train, y_train, metric, pipelines):
        
        self.X_train = X_train
        self.y_train = y_train
        self.pipelines = pipelines
        self.metric = metric
        self.stratified_Kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    
    
    
    def learning_curve(self):
        for pipe_name, pipeline in self.pipelines.items():
            
            train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline, X=self.X_train, y=self.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring=self.metric)
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            plt.title(f"Learning curve pour {pipe_name}")
            plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training fontion cout métier')
            plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
            plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation fontion cout métier')
            plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
            plt.grid(True)
            plt.xlabel('Number of training samples')
            plt.ylabel('fonction_cout métier')
            plt.legend(loc='lower right')
            plt.show()
        
    def run_cross_validation(self):
        df_results_cv = pd.DataFrame()
        df_name_model = pd.DataFrame()
        for pipe_name, pipeline in self.pipelines.items():
            print(colored(pipe_name.upper(), 'red'))
            display(pipeline)
            print()
        
            #Cross validation multi metrics select model
            cv_results = cross_val_score(pipeline, self.X_train, self.y_train, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=77), scoring=self.metric)
            
            print(f"Scores cv pour {colored(pipe_name,'blue')}:", cv_results)
            print(f"Score cv mean pour {colored(pipe_name,'blue')}:", cv_results.mean())
            print(f"Score écart type cv pour {colored(pipe_name,'blue')}:", cv_results.std())
            print()
            
            df_results_cv = df_results_cv.append(pd.DataFrame(cv_results),ignore_index=True)
            df_name_model = df_name_model.append(pd.DataFrame([pipe_name]*5),ignore_index=True)
            df_results_cv_renamed = df_results_cv.add_suffix('_CV_score')
            df_name_model_renamed = df_name_model.add_suffix('_name_model')
            df_results_final = pd.concat([df_results_cv_renamed, df_name_model_renamed], axis=1)
            df_results_final = df_results_final.rename(columns={"0_CV_score": "Scores_CV", "0_name_model": "Name_model"})
            df_results_final_1 = pd.melt(df_results_final, id_vars='Name_model', var_name="Procédure", value_name="Scores_CV")
            results_CV_mean = df_results_final_1.groupby(by=['Name_model']).mean().reset_index() 
        return df_results_final_1, results_CV_mean
            
    
            
            
        
    def barplot_evaluate(self, results_CV_mean):
        fig, ax1 = plt.subplots(figsize=(22, 12))
        plot = sns.barplot(x="Name_model", y="Scores_CV", data=results_CV_mean, ax=ax1)
        for p in plot.patches:
            plot.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points")
        plt.title('Comparaison des pipelines baseline pour la fonction cout métier', size=26)
        plt.legend(bbox_to_anchor=(1.14, 1), borderaxespad=0, title="Fonction_cout_métier")
        plt.tight_layout()
        plt.xlabel('Les pipelines', fontsize=18)
        plt.ylabel("Fonction_cout_métier values mean (cv)", fontsize=18)
        plt.xticks(fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        sns.despine(fig)
        
        
   

        
    
    def boxplot_evaluate(self, df_results_final_1):
        
        meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}


        plt.figure(figsize=(16,12))
        plt.title('Comparaison des pipelines pour la fonction cout métier', size=25)
        sns.boxplot(x="Scores_CV", y= "Name_model", showmeans=True, meanprops=meanprops, data=df_results_final_1)
        plt.xlabel('Val_score', fontsize=15)
        plt.ylabel("Les bests modèles de la procédure d'évaluation", fontsize=15)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()
        print('')
            
    def evaluate_new_data(self, X_test, y_test):
        for pipe_name, pipeline in self.pipelines.items():
            
            print(colored("-" *120, 'green', attrs=['bold', 'underline']))
            print()
            print(colored(f"EVALUATION DE LA PIPELINE {pipe_name} BASELINE SUR LES DONNÉES DE TEST", "blue"))
            print()
            pipeline.fit(self.X_train, self.y_train)
            ypred = pipeline.predict(X_test)
        
            cf_matrix = confusion_matrix(y_test, ypred)
        
            plt.figure(figsize=(12, 8))
            ax = plt.axes()
            plt.title(f"Matrice de confusion pour {pipe_name}", size=20, y=1.1)
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
            print(classification_report(y_test, ypred))
            print()
            
    def __repr__(self):
        return str(self.__dict__)
           
        
           
           