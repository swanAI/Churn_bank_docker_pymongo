import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from termcolor import colored
from IPython.core.display import display
from sklearn import set_config
set_config(display='diagram')
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate



class PipelineBaselineMultiMetrics:
    def __init__(self, X_train, y_train, myscorer, pipelines):
        
        self.X_train = X_train
        self.y_train = y_train
        self.pipelines = pipelines
        self.myscorer = myscorer
        self.scoring = {"fonction_cout_métier": self.myscorer, "accuracy": "accuracy", "f1": "f1",
                        "roc_auc": "roc_auc", "recall": "recall", "precision": "precision"}
        
        self.stratified_Kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
     
    def display(self):
        for pipe_name, pipeline in self.pipelines.items():
            print(colored(f"Classifier {pipe_name}:", 'red'))
            display(pipeline)
    
    def Run_cross_validate(self):
        df_results_cv = pd.DataFrame()
        df_name_model = pd.DataFrame()
        
        for pipe_name, pipeline in self.pipelines.items():
            
            print(colored(pipe_name, 'blue'))
            #Cross validation multi metrics select model
            cv_results = cross_validate(pipeline, self.X_train, self.y_train, 
                          cv=self.stratified_Kfold, scoring=self.scoring)
            # k le nombre de folds 
            k=5 
            #print(colored('---------------------------------------------------------------------------------------------', 'red'))
            # cv_results est un dictionnaire 
            # Création dataframe des résultats de la cross validation multi metrics
            df_results_cv = df_results_cv.append(pd.DataFrame(cv_results), ignore_index=True)
            df_name_model = df_name_model.append(pd.DataFrame([pipe_name]*k),ignore_index=True)
            df_results_final = pd.concat([df_results_cv,df_name_model],axis=1)
            df_results_final_1 = df_results_final.rename(columns={0: "Models"})
            df_results_final_2 = pd.melt(df_results_final_1, id_vars='Models', var_name="Metrics", value_name="mean_val_score")
            results_CV_mean = df_results_final_2.groupby(by=['Models', 'Metrics']).mean().reset_index()
            
        return results_CV_mean, df_results_final_1
    
    
    def barplot_evaluate(self, results_CV_mean):
        # Comparaison des modèles pour chaque métrique avec energie score star
        fig, ax1 = plt.subplots(figsize=(22, 12))
        plot = sns.barplot(x="Models", y="mean_val_score", hue="Metrics", data=results_CV_mean, ax=ax1)
        for p in plot.patches:
            plot.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points")
        plt.title('Comparaison des pipelines baseline multi-metrics', size=26)
        plt.tight_layout()
        plt.xlabel('Les pipelines', fontsize=18)
        plt.ylabel("Metrics values mean (cv)", fontsize=18)
        plt.xticks(fontsize=15, rotation=90)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        sns.despine(fig)
        
    def __repr__(self):
        return str(self.__dict__)