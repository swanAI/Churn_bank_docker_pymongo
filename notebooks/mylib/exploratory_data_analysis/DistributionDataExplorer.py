
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest


class DistributionDataExplorer:
    
    
    def __init__(self, data: pd.DataFrame):
        self.df = data 
    """
    Initialise une instance de DistributionDataExplore avec un DataFrame Pandas en entrée.

    Args: data (pd.DataFrame): Les données à explorer.
    """
        
    
    def Diagnostic_plot(self, variable: str):
        
        """
        Afficher diagnostic_plots afin de vérifier diagnostic des distributions 
        voir les outliers et normalités
    
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
        variable (series) : Colonne du dataframe
            
        Returns:
        -----------------
        plot diagnostic et test normalité 
        """
        
        plt.figure(figsize=(16, 4))

        # histogram
        plt.subplot(1, 3, 1)
        sns.histplot(self.df[variable], bins=30)
        plt.title('Histogram')
    
        # Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(self.df[variable], dist="norm", plot=plt)
        plt.ylabel('Variable quantiles')
    
        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=self.df[variable])
        plt.title('Boxplot')
    
        print('Test Shapiro')
        data = self.df[variable].values
        stat, p = shapiro(data)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
             print('Probablement Gaussien ')
        else:
             print('Probablement pas  Gaussien ')
                
        print('Test normaltest')        
        data = self.df[variable].values
        stat, p = normaltest(data)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probablement Gaussien ')
        else:
            print('Probablement pas  Gaussien ')


        plt.show()

    def __repr__(self):
        """
        Renvoie une représentation sous forme de chaîne des variables d'instance de l'objet.

        Retour:
        str : représentation sous forme de chaîne des variables d'instance de l'objet.
        """
        
        return str(self.__dict__)
