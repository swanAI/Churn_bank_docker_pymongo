import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class BivariateDataExplorer:
    
    
    def __init__(self, data: pd.DataFrame):
        self.df = data
    
    """
    Initialise une instance de BivariateDataExplorer avec un DataFrame Pandas en entrée.

    Args: data (pd.DataFrame): Les données à explorer.
    """
        
    def numVSnum_heatmap_corr(self):
        
        plt.figure(figsize=(16,14))
        sns.heatmap(self.df.corr(),annot=True , cmap='GnBu')
        plt.title('Matrice corrélation')
        
    
    # fonction afficher relation entre variable numérique et object
    def categVSnum_boxplot(self, var_cat: str, var_num: str):
        
        """
        Afficher boxplot entre une variable catégorielle et variable numérique
    
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
        var_cat (object) : 'col' de type object 
        var_num (float, int) : 'col' de type float, int    
        Returns:
        -----------------
        plot boxplot
        """
        
        meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}
        plt.figure(figsize=(16,12))
        plt.title(f"Boxplot entre {var_cat} et {var_num}", size=22)
        sns.boxplot(x=self.df[var_cat], y=self.df[var_num], showmeans=True, meanprops=meanprops,data=self.df)
        plt.xticks(fontsize=9)
        
    
    def categVScateg_countplot(self, var_cat1: str, var_cat_2: str):
        
        """
        Afficher countplot entre une variable catégorielle et variable catégorielle qui prend le nom de la colonne pour l’encodage des couleurs (hue)
    
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
        var_cat1 (object) : 'col' de type object 
        var_cat_2 (object) : 'col' de type object à mettre dans paramètre hue    
        Returns:
        -----------------
        plot countplot
        """
        
        
        #plt.figure(figsize=(16,14))
        fig, ax1 = plt.subplots(figsize=(16, 12))

        plt.title(f'Countplot entre {var_cat1} et {var_cat_2}',size=22)
        plot = sns.countplot(x=var_cat1,hue=var_cat_2, data=self.df)
        for p in plot.patches:
            plot.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points")
        plt.title(f'Countplot entre {var_cat1} et {var_cat_2}',size=22)
        plt.tight_layout()
        plt.legend(loc='best')
        plt.xticks(fontsize=9)
        plt.show()
        sns.despine(fig)
        
    def __repr__(self):
        return str(self.__dict__)