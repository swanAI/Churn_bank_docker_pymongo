import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class UnivariateDataExplorer:
    
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise une instance de UnivariateDataExplorer avec un DataFrame Pandas en entrée.

        Args:
            data (pd.DataFrame): Les données à explorer.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Les données doivent être un DataFrame Pandas")

        self.df = df
        
    
        
    
    def categ_pie_plot(self, col: str):
        
        """
        Afficher Pie plot  
    
        Parameters:
        -----------------
        df (Dataframe) : dataframe analysis 
        col (object): 'col' de type object   
        Returns:
        -----------------
        plot pie
        """
        plt.figure(figsize=(12,8))
        
        self.df[col].value_counts(normalize=True).head(8).plot(kind='pie',subplots=True, shadow = True,autopct='%1.1f%%',
                                    textprops={'fontsize': 17} )
        plt.title(f'Répartition des {col} les plus représentées ', size=25)
        plt.legend(fontsize=15)
        plt.ylabel('', fontsize=17)
        plt.xlabel(f'{col}', fontsize=17)
        plt.xticks(fontsize=17)
        plt.show()
    
    def categ_bar_plot(self, col: str):
        
        """
        Afficher barplot  
    
        Parameters:
        -----------------
        self (Dataframe) : dataframe analysis 
        col (object): 'col' de type object   
        Returns:
        -----------------
        plot barplot
        """
        
        
        fig, ax1 = plt.subplots(figsize=(22, 12))
        plot = self.df[col].value_counts().plot( kind='bar', color='darkred')
        for p in plot.patches:
            plot.annotate(format(p.get_height(), ".3f"), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="center", xytext=(0, 8), textcoords="offset points")
        plt.title(f'Les {col} les plus représentés', size=25)
        plt.ylabel('Frequency', fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.xlabel(f'{col}', fontsize=17)
        plt.legend(fontsize=15)
        plt.show()
        sns.despine(fig)
        print('')
        print('')
        

        
    
    def num_hist_plot(self, col: str ):
        
        """
        Afficher histogramme 
    
        Parameters:
        -----------------
        self (Dataframe) : dataframe analysis 
        col (float, int): 'col' de type float ou int
        Returns:
        -----------------
        plot histogramme
        """
        
        sns.set_style('darkgrid')
        sns.set_color_codes(palette='dark')
        plt.figure(figsize=(12,8))
        sns.histplot(self.df[col],color='red',kde=True)
        plt.title(f'Histogramme de {col}', size=25)
        plt.ylabel('Frequency',fontsize=17)
        plt.xlabel(f'{col}',fontsize=17)
        plt.show()
        print('')
        print('')
        
    def __repr__(self):
        
        return str(self.__dict__)

