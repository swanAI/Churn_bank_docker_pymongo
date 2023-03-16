import pandas as pd
from matplotlib import pyplot as plt
from termcolor import colored


class DataExplorerFond:
    def __init__(self, df):
        self.df = df
        
    """
    Initialise une instance de DataExplorerFond avec un DataFrame Pandas en entrée.

    Args: data (pd.DataFrame): Les données à explorer.
    """
        
    def doublons(self):
        """
        Afficher les informations indiquant les lignes en double.
        
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
            
        Returns:
        -----------------
        Les doublons du dataframe
        """ 
        
        print("Les doublons dans trainset:", len(self.df[self.df.duplicated()]))
        
    def unique_multi_cols(self):
        
        """
        Afficher les informations concernant les valeurs uniques pour chaque colonnes
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
            
        Returns:
        -----------------
        Les valeurs uniques pour chaque colonnes
        """ 
        
        for col in list(self.df.columns):
            
            pct_nan = (self.df[col].isna().sum()/self.df[col].shape[0])
            unique = self.df[col].unique()
            nunique = self.df[col].nunique()
          
            print('')
            print(colored(col, 'red'))
            print('') 
            print((f'Le pourcentage NaN : {pct_nan*100}%'))
            print(f'Nombre de valeurs unique : {nunique}')
            print('')
            print(unique)
            print('')
            print('---------------------------------------------------------------------------------------')
     
    
    # Plot NaN pourcentage
    def plot_pourcentage_NaN_features(self):
        
        """
        Afficher les informartions concernant les valeurs manquantes des rows
        Parameters:
        -----------------
        self (DataFrame) : Dataframe analysis
            
        Returns:
        -----------------
        tabulates les valeurs manquantes des rows
        """
        
        plt.figure(figsize=(20,18))
        plt.title('Le pourcentage de valeurs manquantes pour les features', size=20)
        plt.plot((self.df.isna().sum()/self.df.shape[0]*100).sort_values(ascending=True))
        plt.xlabel('Features dataset', fontsize=18)
        plt.ylabel('Pourcentage NaN dans features', fontsize=18)
        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
        plt.show()        
        pct_dataset = pd.DataFrame((self.df.isna().sum()/self.df.shape[0]*100).sort_values(ascending=False))
        pct_dataset = pct_dataset.rename(columns={0:'Pct_NaN_colonne'})
        pct_dataset = pct_dataset.style.background_gradient(cmap='YlOrRd')
        return