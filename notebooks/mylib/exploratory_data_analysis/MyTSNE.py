import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union


class MyTSNE:
    """
    Classe qui implémente l'algorithme t-SNE pour la réduction de la dimensionnalité.
    
    Args:
    ------
        data: pandas.DataFrame
            Les données à transformer.
        n_components: int, optional(default=2)
            Le nombre de dimensions de l'espace réduit.
        perplexity: int, optional(default=30)
            Le nombre de voisins considérés pour chaque point dans l'espace original.
        learning_rate: int, optional(default=200)
            Le taux d'apprentissage de l'algorithme.
        random_state: int, optional(default=None)
            La seed pour la génération de nombres pseudo-aléatoires.
    
    Attributes:
    -----------
        data: pandas.DataFrame
            Les données à transformer.
        names: pandas.Index
            Les noms des points de données.
        n_components: int
            Le nombre de dimensions de l'espace réduit.
        perplexity: int
            Le nombre de voisins considérés pour chaque point dans l'espace original.
        learning_rate: int
            Le taux d'apprentissage de l'algorithme.
        random_state: int
            La seed pour la génération de nombres pseudo-aléatoires.
        features_extration: Union[pd.DataFrame, None]
            Les nouvelles features extraites.
    
    Methods:
    --------
        fit_transform(self) -> pd.DataFrame:
            Calcule les nouvelles features et les retourne dans un DataFrame pandas.
        
        Plot(self) -> plt.figure:
            Trace un graphe en deux dimensions des données réduites par t-SNE.
    """
    def __init__(self, data: pd.DataFrame, n_components: int = 2, perplexity: int = 30, learning_rate: int = 200, random_state: Optional[int] = None) -> None:
        self.names = data.index
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.features_extration = None
        
    def fit_transform(self) -> pd.DataFrame:
        """
        Calcule les nouvelles features et les retourne dans un DataFrame pandas.
        
        Returns:
        --------
            pd.DataFrame:
                Les nouvelles features extraites.
        """
        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.data)
        
        # Calcul des composantes principales
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, learning_rate=self.learning_rate, random_state=self.random_state)
        self.features_extration = tsne.fit_transform(X_train_scaled)
        return self.features_extration
      
    def Plot(self) -> plt.figure:
        """
        Trace un graphe en deux dimensions des données réduites par t-SNE.
        
        Returns:
        --------
            plt.figure:
                La figure du graphe.
        """
        if self.features_extration is None:
            self.fit_transform()
        df_tsne = pd.DataFrame(self.features_extration, columns=[f"TSNE_{i}" for i in range(1, self.n_components+1)])
        plt.figure(figsize=(12,8))
        sns.scatterplot(
            x="TSNE_1", y="TSNE_2", hue=self.names, data=df_tsne).set_title('TSNE selon le churn', fontsize = 30)
        plt.xlabel('tsne1', fontsize = 26)
        plt.ylabel('tsne2', fontsize = 26)
        plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)) 
        plt.show()
        return plt
    
    
    