import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MyPCA:
    def __init__(self, data_pca: pd.DataFrame, n_components: int) -> None:
        
        """
        Initialise une instance de MyPCA.

        Args:
            data_pca (pd.DataFrame): Le DataFrame contenant les données.
            n_components (int): Le nombre de composantes principales à extraire.

        Returns:
            None
        """
        
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.data_pca = data_pca
        self.X = data_pca.values
        self.names = data_pca.index
        self.features = data_pca.columns
        self.features_extration = None
        
    def fit_transform(self):
        """
        Standardise les données et calcule les composantes principales.

        Args:
            None

        Returns:
            np.ndarray: Le tableau numpy contenant les composantes principales.
        """
        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X)
        
        # Calcul des composantes principales
        self.features_extration = self.pca.fit_transform(X_train_scaled)
        return self.features_extration
        
    def plot(self):
        """
        Trace le graphique de l'eboulis des valeurs propres, le cercle des corrélations et la projection des individus sur le plan factoriel.

        Args:
            None

        Returns:
            None
        """
        if self.features_extration is None:
            self.fit_transform()
        
        # Eboulis des valeurs propres
        plt.figure(figsize=(10,8))
        scree = self.pca.explained_variance_ratio_*100
        plt.bar(np.arange(len(scree))+1, scree)
        plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
        plt.xlabel("rang de l'axe d'inertie")
        plt.ylabel("pourcentage d'inertie")
        plt.title("Eboulis des valeurs propres")
        plt.show(block=False)
        
        
        # Cercle des corrélations
        fig = plt.figure(figsize=(10,10))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        
        for i, feature in enumerate(self.features):
            x = self.pca.components_[0,i]
            y = self.pca.components_[1,i]
            plt.plot([0,x], [0,y], color='k')
            plt.text(x, y, feature, fontsize='12')
    
            circle = Circle((0,0), radius=1, facecolor='none', edgecolor='b')
            plt.gca().add_patch(circle)
            plt.title('Cercle des corrélations')
            plt.xlabel('PC1 ({:.1f}%)'.format(scree[0]))
            plt.ylabel('PC2 ({:.1f}%)'.format(scree[1]))
        plt.show()
        
        
        # Plan factoriel
        df_proj = pd.DataFrame(self.features_extration[:,:2], columns=['PC1', 'PC2'], index=self.names)
        sns.scatterplot(data=df_proj, x='PC1', y='PC2', hue=self.names, palette='deep')
        plt.title('Projection des individus sur le plan factoriel')
        plt.show()

