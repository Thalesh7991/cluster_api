import pickle
import numpy as np
import pandas as pd
import json
import joblib

class CustomerClusteringModel:
    def __init__(self,scaler, kmeans, umap):
        self.scaler = scaler
        self.kmeans = kmeans
        self.umap = umap

    def preprocess_data(self, df2):
        """Pré-processa os dados usando o scaler carregado."""
        df2["Spent"] = df2["MntWines"]+ df2["MntFruits"]+ df2["MntMeatProducts"]+ df2["MntFishProducts"]+ df2["MntSweetProducts"]+ df2["MntGoldProds"]

        df2["Children"]=df2["Kidhome"]+df2["Teenhome"]
        df2["Is_Parent"] = np.where(df2.Children> 0, 1, 0)

        df2["Age"] = 2024-df2["Year_Birth"]

        for col in df2.columns:
            if col in self.scaler:  # Verifica se o scaler existe para a coluna
                df2[col] = self.scaler[col].transform(df2[[col]])
        
        df2 = df2.drop(columns=['ID','Education', 'Marital_Status', 'Dt_Customer', "Z_CostContact", "Z_Revenue"])
        df2 = df2[['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
                    'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                    'AcceptedCmp2', 'Complain', 'Response', 'Spent', 'Children',
                    'Is_Parent', 'Age']]

        return df2
    
    def reduce_dimensionality(self, data):
        """Reduz a dimensionalidade dos dados usando o modelo TSNE carregado."""
        data = self.umap.transform(data)
        data = pd.DataFrame(data, columns=['embedding_x', 'embedding_y'])
        return data
    
    def predict_cluster(self, data):
        """Pipeline completo para prever o cluster de novos dados."""
        # Passo 1: Pré-processamento dos dados
        data_scaled = self.preprocess_data(data)

        
        # Passo 2: Redução de dimensionalidade
        data_reduced = self.reduce_dimensionality(data_scaled)

        # Passo 3: Previsão de cluster
        clusters = self.kmeans.predict(data_reduced)

        return json.dumps(clusters.tolist())
