import pickle
import json

class CustomerClusteringModel:
    def __init__(self):
        self.kmeans = pickle.load(open('api/model/kmeans_model.pkl','rb'))
        
    
    def predict_cluster(self, data):
        """Pipeline completo para prever o cluster de novos dados."""

        # Passo 3: Previsão de cluster
        clusters = self.kmeans.predict(data)
        
        return json.dumps(clusters.tolist())
