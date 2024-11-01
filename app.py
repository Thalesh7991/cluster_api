from flask import Flask, request, Response
import os
import pandas as pd
import json
import pickle
import joblib

from empresa.empresa import CustomerClusteringModel


with open('scaler/scalers.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('model/kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)
umap = joblib.load('reducer/umap_reducer.pkl')

app = Flask(__name__)

@app.route('/empresa/predict', methods=['POST'])
def emprestimo_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        pipeline = CustomerClusteringModel(scaler, kmeans, umap)

        df_predict = pipeline.predict_cluster(test_raw)

        return df_predict
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
