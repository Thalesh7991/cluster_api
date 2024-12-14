from flask import Flask, request, Response
import os
import pandas as pd

from empresa.empresa import CustomerClusteringModel


app = Flask(__name__)

@app.route('/empresa/predict', methods=['POST'])
def emprestimo_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        pipeline = CustomerClusteringModel()

        test_raw['embedding_x'] = test_raw['embedding_x'].astype('float64')
        test_raw['embedding_y'] = test_raw['embedding_y'].astype('float64')
        

        df_predict = pipeline.predict_cluster(test_raw)

        return df_predict
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
