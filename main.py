from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from utils.preprocessor import preprocess_input  # This should exist

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Home page route (renders form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Input processing
        input_data = {
            'Substance': data['substance'],
            'Unit': data['unit'],
            'Supply Chain Emission Factors without Margins': float(data['supply_wo_margin']),
            'Margins of Supply Chain Emission Factors': float(data['margin']),
            'DQ ReliabilityScore of Factors without Margins': float(data['dq_reliability']),
            'DQ TemporalCorrelation of Factors without Margins': float(data['dq_temporal']),
            'DQ GeographicalCorrelation of Factors without Margins': float(data['dq_geo']),
            'DQ TechnologicalCorrelation of Factors without Margins': float(data['dq_tech']),
            'DQ DataCollection of Factors without Margins': float(data['dq_data']),
            'Source': data['source']
        }

        df = pd.DataFrame([input_data])
        df_processed = preprocess_input(df)
        df_scaled = scaler.transform(df_processed)
        prediction = model.predict(df_scaled)

        return f"<h2>Predicted Emission Factor: {prediction[0]:.4f}</h2>"

    except Exception as e:
        return f"<h2>Error occurred: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
