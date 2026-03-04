from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model saved from your notebook
model_path = os.path.join(os.path.dirname(__file__), '../model/rf_churn_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Churn Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Convert input JSON to DataFrame to match model training format
    df = pd.DataFrame([data])
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    
    return jsonify({
        'churn_prediction': int(prediction[0]),
        'churn_probability': float(probability[0])
    })

if __name__ == '__main__':
    app.run()
