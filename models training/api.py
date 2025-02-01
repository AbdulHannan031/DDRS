from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import pickle
import numpy as np

model = joblib.load("models training/savedmodels/generalmodel/best_model.pkl")
scaler = joblib.load("models training/savedmodels/generalmodel/scaler.pkl")
heart_disease_model = pickle.load(open('models training/savedmodels/heartdisease.pkl', 'rb'))
with open('models training/savedmodels/diabetes_model.sav', 'rb') as file:
    data = pickle.load(file)
    diabetes_model = data['model']
    diabetes_scaler = data['scaler']
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_disease_with_saved_model():
    try:
        data = request.json
        temp_f = float(data['temperature'])
        pulse_rate_bpm = float(data['pulse_rate'])
        vomiting = int(data['vomiting'])
        yellowish_urine = int(data['yellowish_urine'])
        indigestion = int(data['indigestion'])

        user_input = pd.DataFrame({
            'Temp': [temp_f],
            'Pulserate': [pulse_rate_bpm],
            'Vomiting': [vomiting],
            'YellowishUrine': [yellowish_urine],
            'Indigestion': [indigestion]
        })

        user_input_scaled = scaler.transform(user_input)

        predicted_disease = model.predict(user_input_scaled)[0]
        disease_names = {
            0: 'Heart Disease',
            1: 'Viral Fever/Cold',
            2: 'Jaundice',
            3: 'Food Poisoning',
            4: 'Normal'
        }

        return jsonify({
            'prediction': disease_names[predicted_disease],
            'message': 'Disease prediction result'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/heart', methods=['POST'])
def predict_heart_disease():
    try:
        data = request.get_json()

        age = data['age']
        sex = data['sex']
        cp = data['chestPain']
        trestbps = data['bloodPressure']
        chol = data['cholesterol']
        fbs = data['bloodSugar']
        restecg = data['restingEcg']
        thalach = data['maxHeartRate']
        exang = data['exerciseAngina']
        oldpeak = data['stDepression']
        slope = data['stSlope']
        ca = data['fluoroscopy']
        thal = data['nuclearStressTest']

        sex_mapping = {'Male': 1, 'Female': 0}
        sex_numeric = sex_mapping.get(sex, 0) 

        input_data = [float(age), float(sex_numeric), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                      float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
        input_data = np.array(input_data).reshape(1, -1)

        prediction = heart_disease_model.predict(input_data)

        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        return jsonify({
            'prediction': result,
            'message': 'Heart disease prediction result'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()

        preg = int(data['pregnancies'])
        glucose = int(data['glucoseLevel'])
        bp = int(data['bloodPressure'])
        skin = int(data['skinThickness'])
        insulin = int(data['insulinLevel'])
        bmi = float(data['bmi'])
        dpf = float(data['pedigreeFunction'])
        age = int(data['age'])

        input_data = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)

        if not hasattr(diabetes_scaler, 'transform'):
            return jsonify({'error': 'Scaler not loaded correctly'}), 500

        input_data_scaled = diabetes_scaler.transform(input_data)

        prediction = diabetes_model.predict(input_data_scaled)

        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        return jsonify({
            'prediction': result,
            'message': 'Diabetes prediction result'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3001)
