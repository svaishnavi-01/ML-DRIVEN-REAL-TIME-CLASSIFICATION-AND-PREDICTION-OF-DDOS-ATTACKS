"""
Real-time Prediction Module
"""
import joblib
import numpy as np
import pandas as pd
import os



class DDoSPredictor:
    def __init__(self, model_path='C:\\Users\\DELL\\OneDrive\\Desktop\\major_proj_19\\data\\models\\', preprocessor_path='C:\\Users\\DELL\\OneDrive\\Desktop\\major_proj_19\\data\\processed\\'):
        # Load models
        # joblib.load(os.path.join(model_path, 'binary_classifier_rf.pkl'))
        # joblib.load(os.path.join(model_path, 'multiclass_classifier_xgb.pkl'))
   
        self.binary_model = joblib.load(os.path.join(model_path, 'binary_classifier_rf.pkl'))
        self.multiclass_model = joblib.load(os.path.join(model_path, 'multiclass_classifier_xgb.pkl'))
        # Load preprocessors
        # joblib.load(os.path.join(preprocessor_path, 'scaler.pkl'))
        # joblib.load(os.path.join(preprocessor_path, 'label_encoders.pkl'))
        # joblib.load(os.path.join(preprocessor_path, 'attack_label_encoder.pkl'))
        # joblib.load(os.path.join(preprocessor_path, 'feature_names.pkl'))
        self.scaler = joblib.load(os.path.join(preprocessor_path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(preprocessor_path, 'label_encoders.pkl'))
        self.attack_label_encoder = joblib.load(os.path.join(preprocessor_path, 'attack_label_encoder.pkl'))
        self.feature_names = joblib.load(os.path.join(preprocessor_path, 'feature_names.pkl'))


        print("✅ Predictor initialized successfully!")

    def preprocess_input(self, data_dict):
        """
        Preprocess single traffic record for prediction
        """
        df = pd.DataFrame([data_dict])

        # Encode categorical features
        for col in self.label_encoders:
            if col in df.columns:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0

        # Select only trained features
        df = df[self.feature_names]

        # Scale features
        scaled_data = self.scaler.transform(df)
        return scaled_data

    def predict(self, data_dict):
        """
        Make prediction on network traffic

        Returns:
        --------
        result : dict
            Contains is_attack, attack_type, confidence
        """
        # Preprocess input
        X = self.preprocess_input(data_dict)

        # Binary prediction (Attack or Normal)
        is_attack = self.binary_model.predict(X)[0]
        binary_confidence = self.binary_model.predict_proba(X)[0]

        result = {
            'is_attack': bool(is_attack),
            'detection_confidence': float(binary_confidence[int(is_attack)]),
            'attack_type': 'Normal',
            'attack_confidence': 0.0
        }

        # If attack detected, classify attack type
        if is_attack:
            attack_type_encoded = self.multiclass_model.predict(X)[0]
            attack_proba = self.multiclass_model.predict_proba(X)[0]

            attack_type = self.attack_label_encoder.inverse_transform(
                [attack_type_encoded]
            )[0]

            result['attack_type'] = attack_type
            result['attack_confidence'] = float(attack_proba[attack_type_encoded])

        return result

    def predict_batch(self, data_list):
        """
        Predict on multiple traffic records
        """
        results = []
        for data_dict in data_list:
            result = self.predict(data_dict)
            results.append(result)
        return results
