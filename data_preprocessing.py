"""
Data Preprocessing Module for DDoS Detection System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.attack_label_encoder = None
        self.feature_names = []

    def load_data(self, train_path, test_path=None):
        print("Loading dataset...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None
        return train_df, test_df

    def clean_data(self, df):
        print("Cleaning data...")
        df = df.drop_duplicates()

        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        print(f"Dataset shape: {df.shape}")
        return df

    def encode_features(self, df, fit=True):
        print("Encoding categorical features...")
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['label', 'attack_cat']]

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def prepare_labels(self, df):
        binary_labels = df['label'].values if 'label' in df.columns else None

        if 'attack_cat' in df.columns:
            if self.attack_label_encoder is None:
                self.attack_label_encoder = LabelEncoder()
                multiclass_labels = self.attack_label_encoder.fit_transform(df['attack_cat'].astype(str))
            else:
                multiclass_labels = self.attack_label_encoder.transform(df['attack_cat'].astype(str))
        else:
            multiclass_labels = None

        return binary_labels, multiclass_labels

    def select_features(self, df, drop_cols=['label', 'attack_cat', 'id']):
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop)
        self.feature_names = X.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        return X

    def normalize_features(self, X, fit=True):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

    def preprocess_pipeline(self, train_path, test_path=None, save_path='data/processed/'):
        os.makedirs(save_path, exist_ok=True)

        # Load and process training data
        train_df, test_df = self.load_data(train_path, test_path)
        train_df = self.clean_data(train_df)
        train_df = self.encode_features(train_df, fit=True)
        y_binary, y_multiclass = self.prepare_labels(train_df)
        X_train = self.select_features(train_df)
        X_train_scaled = self.normalize_features(X_train, fit=True)

        # Process test data
        if test_df is not None:
            test_df = self.clean_data(test_df)
            test_df = self.encode_features(test_df, fit=False)
            y_test_binary, y_test_multiclass = self.prepare_labels(test_df)
            X_test = self.select_features(test_df)
            X_test_scaled = self.normalize_features(X_test, fit=False)
        else:
            X_train_scaled, X_test_scaled, y_binary, y_test_binary, y_multiclass, y_test_multiclass = train_test_split(
                X_train_scaled, y_binary, y_multiclass,
                test_size=0.3, random_state=42, stratify=y_binary
            )

        # Save preprocessor objects
        joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_path, 'label_encoders.pkl'))
        joblib.dump(self.attack_label_encoder, os.path.join(save_path, 'attack_label_encoder.pkl'))
        joblib.dump(self.feature_names, os.path.join(save_path, 'feature_names.pkl'))

        print("\nPreprocessing completed!")
        print(f"Training samples: {X_train_scaled.shape[0]}")
        print(f"Testing samples: {X_test_scaled.shape[0]}")

        return (X_train_scaled, X_test_scaled, y_binary,
                y_test_binary, y_multiclass, y_test_multiclass)


# Example usage:
# if __name__ == "__main__":
#     preprocessor = DataPreprocessor()
#     preprocessor.preprocess_pipeline('data/train.csv', 'data/test.csv')
