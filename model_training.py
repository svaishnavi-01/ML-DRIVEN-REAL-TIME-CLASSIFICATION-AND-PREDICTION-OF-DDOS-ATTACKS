"""
Model Training Module for DDoS Detection
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os


class DDoSModelTrainer:
    def __init__(self):
        self.binary_classifier = None
        self.multiclass_classifier = None

    def train_binary_classifier(self, X_train, y_train, algorithm='random_forest'):
        print(f"\nTraining Binary Classifier ({algorithm})...")

        if algorithm == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

        elif algorithm == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        elif algorithm == 'svm':
            model = SVC(kernel='rbf', probability=True, random_state=42)

        elif algorithm == 'naive_bayes':
            model = GaussianNB()

        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=20, random_state=42)

        elif algorithm == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model.fit(X_train, y_train)
        self.binary_classifier = model
        print("✅ Binary classifier training completed!")
        return model

    def train_multiclass_classifier(self, X_train, y_train, algorithm='xgboost'):
        print(f"\nTraining Multi-class Classifier ({algorithm})...")

        if algorithm == 'xgboost':
            model = XGBClassifier(
                n_estimators=150,
                max_depth=15,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                objective='multi:softmax',
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

        elif algorithm == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )

        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=25, random_state=42)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model.fit(X_train, y_train)
        self.multiclass_classifier = model
        print("✅ Multi-class classifier training completed!")
        return model

    def save_models(self, save_path='data/models/'):
        os.makedirs(save_path, exist_ok=True)

        if self.binary_classifier:
            joblib.dump(self.binary_classifier, os.path.join(save_path, 'binary_classifier.pkl'))
            print(f"💾 Binary classifier saved to {save_path}")

        if self.multiclass_classifier:
            joblib.dump(self.multiclass_classifier, os.path.join(save_path, 'multiclass_classifier.pkl'))
            print(f"💾 Multi-class classifier saved to {save_path}")

    def load_models(self, binary_path, multiclass_path):
        self.binary_classifier = joblib.load(binary_path)
        self.multiclass_classifier = joblib.load(multiclass_path)
        print("✅ Models loaded successfully!")


# Example usage:
# if __name__ == "__main__":
#     trainer = DDoSModelTrainer()
#     trainer.train_binary_classifier(X_train, y_train, algorithm='random_forest')
#     trainer.train_multiclass_classifier(X_train, y_multiclass, algorithm='xgboost')
#     trainer.save_models()
