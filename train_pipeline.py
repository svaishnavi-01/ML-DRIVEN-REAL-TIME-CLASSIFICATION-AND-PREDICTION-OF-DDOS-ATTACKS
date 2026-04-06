from data_preprocessing import DataPreprocessor
from model_training import DDoSModelTrainer
from model_evaluation import ModelEvaluator
import joblib


def main():
    print("=" * 80)
    print("DDoS DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 80)

    # Step 1: Data Preprocessing
    print("\n[1/5] Starting data preprocessing...")
    preprocessor = DataPreprocessor()

    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = preprocessor.preprocess_pipeline(
        train_path="C:/Users/DELL/OneDrive/Desktop/major_proj_19/Training and Testing Sets/UNSW_NB15_training-set.csv",
        test_path= "C:/Users/DELL/OneDrive/Desktop/major_proj_19/Training and Testing Sets/UNSW_NB15_testing-set.csv",
        save_path="C:/Users/DELL/OneDrive/Desktop/major_proj_19/data/processed"
    )

    # Step 2: Train Binary Classifier
    print("\n[2/5] Training binary classifier...")
    trainer = DDoSModelTrainer()
    binary_model = trainer.train_binary_classifier(
        X_train, y_train_bin, algorithm='random_forest'
    )

    # Step 3: Train Multi-class Classifier
    print("\n[3/5] Training multi-class classifier...")
    multiclass_model = trainer.train_multiclass_classifier(
        X_train, y_train_multi, algorithm='xgboost'
    )

    # Step 4: Evaluate Models
    print("\n[4/5] Evaluating models...")
    evaluator = ModelEvaluator()

    binary_metrics = evaluator.evaluate_binary_model(
        binary_model, X_test, y_test_bin
    )
    multiclass_metrics = evaluator.evaluate_multiclass_model(
        multiclass_model, X_test, y_test_multi,
        preprocessor.attack_label_encoder
    )

    # Step 5: Save Models
    print("\n[5/5] Saving models...")
    trainer.save_models('data/models/')

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Binary Classification Accuracy: {binary_metrics['accuracy'] * 100:.2f}%")
    print(f"Multi-class Classification Accuracy: {multiclass_metrics['accuracy'] * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
