from prediction import DDoSPredictor
from severity_scoring import SeverityScorer
import pandas as pd


def test_predictions():
    # Initialize predictor and scorer
    predictor = DDoSPredictor()
    scorer = SeverityScorer()

    # Load a sample from test set
    test_data = pd.read_csv('data/raw/UNSW_NB15_testing-set.csv')

    print("\nTesting predictions on sample traffic...")
    print("=" * 80)

    # Test on first 5 samples
    for i in range(5):
        row = test_data.iloc[i]
        data_dict = row.to_dict()

        # Make prediction
        result = predictor.predict(data_dict)

        # Calculate severity if attack detected
        if result['is_attack']:
            score, severity = scorer.score_from_dataframe_row(
                row, result['attack_type']
            )
            result['severity_score'] = score
            result['severity_level'] = severity

        # Display result
        print(f"\nSample {i + 1}:")
        print(f" Source IP: {row.get('srcip', 'N/A')}")
        print(f" Destination IP: {row.get('dstip', 'N/A')}")
        print(f" Is Attack: {result['is_attack']}")
        print(f" Attack Type: {result['attack_type']}")
        print(f" Confidence: {result['detection_confidence'] * 100:.2f}%")

        if result['is_attack']:
            print(f" Severity Score: {result['severity_score']}/10")
            print(f" Severity Level: {result['severity_level']}")

        print("-" * 80)


if __name__ == "__main__":
    test_predictions()
