from prediction import DDoSPredictor
import pandas as pd

# Initialize your model (make sure paths are correct for your project)
predictor = DDoSPredictor(
    model_path='data/models/', 
    preprocessor_path='data/processed/'
)

# Load some network traffic data (here, from test CSV)
test_data = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/major_proj_19/Training and Testing Sets/UNSW_NB15_testing-set.csv")

print("Sample predictions:\n")
for idx, row in test_data.sample(10, random_state=42).iterrows():
    # Convert row to dict for prediction
    data_dict = row.to_dict()
    # Get prediction
    result = predictor.predict(data_dict)
    # Display in CMD
    print(f"Sample {idx+1}: Source IP: {row.get('srcip')} --> Dest IP: {row.get('dstip')}")
    if result['is_attack']:
        print(f"  [ATTACK]   Type: {result['attack_type']}   Confidence: {result['attack_confidence']*100:.1f}%")
    else:
        print(f"  [NORMAL]   Network traffic is normal.")
    print("-" * 50)
