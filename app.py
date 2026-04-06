"""
Simple Flask Backend for DDoS Detection Dashboard
"""

from flask import Flask, render_template, jsonify
from prediction import DDoSPredictor
from severity_scoring import SeverityScorer
import pandas as pd
import random
import os

app = Flask(__name__)

# -----------------------------------------------------
#  FIX: Use ABSOLUTE PATH so models load correctly
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

predictor = DDoSPredictor(
    model_path=os.path.join(BASE_DIR, 'data', 'models'),
    preprocessor_path=os.path.join(BASE_DIR, 'data', 'processed')
)

# -----------------------------------------------------

scorer = SeverityScorer()

# Load test data for demo
test_data = pd.read_csv(os.path.join(BASE_DIR, 'Training and Testing Sets', 'UNSW_NB15_testing-set.csv'))

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predict')
def get_prediction():
    """API endpoint to get prediction for random sample"""
    
    # Get random sample from test data
    sample = test_data.sample(1).iloc[0]
    data_dict = sample.to_dict()
    
    # Make prediction
    result = predictor.predict(data_dict)
    
    # Calculate severity if attack
    severity_score = 0
    severity_level = "N/A"
    
    if result['is_attack']:
        severity_score, severity_level = scorer.score_from_dataframe_row(
            sample, result['attack_type']
        )
    
    # Prepare response
    response = {
        'source_ip': str(sample.get('srcip', 'Unknown')),
        'dest_ip': str(sample.get('dstip', 'Unknown')),
        'protocol': str(sample.get('proto', 'Unknown')),
        'is_attack': result['is_attack'],
        'classification': 'ATTACK' if result['is_attack'] else 'NORMAL',
        'attack_type': result['attack_type'],
        'confidence': round(result['detection_confidence'] * 100, 2),
        'severity_score': severity_score,
        'severity_level': severity_level,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(response)

@app.route('/api/stats')
def get_stats():
    """API endpoint to get overall statistics"""
    
    # Simulate statistics (in real system, this comes from database)
    stats = {
        'total_traffic': random.randint(1000, 2000),
        'attacks_detected': random.randint(20, 50),
        'attack_rate': round(random.uniform(1.5, 3.5), 2),
        'active_alerts': random.randint(2, 8),
        'binary_accuracy': 89.23,
        'multiclass_accuracy': 91.56
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    print("="*60)
    print("DDoS Detection Dashboard Server Starting...")
    print("Open browser and visit: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, port=5000)
