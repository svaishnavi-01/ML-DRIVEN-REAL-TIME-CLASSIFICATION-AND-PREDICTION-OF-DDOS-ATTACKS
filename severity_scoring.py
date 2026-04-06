"""
Attack Severity Scoring Module
"""
import numpy as np

class SeverityScorer:
    def __init__(self):
        # Threshold values (can be tuned based on network baseline)
        self.traffic_threshold = 100000  # bytes
        self.packet_rate_threshold = 1000  # packets/sec
        self.duration_threshold = 60  # seconds

        # Attack type weights
        self.attack_weights = {
            'DoS': 1.0,
            'DDoS': 1.0,
            'Exploits': 0.9,
            'Backdoor': 0.95,
            'Reconnaissance': 0.6,
            'Fuzzers': 0.7,
            'Analysis': 0.5,
            'Shellcode': 0.85,
            'Worms': 0.9,
            'Generic': 0.7,
            'Normal': 0.0
        }

    def calculate_severity_score(self, traffic_volume, packet_rate, duration, attack_type):
        """
        Calculate severity score (1-10) based on traffic characteristics

        Parameters:
        -----------
        traffic_volume : float
            Total bytes transferred
        packet_rate : float
            Packets per second
        duration : float
            Attack duration in seconds
        attack_type : str
            Type of attack detected

        Returns:
        --------
        score : float (1-10)
        severity_level : str (Low/Medium/High)
        """

        # Normalize factors
        traffic_factor = min(traffic_volume / self.traffic_threshold, 1.0)
        rate_factor = min(packet_rate / self.packet_rate_threshold, 1.0)
        duration_factor = min(duration / self.duration_threshold, 1.0)

        # Get attack weight
        attack_weight = self.attack_weights.get(attack_type, 0.7)

        # Calculate base score (1-10)
        base_score = (traffic_factor + rate_factor + duration_factor) / 3
        weighted_score = base_score * attack_weight * 10

        # Ensure score is between 1-10
        final_score = max(1.0, min(10.0, weighted_score))

        # Determine severity level
        if final_score >= 7.0:
            severity_level = "High"
        elif final_score >= 4.0:
            severity_level = "Medium"
        else:
            severity_level = "Low"

        return round(final_score, 2), severity_level

    def score_from_dataframe_row(self, row, attack_type):
        """
        Calculate severity from a DataFrame row
        """
        # Extract relevant features from UNSW-NB15
        traffic_volume = row.get('sbytes', 0) + row.get('dbytes', 0)
        packet_rate = row.get('spkts', 0) + row.get('dpkts', 0)
        duration = row.get('dur', 1)

        return self.calculate_severity_score(
            traffic_volume, packet_rate, duration, attack_type
        )
