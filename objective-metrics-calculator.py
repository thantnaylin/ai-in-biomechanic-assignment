import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ObjectiveMetricsCalculator:
    """
    Calculate objective clinical metrics for knee recovery assessment
    """

    def __init__(self):
        """
        Initialize the objective metrics calculator
        """

        print("Initializing Objective Metrics Calculator")

        # Clinical reference values (degrees)
        # Should be based on demographic, but we don't have time
        self.reference_values = {
            'normal_knee_rom': 140, # Normal knee range of motion
            'functional_flexion': 90, # Functional squat depth
            'asymmetry_threshold': 5, # Clinical significance threshold
            'normal_extension': 180, # Full knee extension
        }

        print("Calculator Initialized!")

    def load_biomechanical_data(self, csv_path:str) -> pd.DataFrame:
        """
        Load processed biomechanical data
        :param csv_path: Path to the biomechanical CSV file
        :return: DataFrame with processed biomechanical data
        """

        print(f"Loading biomechanical data: {Path(csv_path).name}")

        df = pd.read_csv(csv_path)

        # Data quality check
        required_columns = [
            'timestamp', 'left_knee_angle', 'right_knee_angle',
            'left_knee_velocity', 'right_knee_velocity', 'bilateral_difference'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")

        print(f"Loaded {len(df)} data points spanning {df['timestamp'].max():.1f} seconds")
        return df

    def detect_squat_repetitions(self, df: pd.DataFrame, prominence_threshold: float = 10) -> List[Dict]:
        """
        Detect individual squat repetitions from continuous data
        :param df: DataFrame with knee angle data
        :param prominence_threshold: Minimum peak prominence for detection
        :return: List of dictionaries containing repetition information
        """
        print("Detecting squat repetitions...")

        # Use average knee angle for repetition detection
        avg_angle = (df['left_knee_angle'] + df['right_knee_angle']) / 2
        avg_angle_smooth = gaussian_filter1d(avg_angle.fillna(method='ffill'), sigma=2)

        # Find peak (standing positions - higher angles)
        peaks, peak_properties = signal.find_peaks(
            avg_angle_smooth,
            prominence=prominence_threshold,
            distance=int(len(df) * 0.1) # Minimum distance between peaks
        )

        # Find valley (squat positions - lower angles)
        valleys, valley_properties = signal.find_peaks(
            -avg_angle_smooth,
            prominence=prominence_threshold,
            distance=int(len(df) * 0.1)
        )

        repetitions = []
        # Match peaks and valleys to create repetitions
        for i in range(min(len(peaks) - 1, len(valleys))):
            start_peak = peaks[i]
            valley = valleys[valleys > start_peak]

            if len(valley) > 0:
                valley_idx = valley[0]
                end_peak = peaks[peaks > valley_idx]

                if len(end_peak) > 0:
                    end_peak_idx = end_peak[0]

                    repetition = {
                        'rep_number': i + 1,
                        'start_frame': start_peak,
                        'bottom_frame': valley_idx,
                        'end_frame': end_peak_idx,
                        'start_time': df.iloc[start_peak]['timestamp'],
                        'bottom_time': df.iloc[valley_idx]['timestamp'],
                        'end_time': df.iloc[end_peak_idx]['timestamp'],
                        'duration': df.iloc[end_peak_idx]['timestamp'] - df.iloc[start_peak]['timestamp'],
                        'descent_duration': df.iloc[valley_idx]['timestamp'] - df.iloc[start_peak]['timestamp'],
                        'ascent_duration': df.iloc[end_peak_idx]['timestamp'] - df.iloc[valley_idx]['timestamp']
                    }
                    repetitions.append(repetition)

        print(f"Detected {len(repetitions)} squat repetitions")

        return repetitions

    def calculate_rom_metrics(self, df: pd.DataFrame, repetitions: List[Dict]) -> Dict:
        """
        Calculate Range of Motion (ROM) metrics
        :param df: DataFrame with biomechanical data
        :param repetitions: List of detected squat repetitions
        :return: Dictionary containing ROM metrics
        """

        print("Calculating ROM metrics...")

        rom_metrics = {
            'left_knee_rom': {},
            'right_knee_rom': {},
            'bilateral_rom': {},
            'per_repetition': []
        }

        # Overall ROM analysis
        left_angles = df['left_knee_angle'].dropna()
        right_angles = df['right_knee_angle'].dropna()

        # Left knee ROM
        rom_metrics['left_knee_rom'] = {
            'max_flexion': float(left_angles.min()),  # Deepest squat
            'max_extension': float(left_angles.max()),  # Standing position
            'total_rom': float(left_angles.max() - left_angles.min()),
            'mean_angle': float(left_angles.mean()),
            'std_angle': float(left_angles.std()),
            'functional_rom_percent': float((left_angles.max() - left_angles.min()) /
                                          self.reference_values['normal_knee_rom'] * 100)
        }

        # Right knee ROM
        rom_metrics['right_knee_rom'] = {
            'max_flexion': float(right_angles.min()),
            'max_extension': float(right_angles.max()),
            'total_rom': float(right_angles.max() - right_angles.min()),
            'mean_angle': float(right_angles.mean()),
            'std_angle': float(right_angles.std()),
            'functional_rom_percent': float((right_angles.max() - right_angles.min()) /
                                          self.reference_values['normal_knee_rom'] * 100)
        }

        # Bilateral ROM comparison
        rom_metrics['bilateral_rom'] = {
            'left_right_rom_difference': abs(rom_metrics['left_knee_rom']['total_rom'] -
                                           rom_metrics['right_knee_rom']['total_rom']),
            'left_right_flexion_difference': abs(rom_metrics['left_knee_rom']['max_flexion'] -
                                                rom_metrics['right_knee_rom']['max_flexion']),
            'left_right_extension_difference': abs(rom_metrics['left_knee_rom']['max_extension'] -
                                                  rom_metrics['right_knee_rom']['max_extension'])
        }

        # Per-repetition analysis
        for rep in repetitions:
            start_idx = rep['start_frame']
            end_idx = rep['end_frame']
            bottom_idx = rep['bottom_frame']

            rep_data = df.iloc[start_idx:end_idx + 1]

            rep_rom = {
                'repetition': rep['rep_number'],
                'left_max_flexion': float(rep_data['left_knee_angle'].min()),
                'right_max_flexion': float(rep_data['right_knee_angle'].min()),
                'left_max_extension': float(rep_data['left_knee_angle'].max()),
                'right_max_extension': float(rep_data['right_knee_angle'].max()),
                'bilateral_depth_difference': abs(rep_data['left_knee_angle'].min() -
                                                  rep_data['right_knee_angle'].min())
            }
            rom_metrics['per_repetition'].append(rep_rom)

        print("ROM metrics calculated.")
        return rom_metrics