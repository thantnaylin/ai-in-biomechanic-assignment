import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class BiomechanicalProcessor:
    """
    Process 3D pose data to extract biomechanical movement metrics
    """

    def __init__(self):
        """Initialize the biomechanical processor"""
        print("Initializing Biomechanical Data Processor...")

        # Joint indices for bilateral analysis
        self.join_mapping = {
            'LEFT_HIP': 0,
            'LEFT_KNEE': 1,
            'LEFT_ANKLE': 2,
            'RIGHT_HIP': 3,
            'RIGHT_KNEE': 4,
            'RIGHT_ANKLE': 5
        }

        print("Processor Initialized!")

    def load_extracted_data(self, npy_path: str, json_path: str = None) -> Dict:
        """
        Load the extracted 3D pose data
        :param npy_path: Path to the .npy file
        :param json_path: Path to the .json metadat file
        :return: Dictionary containing loaded data
        """
        print(f"Loading extracted data from {Path(npy_path).name}...")

        # Load numpy array
        pose_array = np.load(npy_path)
        print(f"Data shape: {pose_array.shape}")

        # LOAD JSON metadata if available
        metadata = None
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata with {len(metadata['frames'])} frames")
        elif json_path is None:
            # Try to find JSON file automatically
            json_path = npy_path.replace('.npy', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata with {len(metadata['frames'])} frames")

        return {
            'pose_array': pose_array,
            'metadata': metadata,
            'data_path': npy_path
        }

    def extract_joint_coordinates(self, data: Dict) -> pd.DataFrame:
        """
        Extract joint coordinates from the loaded data
        :param data:  dictionary from load_extracted_data
        :return: DataFrame with joint coordinates and timestamps
        """
        print("Extracting joint coordinates...")

        pose_array = data['pose_array']
        metadata = data['metadata']

        # Initialize DataFrame
        df_data = []

        if metadata:
            # Use metadata for accurate extraction
            for frame_data in metadata['frames']:
                frame_info = {
                    'frame': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                }
                # Extract 3D coordinates for bilateral knee analysis
                joints_3d = frame_data['landmarks_3d']
                visibility = frame_data['landmark_visibility']

                if 'LEFT_HIP' in joints_3d:
                    frame_info.update({
                        # Left leg
                        'left_hip_x': joints_3d['LEFT_HIP']['x'],
                        'left_hip_y': joints_3d['LEFT_HIP']['y'],
                        'left_hip_z': joints_3d['LEFT_HIP']['z'],
                        'left_knee_x': joints_3d['LEFT_KNEE']['x'],
                        'left_knee_y': joints_3d['LEFT_KNEE']['y'],
                        'left_knee_z': joints_3d['LEFT_KNEE']['z'],
                        'left_ankle_x': joints_3d['LEFT_ANKLE']['x'],
                        'left_ankle_y': joints_3d['LEFT_ANKLE']['y'],
                        'left_ankle_z': joints_3d['LEFT_ANKLE']['z'],
                        # Right leg
                        'right_hip_x': joints_3d['RIGHT_HIP']['x'],
                        'right_hip_y': joints_3d['RIGHT_HIP']['y'],
                        'right_hip_z': joints_3d['RIGHT_HIP']['z'],
                        'right_knee_x': joints_3d['RIGHT_KNEE']['x'],
                        'right_knee_y': joints_3d['RIGHT_KNEE']['y'],
                        'right_knee_z': joints_3d['RIGHT_KNEE']['z'],
                        'right_ankle_x': joints_3d['RIGHT_ANKLE']['x'],
                        'right_ankle_y': joints_3d['RIGHT_ANKLE']['y'],
                        'right_ankle_z': joints_3d['RIGHT_ANKLE']['z'],
                        # Visibility scores
                        'left_leg_visibility': (visibility['LEFT_HIP'] + visibility['LEFT_KNEE'] + visibility[
                            'LEFT_ANKLE']) / 3,
                        'right_leg_visibility': (visibility['RIGHT_HIP'] + visibility['RIGHT_KNEE'] + visibility[
                            'RIGHT_ANKLE']) / 3
                    })

                    df_data.append(frame_info)

        else:
            # Fallback: extract from numpy array directly
            print("No metadata found, using direct array extraction")
            for i, row in enumerate(pose_array):
                frame_info = {
                    'frame': row[0] if len(row) > 0 else i,
                    'timestamp': row[1] if len(row) > 1 else i * 0.033  # Assume 30fps
                }

                # Extract coordinates (this depends on your array structure)
                if len(row) >= 26:  # Ensure we have enough data
                    frame_info.update({
                        'left_hip_x': row[2], 'left_hip_y': row[3], 'left_hip_z': row[4],
                        'left_knee_x': row[6], 'left_knee_y': row[7], 'left_knee_z': row[8],
                        'left_ankle_x': row[10], 'left_ankle_y': row[11], 'left_ankle_z': row[12],
                        'right_hip_x': row[14], 'right_hip_y': row[15], 'right_hip_z': row[16],
                        'right_knee_x': row[18], 'right_knee_y': row[19], 'right_knee_z': row[20],
                        'right_ankle_x': row[22], 'right_ankle_y': row[23], 'right_ankle_z': row[24],
                        'left_leg_visibility': 1.0,  # Default
                        'right_leg_visibility': 1.0  # Default
                    })

                df_data.append(frame_info)

        df = pd.DataFrame(df_data)
        print(f"Extracted coordinates for {len(df)} frames")
        return df

    def calculate_3d_angle(self, point1: Tuple, point2: Tuple, point3: Tuple) -> float:
        """
        Calculate 3D angle between three points (hip-knee-ankle)
        :param point1: Hip coordinates (x, y, z)
        :param point2: Knee coordinates (x, y, z) - vertex
        :param point3: Ankle coordinates (x, y, z)
        :return: Angle in degrees
        """

        # Convert to numpy arrays
        p1 = np.array(point1) # Hip
        p2 = np.array(point2) # Knee (vertex)
        p3 = np.array(point3) # Ankle

        # Calculate vectors
        v1 = p1 - p2 # Hip to Knee
        v2 = p3 - p2 # Ankle to knee

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Calculate angle in degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_bilateral_knee_angle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bilateral knee joint angles from 3D coordinates
        :param df: DataFrame with joint coordinates
        :return: DataFrame with added knee angles
        """
        print("Calculating bilateral knee angles...")

        left_angles = []
        right_angles = []

        for _, row in df.iterrows():
            try:
                # Left knee angle (hip-knee-angle)
                left_hip = (row['left_hip_x'], row['left_hip_y'], row['left_hip_z'])
                left_knee = (row['left_knee_x'], row['left_knee_y'], row['left_knee_z'])
                left_ankle = (row['left_ankle_x'], row['left_ankle_y'], row['left_ankle_z'])

                left_angle = self.calculate_3d_angle(left_hip, left_knee, left_ankle)
                left_angles.append(left_angle)

                # Right knee angle (hip-knee-ankle)
                right_hip = (row['right_hip_x'], row['right_hip_y'], row['right_hip_z'])
                right_knee = (row['right_knee_x'], row['right_knee_y'], row['right_knee_z'])
                right_ankle = (row['right_ankle_x'], row['right_ankle_y'], row['right_ankle_z'])

                right_angle = self.calculate_3d_angle(right_hip, right_knee, right_ankle)
                right_angles.append(right_angle)

            except Exception as e:
                # Handle missing or invalid data
                left_angles.append(np.nan)
                right_angles.append(np.nan)

        # Add angles to dataframe
        df['left_knee_angle'] = left_angles
        df['right_knee_angle'] = right_angles

        # Calculate bilateral metrics
        df['bilateral_difference'] = abs(df['left_knee_angle'] - df['right_knee_angle'])
        df['average_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2

        print(f"Calculated knee angles for {len(df)} frames")
        return df

    def calculate_temporal_features(self, df: pd.DataFrame, smoothing=True) -> pd.DataFrame:
        """
        Calculate temporal movement features (velocity, acceleration)

        Args:
            df: DataFrame with knee angles
            smoothing: Apply Gaussian smoothing to reduce noise

        Returns:
            DataFrame with temporal features
        """
        print("⏱Calculating temporal movement features...")

        if smoothing:
            # Apply Gaussian smoothing to reduce noise
            sigma = 2  # Smoothing parameter
            df['left_knee_angle_smooth'] = gaussian_filter1d(df['left_knee_angle'].fillna(method='ffill'), sigma)
            df['right_knee_angle_smooth'] = gaussian_filter1d(df['right_knee_angle'].fillna(method='ffill'), sigma)
        else:
            df['left_knee_angle_smooth'] = df['left_knee_angle']
            df['right_knee_angle_smooth'] = df['right_knee_angle']

        # Calculate time differences
        df['time_diff'] = df['timestamp'].diff()

        # Calculate angular velocities (degrees/second)
        df['left_knee_velocity'] = df['left_knee_angle_smooth'].diff() / df['time_diff']
        df['right_knee_velocity'] = df['right_knee_angle_smooth'].diff() / df['time_diff']
        df['bilateral_velocity_diff'] = abs(df['left_knee_velocity'] - df['right_knee_velocity'])

        # Calculate angular accelerations (degrees/second²)
        df['left_knee_acceleration'] = df['left_knee_velocity'].diff() / df['time_diff']
        df['right_knee_acceleration'] = df['right_knee_velocity'].diff() / df['time_diff']
        df['bilateral_acceleration_diff'] = abs(df['left_knee_acceleration'] - df['right_knee_acceleration'])

        # Movement direction indicators
        df['left_knee_direction'] = np.where(df['left_knee_velocity'] > 0, 1,
                                             np.where(df['left_knee_velocity'] < 0, -1, 0))
        df['right_knee_direction'] = np.where(df['right_knee_velocity'] > 0, 1,
                                              np.where(df['right_knee_velocity'] < 0, -1, 0))

        print("Temporal features calculated")
        return df

    def visualize_biomechanical_analysis(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive visualization of biomechanical analysis

        Args:
            df: DataFrame with all calculated features
            save_path: Path to save the plot
        """
        print("Creating biomechanical analysis visualization...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Bilateral Knee Biomechanical Analysis', fontsize=16, fontweight='bold')

        # 1. Bilateral knee angles over time
        axes[0, 0].plot(df['timestamp'], df['left_knee_angle'], 'b-', label='Left Knee', linewidth=2)
        axes[0, 0].plot(df['timestamp'], df['right_knee_angle'], 'r-', label='Right Knee', linewidth=2)
        axes[0, 0].set_title('Bilateral Knee Angles Over Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Knee Angle (degrees)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Bilateral symmetry analysis
        axes[0, 1].plot(df['timestamp'], df['bilateral_difference'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=df['bilateral_difference'].mean(), color='orange', linestyle='--',
                           label=f'Mean: {df["bilateral_difference"].mean():.1f}°')
        axes[0, 1].set_title('Bilateral Symmetry (Angle Difference)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angle Difference (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Angular velocities
        axes[1, 0].plot(df['timestamp'], df['left_knee_velocity'], 'b-', label='Left Knee', linewidth=2)
        axes[1, 0].plot(df['timestamp'], df['right_knee_velocity'], 'r-', label='Right Knee', linewidth=2)
        axes[1, 0].set_title('Angular Velocities')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Angular accelerations
        axes[1, 1].plot(df['timestamp'], df['left_knee_acceleration'], 'b-', label='Left Knee', linewidth=2)
        axes[1, 1].plot(df['timestamp'], df['right_knee_acceleration'], 'r-', label='Right Knee', linewidth=2)
        axes[1, 1].set_title('Angular Accelerations')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angular Acceleration (deg/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Knee angle distribution comparison
        axes[2, 0].hist(df['left_knee_angle'].dropna(), alpha=0.7, bins=30,
                        label='Left Knee', color='blue', density=True)
        axes[2, 0].hist(df['right_knee_angle'].dropna(), alpha=0.7, bins=30,
                        label='Right Knee', color='red', density=True)
        axes[2, 0].set_title('Knee Angle Distributions')
        axes[2, 0].set_xlabel('Angle (degrees)')
        axes[2, 0].set_ylabel('Density')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Movement pattern correlation
        valid_data = df.dropna(subset=['left_knee_angle', 'right_knee_angle'])
        axes[2, 1].scatter(valid_data['left_knee_angle'], valid_data['right_knee_angle'],
                           alpha=0.6, s=20, c=valid_data['timestamp'], cmap='viridis')
        axes[2, 1].plot([valid_data['left_knee_angle'].min(), valid_data['left_knee_angle'].max()],
                        [valid_data['right_knee_angle'].min(), valid_data['right_knee_angle'].max()],
                        'r--', alpha=0.8, label='Perfect Symmetry')
        axes[2, 1].set_title('Left vs Right Knee Angle Correlation')
        axes[2, 1].set_xlabel('Left Knee Angle (degrees)')
        axes[2, 1].set_ylabel('Right Knee Angle (degrees)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")

        plt.show()

    def process_extracted_data(self, npy_path: str, output_dir: str = "biomechanical analysis") -> pd.DataFrame:
        """
        Complete biomechanical processing pipeline
        :param npy_path: Path to extracted 3D pose data
        :param output_dir: Output directory for results
        :return:
            DataFrame with all biomechanical features
        """
        print("Starting biomechanical analysis pipeline...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load extracted data
        data = self.load_extracted_data(npy_path)

        # Step 2: Extract joint coordinates
        df = self.extract_joint_coordinates(data)

        print(df.head())
        # Step 3: Calculate bilateral knee angles
        df = self.calculate_bilateral_knee_angle(df)

        # Step 4: Calculate temporal features
        df = self.calculate_temporal_features(df)

        # Step 5: Save processed data
        video_name = Path(npy_path).stem.replace('_3d_pose_data', '')
        processed_file = os.path.join(output_dir, video_name + '_biomechanical_data.csv')
        df.to_csv(processed_file, index=False)
        print(f"Processed data saved: {processed_file}")

        # Step 6: Create visualization
        plot_file = os.path.join(output_dir, f"{video_name}_biomechanical_analysis.png")
        self.visualize_biomechanical_analysis(df, plot_file)

        print("Biomechanical analysis complete!")

        return df

if __name__ == "__main__":
    # Initialize processor
    processor = BiomechanicalProcessor()

    # Process extracted 3D pose data
    npy_file_path = './squat_3d_data/Squat-front-view_3d_pose_data.npy'

    # Run complete analysis
    biomechanical_data = processor.process_extracted_data(npy_path=npy_file_path, output_dir="biomechanical_results")