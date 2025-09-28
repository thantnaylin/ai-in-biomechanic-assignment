import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    def calculate_symmetry_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate bilateral symmetry metrics
        :param df: DataFrame with biomechanical data
        :return: Dictionary containing symmetry metrics"""

        print("Calculating bilateral symmetry metrics...")
        left_angles = df['left_knee_angle'].dropna()
        right_angles = df['right_knee_angle'].dropna()
        bilateral_diff = df['bilateral_difference'].dropna()

        # Ensure same length for correlation
        min_len = min(len(left_angles), len(right_angles))
        left_sync = left_angles.iloc[:min_len]
        right_sync = right_angles.iloc[:min_len]

        symmetry_metrics = {
            'mean_asymmetry': float(bilateral_diff.mean()),
            'max_asymmetry': float(bilateral_diff.max()),
            'std_asymmetry': float(bilateral_diff.std()),
             'asymmetry_cv': float(bilateral_diff.std() / bilateral_diff.mean() * 100) if bilateral_diff.mean() != 0 else 0,
            # Symmetry indices
            'symmetry_index': float(1 - (bilateral_diff.mean() / max(left_angles.mean(), right_angles.mean()))),
            'bilateral_correlation': float(stats.pearsonr(left_sync, right_sync)[0]),
            'correlation_p_value': float(stats.pearsonr(left_sync, right_sync)[1]),

            # Clinical categorization
            'asymmetry_severity': 'mild' if bilateral_diff.mean() < 3 else
            'moderate' if bilateral_diff.mean() < 7 else 'severe',

            # Percentage of time with significant asymmetry
            'percent_asymmetric': float(
                (bilateral_diff > self.reference_values['asymmetry_threshold']).sum() / len(bilateral_diff) * 100)
        }

        print("Symmetry metrics calculated.")
        return symmetry_metrics

    def calculate_movement_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate movement quality and smoothness metrics

        Args:
            df: DataFrame with biomechanical data

        Returns:
            Dictionary containing movement quality metrics
        """
        print("Calculating movement quality metrics...")

        # Extract movement data
        left_vel = df['left_knee_velocity'].dropna()
        right_vel = df['right_knee_velocity'].dropna()
        left_acc = df['left_knee_acceleration'].dropna()
        right_acc = df['right_knee_acceleration'].dropna()

        # Movement smoothness (jerk approximation)
        left_jerk = np.sqrt(np.mean(np.diff(left_acc) ** 2)) if len(left_acc) > 1 else 0
        right_jerk = np.sqrt(np.mean(np.diff(right_acc) ** 2)) if len(right_acc) > 1 else 0

        quality_metrics = {
            'left_knee_quality': {
                'max_velocity': float(left_vel.abs().max()) if len(left_vel) > 0 else 0,
                'mean_velocity': float(left_vel.abs().mean()) if len(left_vel) > 0 else 0,
                'velocity_variability': float(left_vel.abs().std()) if len(left_vel) > 0 else 0,
                'max_acceleration': float(left_acc.abs().max()) if len(left_acc) > 0 else 0,
                'jerk_score': float(left_jerk)
            },
            'right_knee_quality': {
                'max_velocity': float(right_vel.abs().max()) if len(right_vel) > 0 else 0,
                'mean_velocity': float(right_vel.abs().mean()) if len(right_vel) > 0 else 0,
                'velocity_variability': float(right_vel.abs().std()) if len(right_vel) > 0 else 0,
                'max_acceleration': float(right_acc.abs().max()) if len(right_acc) > 0 else 0,
                'jerk_score': float(right_jerk)
            },
            'bilateral_quality': {
                'velocity_asymmetry': float(abs(left_vel.abs().mean() - right_vel.abs().mean())) if len(
                    left_vel) > 0 and len(right_vel) > 0 else 0,
                'acceleration_asymmetry': float(abs(left_acc.abs().mean() - right_acc.abs().mean())) if len(
                    left_acc) > 0 and len(right_acc) > 0 else 0,
                'combined_jerk_score': float((left_jerk + right_jerk) / 2),
                'movement_efficiency': float(1 / (1 + (left_jerk + right_jerk) / 2))  # Higher = more efficient
            }
        }

        print("Movement quality metrics calculated")
        return quality_metrics

    def calculate_temporal_metrics(self, repetitions: List[Dict]) -> Dict:
        """
        Calculate temporal and timing metrics

        Args:
            repetitions: List of detected repetitions

        Returns:
            Dictionary containing temporal metrics
        """
        print("Calculating temporal metrics...")

        if not repetitions:
            print("No repetitions detected for temporal analysis")
            return {}

        durations = [rep['duration'] for rep in repetitions]
        descent_durations = [rep['descent_duration'] for rep in repetitions]
        ascent_durations = [rep['ascent_duration'] for rep in repetitions]

        # Calculate ratios
        descent_ascent_ratios = [desc / asc if asc > 0 else 0
                                 for desc, asc in zip(descent_durations, ascent_durations)]

        temporal_metrics = {
            'repetition_analysis': {
                'total_repetitions': len(repetitions),
                'mean_duration': float(np.mean(durations)),
                'std_duration': float(np.std(durations)),
                'duration_cv': float(np.std(durations) / np.mean(durations) * 100) if np.mean(durations) != 0 else 0
            },
            'phase_analysis': {
                'mean_descent_duration': float(np.mean(descent_durations)),
                'mean_ascent_duration': float(np.mean(ascent_durations)),
                'descent_ascent_ratio': float(np.mean(descent_ascent_ratios)),
                'phase_consistency': float(np.std(descent_ascent_ratios))
            },
            'timing_variability': {
                'duration_variability': float(np.std(durations)),
                'descent_variability': float(np.std(descent_durations)),
                'ascent_variability': float(np.std(ascent_durations))
            }
        }

        # Calculate movement rate
        if repetitions:
            total_time = repetitions[-1]['end_time'] - repetitions[0]['start_time']
            temporal_metrics['movement_rate'] = {
                'reps_per_minute': float(len(repetitions) / (total_time / 60)) if total_time > 0 else 0,
                'total_exercise_duration': float(total_time)
            }

        print("Temporal metrics calculated")
        return temporal_metrics

    def calculate_consistency_metrics(self, rom_metrics: Dict,
                                      temporal_metrics: Dict) -> Dict:
        """
        Calculate movement consistency metrics across repetitions

        Args:
            rom_metrics: ROM metrics from calculate_rom_metrics
            temporal_metrics: Temporal metrics from calculate_temporal_metrics

        Returns:
            Dictionary containing consistency metrics
        """
        print("Calculating consistency metrics...")

        consistency_metrics = {}

        if 'per_repetition' in rom_metrics and rom_metrics['per_repetition']:
            # Depth consistency
            left_flexions = [rep['left_max_flexion'] for rep in rom_metrics['per_repetition']]
            right_flexions = [rep['right_max_flexion'] for rep in rom_metrics['per_repetition']]
            bilateral_diffs = [rep['bilateral_depth_difference'] for rep in rom_metrics['per_repetition']]

            consistency_metrics = {
                'depth_consistency': {
                    'left_flexion_variability': float(np.std(left_flexions)),
                    'right_flexion_variability': float(np.std(right_flexions)),
                    'bilateral_consistency': float(np.std(bilateral_diffs)),
                    'depth_consistency_score': float(1 / (1 + np.std(left_flexions + right_flexions)))
                    # Higher = more consistent
                }
            }

        # Add temporal consistency if available
        if 'timing_variability' in temporal_metrics:
            consistency_metrics['temporal_consistency'] = temporal_metrics['timing_variability']

        print("Consistency metrics calculated")
        return consistency_metrics

    def generate_clinical_summary(self, all_metrics: Dict) -> Dict:
        """
        Generate clinical summary and interpretation

        Args:
            all_metrics: Combined metrics dictionary

        Returns:
            Clinical summary dictionary
        """
        print("Generating clinical summary...")

        summary = {
            'assessment_date': pd.Timestamp.now().isoformat(),
            'overall_scores': {},
            'clinical_flags': [],
            'recommendations': []
        }

        # Overall ROM score (0-100)
        if 'rom_metrics' in all_metrics:
            rom = all_metrics['rom_metrics']
            left_rom_score = min(100, rom['left_knee_rom']['functional_rom_percent'])
            right_rom_score = min(100, rom['right_knee_rom']['functional_rom_percent'])
            summary['overall_scores']['rom_score'] = (left_rom_score + right_rom_score) / 2

        # Symmetry score (0-100)
        if 'symmetry_metrics' in all_metrics:
            sym = all_metrics['symmetry_metrics']
            symmetry_score = max(0, min(100, sym['symmetry_index'] * 100))
            summary['overall_scores']['symmetry_score'] = symmetry_score

        # Movement quality score (0-100)
        if 'quality_metrics' in all_metrics:
            qual = all_metrics['quality_metrics']
            quality_score = max(0, min(100, qual['bilateral_quality']['movement_efficiency'] * 100))
            summary['overall_scores']['quality_score'] = quality_score

        # Clinical flags
        if 'symmetry_metrics' in all_metrics:
            if all_metrics['symmetry_metrics']['mean_asymmetry'] > 10:
                summary['clinical_flags'].append("Significant bilateral asymmetry detected")

            if all_metrics['symmetry_metrics']['bilateral_correlation'] < 0.7:
                summary['clinical_flags'].append("Poor bilateral coordination")

        if 'rom_metrics' in all_metrics:
            rom = all_metrics['rom_metrics']
            if rom['left_knee_rom']['functional_rom_percent'] < 80:
                summary['clinical_flags'].append("Limited left knee range of motion")
            if rom['right_knee_rom']['functional_rom_percent'] < 80:
                summary['clinical_flags'].append("Limited right knee range of motion")

        # Recommendations
        if len(summary['clinical_flags']) == 0:
            summary['recommendations'].append("Movement patterns within normal parameters")
        else:
            if "asymmetry" in str(summary['clinical_flags']):
                summary['recommendations'].append("Consider unilateral strengthening exercises")
            if "range of motion" in str(summary['clinical_flags']):
                summary['recommendations'].append("Focus on mobility and flexibility training")

        print("Clinical summary generated")
        return summary

    def create_metrics_summary_csv(self, metrics: Dict, output_dir: str):
        """Create summary CSV of key metrics"""
        summary_data = []

        # Extract key metrics for CSV
        if 'rom_metrics' in metrics:
            rom = metrics['rom_metrics']
            summary_data.append({
                'Metric': 'Left Knee ROM (degrees)',
                'Value': f"{rom['left_knee_rom']['total_rom']:.1f}",
                'Reference': '140°',
                'Percentage': f"{rom['left_knee_rom']['functional_rom_percent']:.1f}%"
            })
            summary_data.append({
                'Metric': 'Right Knee ROM (degrees)',
                'Value': f"{rom['right_knee_rom']['total_rom']:.1f}",
                'Reference': '140°',
                'Percentage': f"{rom['right_knee_rom']['functional_rom_percent']:.1f}%"
            })

        if 'symmetry_metrics' in metrics:
            sym = metrics['symmetry_metrics']
            summary_data.append({
                'Metric': 'Bilateral Asymmetry (degrees)',
                'Value': f"{sym['mean_asymmetry']:.1f}",
                'Reference': '<5°',
                'Percentage': f"{sym['percent_asymmetric']:.1f}% of time"
            })
            summary_data.append({
                'Metric': 'Bilateral Correlation',
                'Value': f"{sym['bilateral_correlation']:.3f}",
                'Reference': '>0.85',
                'Percentage': '-'
            })

        if 'temporal_metrics' in metrics and 'repetition_analysis' in metrics['temporal_metrics']:
            temp = metrics['temporal_metrics']
            summary_data.append({
                'Metric': 'Repetitions Detected',
                'Value': str(temp['repetition_analysis']['total_repetitions']),
                'Reference': '-',
                'Percentage': '-'
            })

        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved: {csv_path}")

    def create_comprehensive_report(self, df: pd.DataFrame,
                                    output_dir: str = "objective_metrics") -> Dict:
        """
        Generate comprehensive objective metrics report

        Args:
            df: Biomechanical data DataFrame
            output_dir: Output directory for results

        Returns:
            Complete metrics dictionary
        """
        print("Generating comprehensive objective metrics report...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Detect repetitions
        repetitions = self.detect_squat_repetitions(df)

        # Step 2: Calculate all metrics
        rom_metrics = self.calculate_rom_metrics(df, repetitions)
        symmetry_metrics = self.calculate_symmetry_metrics(df)
        quality_metrics = self.calculate_movement_quality_metrics(df)
        temporal_metrics = self.calculate_temporal_metrics(repetitions)
        consistency_metrics = self.calculate_consistency_metrics(rom_metrics, temporal_metrics)

        # Step 3: Combine all metrics
        all_metrics = {
            'rom_metrics': rom_metrics,
            'symmetry_metrics': symmetry_metrics,
            'quality_metrics': quality_metrics,
            'temporal_metrics': temporal_metrics,
            'consistency_metrics': consistency_metrics,
            'repetitions_detected': repetitions
        }

        # Step 4: Generate clinical summary
        clinical_summary = self.generate_clinical_summary(all_metrics)
        all_metrics['clinical_summary'] = clinical_summary

        # Step 5: Save results
        report_file = os.path.join(output_dir, "objective_metrics_report.json")
        with open(report_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)

        # Step 6: Create summary CSV
        self.create_metrics_summary_csv(all_metrics, output_dir)

        # Step 7: Create visualization
        self.create_metrics_visualization(df, all_metrics, output_dir)

        print("Comprehensive report generated!")
        print(f"Report saved to: {report_file}")

        return all_metrics

    def create_metrics_visualization(self, df: pd.DataFrame,
                                     metrics: Dict, output_dir: str):
        """Create comprehensive metrics visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Objective Knee Recovery Assessment Metrics', fontsize=16, fontweight='bold')

        # 1. ROM Comparison
        if 'rom_metrics' in metrics:
            rom = metrics['rom_metrics']
            categories = ['Left Knee ROM', 'Right Knee ROM']
            values = [rom['left_knee_rom']['total_rom'], rom['right_knee_rom']['total_rom']]
            colors = ['blue', 'red']

            axes[0, 0].bar(categories, values, color=colors, alpha=0.7)
            axes[0, 0].axhline(y=140, color='green', linestyle='--', label='Normal ROM (140°)')
            axes[0, 0].set_title('Range of Motion Comparison')
            axes[0, 0].set_ylabel('Degrees')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Symmetry Analysis
        if 'symmetry_metrics' in metrics:
            bilateral_diff = df['bilateral_difference'].dropna()
            axes[0, 1].hist(bilateral_diff, bins=20, alpha=0.7, color='green')
            axes[0, 1].axvline(x=5, color='red', linestyle='--', label='Clinical Threshold (5°)')
            axes[0, 1].set_title('Bilateral Asymmetry Distribution')
            axes[0, 1].set_xlabel('Angle Difference (degrees)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Movement Quality Scores
        if 'quality_metrics' in metrics:
            qual = metrics['quality_metrics']
            scores = [
                qual['left_knee_quality']['jerk_score'],
                qual['right_knee_quality']['jerk_score']
            ]
            axes[0, 2].bar(['Left Jerk Score', 'Right Jerk Score'], scores,
                           color=['blue', 'red'], alpha=0.7)
            axes[0, 2].set_title('Movement Smoothness (Lower = Better)')
            axes[0, 2].set_ylabel('Jerk Score')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Clinical Summary Scores
        if 'clinical_summary' in metrics and 'overall_scores' in metrics['clinical_summary']:
            scores = metrics['clinical_summary']['overall_scores']
            score_names = list(scores.keys())
            score_values = list(scores.values())

            bars = axes[1, 0].bar(score_names, score_values, color='purple', alpha=0.7)
            axes[1, 0].set_title('Overall Assessment Scores')
            axes[1, 0].set_ylabel('Score (0-100)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)

            # Add score labels on bars
            for bar, value in zip(bars, score_values):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                                f'{value:.1f}', ha='center', va='bottom')

        # 5. Temporal Analysis
        if 'repetitions_detected' in metrics and metrics['repetitions_detected']:
            reps = metrics['repetitions_detected']
            durations = [rep['duration'] for rep in reps]
            rep_numbers = [rep['rep_number'] for rep in reps]

            axes[1, 1].plot(rep_numbers, durations, 'o-', color='orange', linewidth=2)
            axes[1, 1].set_title('Repetition Duration Consistency')
            axes[1, 1].set_xlabel('Repetition Number')
            axes[1, 1].set_ylabel('Duration (seconds)')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Bilateral Correlation Plot
        left_angles = df['left_knee_angle'].dropna()
        right_angles = df['right_knee_angle'].dropna()
        min_len = min(len(left_angles), len(right_angles))

        if min_len > 0:
            axes[1, 2].scatter(left_angles[:min_len], right_angles[:min_len],
                               alpha=0.6, s=10)
            axes[1, 2].plot([left_angles.min(), left_angles.max()],
                            [left_angles.min(), left_angles.max()],
                            'r--', alpha=0.8, label='Perfect Symmetry')
            axes[1, 2].set_title('Bilateral Movement Correlation')
            axes[1, 2].set_xlabel('Left Knee Angle (degrees)')
            axes[1, 2].set_ylabel('Right Knee Angle (degrees)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        plot_path = os.path.join(output_dir, "objective_metrics_visualization.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved: {plot_path}")


if __name__ == '__main__':
    # Initialize the objective metric calculator
    calculator = ObjectiveMetricsCalculator()

    # Load biomechanical data
    csv_file_path = './biomechanical_results/Squat-front-view_biomechanical_data.csv'

    # Load the processed data
    biomech_data = calculator.load_biomechanical_data(csv_file_path)

    # Generate comprehensive objective metric report
    metrics_report = calculator.create_comprehensive_report(df=biomech_data, output_dir="objective-assessment-report")