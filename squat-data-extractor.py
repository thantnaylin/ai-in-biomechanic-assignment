import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def create_extraction_summary(pose_data: Dict) -> None:
    """
    Create a summary of the extraction process
    """
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)

    video_info = pose_data['video_info']
    stats = pose_data['extraction_stats']

    print(f"Video: {video_info['filename']}")
    print(f"Resolution: {video_info['resolution']}")
    print(f"Duration: {video_info['duration']:.1f} seconds")
    print(f"Frame Rate: {video_info['fps']} FPS")
    print(f"Total Frames: {stats['total_frames_processed']}")
    print(f"Successful Detections: {stats['successful_detections']}")
    print(f"Detection Rate: {stats['detection_rate_percent']:.1f}%")
    print(f"Frames with Pose Data: {stats['frames_with_pose_data']}")
    print("=" * 50)


class SquatDataExtractor:
    def __init__(self, confidence_threshold: float = 0.5):
        print("Initializing AI Pose Estimation Pipeline...")

        # Initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2, # Highest accuracy model
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )

        # MediaPipe utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Define key joints for bilateral knee analysis
        self.key_landmarks = {
            # Left leg
            'LEFT_HIP': 23,
            'LEFT_KNEE': 25,
            'LEFT_ANKLE': 27,
            # Right leg
            'RIGHT_HIP': 24,
            'RIGHT_KNEE': 26,
            'RIGHT_ANKLE': 28,
            # Additional reference points
            'NOISE': 0,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
        }

        print("AI Pipeline Initialized Successfully!")

    @staticmethod
    def extract_3d_landmarks_from_video(self, video_path: str,
                                        save_frames: bool = False,
                                        output_dir: str = None) -> Optional[Dict]:
        """
        Extract 3D skeleton data from squat video using AI pose estimation

        Args:
            video_path: Path to input video
            save_frames: Whether to save annotated frames
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing extracted 3D pose data
        """
        print(f"Processing Video: {Path(video_path).name}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        print(f"Video Properties: {width}x{height}, {fps} FPS, {duration:.1f}s, {total_frames} frames")

        # Storage for extracted data
        pose_data = {
            'video_info': {
                'filename': Path(video_path).name,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'resolution': f"{width}x{height}",
                'extraction_date': datetime.now().isoformat()
            },
            'frames': []
        }

        frame_count = 0
        successful_detections = 0

        # Create output directory for frames if requested
        if save_frames and output_dir:
            frames_dir = os.path.join(output_dir, 'annotated_frames')
            os.makedirs(frames_dir, exist_ok=True)

        print("🧠 Running AI Pose Detection...")

        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run AI pose estimation
            results = self.pose.process(rgb_frame)

            # Process results if pose detected
            if results.pose_landmarks:
                successful_detections += 1

                # Extract 3D coordinates and visibility
                landmarks = results.pose_landmarks.landmark
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'landmarks_3d': {},
                    'landmark_visibility': {}
                }

                # Extract key joint coordinates
                for joint_name, landmark_idx in self.key_landmarks.items():
                    landmark = landmarks[landmark_idx]
                    frame_data['landmarks_3d'][joint_name] = {
                        'x': landmark.x,  # Normalized coordinates
                        'y': landmark.y,
                        'z': landmark.z,  # Depth estimate
                    }
                    frame_data['landmark_visibility'][joint_name] = landmark.visibility

                pose_data['frames'].append(frame_data)

                # Save annotated frame if requested
                if save_frames and output_dir:
                    # Draw pose landmarks on frame
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Add frame info
                    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Time: {frame_count / fps:.2f}s", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save frame
                    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, annotated_frame)

            frame_count += 1

            # Progress indicator
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.0f}% ({frame_count}/{total_frames} frames)")

        cap.release()

        # Calculate detection statistics
        detection_rate = (successful_detections / total_frames) * 100
        pose_data['extraction_stats'] = {
            'total_frames_processed': frame_count,
            'successful_detections': successful_detections,
            'detection_rate_percent': detection_rate,
            'frames_with_pose_data': len(pose_data['frames'])
        }

        print(f"Extraction Complete!")
        print(f"Detection Rate: {detection_rate:.1f}% ({successful_detections}/{total_frames} frames)")

        return pose_data

    def save_extracted_data(self, pose_data: Dict, output_path: str) -> None:
        """
        Save extracted 3D pose data to file

        Args:
            pose_data: Extracted pose data dictionary
            output_path: Path to save the data
        """
        print(f"💾 Saving extracted data to: {output_path}")

        # Save as NumPy array (similar to your dataset format)
        if output_path.endswith('.npy'):
            # Convert to structured numpy array
            frames_data = []
            for frame in pose_data['frames']:
                frame_array = []
                # Add frame info
                frame_array.extend([frame['frame_number'], frame['timestamp']])

                # Add 3D coordinates for each joint
                for joint_name in self.key_landmarks.keys():
                    if joint_name in frame['landmarks_3d']:
                        coords = frame['landmarks_3d'][joint_name]
                        visibility = frame['landmark_visibility'][joint_name]
                        frame_array.extend([coords['x'], coords['y'], coords['z'], visibility])
                    else:
                        frame_array.extend([0, 0, 0, 0])  # Missing data

                frames_data.append(frame_array)

            np.save(output_path, np.array(frames_data))

        # Also save as JSON for readability
        json_path = output_path.replace('.npy', '.json')
        with open(json_path, 'w') as f:
            json.dump(pose_data, f, indent=2)

        print(f"Data saved successfully!")
        print(f"NumPy format: {output_path}")
        print(f"JSON format: {json_path}")

    def create_extraction_summary(self, pose_data: Dict) -> None:
        """
        Create a summary of the extraction process
        """
        print("\n" + "=" * 50)
        print("EXTRACTION SUMMARY")
        print("=" * 50)

        video_info = pose_data['video_info']
        stats = pose_data['extraction_stats']

        print(f"Video: {video_info['filename']}")
        print(f"Resolution: {video_info['resolution']}")
        print(f"Duration: {video_info['duration']:.1f} seconds")
        print(f"Frame Rate: {video_info['fps']} FPS")
        print(f"Total Frames: {stats['total_frames_processed']}")
        print(f"Successful Detections: {stats['successful_detections']}")
        print(f"Detection Rate: {stats['detection_rate_percent']:.1f}%")
        print(f"Frames with Pose Data: {stats['frames_with_pose_data']}")
        print("=" * 50)

    def process_video(self, video_path: str, output_dir: str = "extracted_data", save_frames: bool = False) -> Dict:
        """
        Complete pipeline to process a single video

        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            save_frames: Whether to save annotated frames

        Returns:
            Extracted pose data dictionary
        """
        # Create output director
        os.makedirs(output_dir, exist_ok=True)

        # Extract 3D pose data
        pose_data = self.extract_3d_landmarks_from_video(
            self, video_path, save_frames=save_frames, output_dir=output_dir
        )

        if pose_data is None:
          return None

        # Save extracted data
        video_name = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{video_name}_3d_pose_data.npy")
        self.save_extracted_data(pose_data, output_file)

        # Create summary
        create_extraction_summary(pose_data)

        return pose_data

    def process_multiple_videos(self, video_directory: str, output_dir: str = "extracted_data"):
        """
        Process multiple squat videos in a directory

        Args:
            video_directory: Directory containing squat videos
            output_dir: Output directory for all results
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []

        # Find all video files
        for ext in video_extensions:
            video_files.extend(Path(video_directory).glob(f"*{ext}"))

        if not video_files:
            print(f"No video files found in {video_directory}")
            return

        print(f"🎬 Found {len(video_files)} video files to process")

        # Process each video
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing Video {i}/{len(video_files)}: {video_path.name}")
            print(f"{'=' * 60}")

            try:
                self.process_video(str(video_path), output_dir)
                print(f"✅ Successfully processed: {video_path.name}")
            except Exception as e:
                print(f"Error processing {video_path.name}: {str(e)}")

        print(f"\nBatch processing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    # Initialize the AI data extractor
    extractor = SquatDataExtractor(confidence_threshold=0.5)

    # Process a video
    video_path = "videos/Squat-front-view.mp4"
    output_directory = "squat_3d_data"

    # Extract 3D pose data
    pose_data = extractor.process_video(
        video_path=video_path,
        output_dir=output_directory,
        save_frames=True
    )

    print("Pose extraction complete!")

