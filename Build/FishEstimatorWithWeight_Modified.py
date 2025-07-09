"""
Fish Length Detection with Advanced Compression Techniques
========================================================

This script detects fish using a trained YOLO model (.pt file) and estimates their length
using a 5x5 cm ArUco marker as a scale reference, with multiple compression techniques implemented.

IMPLEMENTED COMPRESSION TECHNIQUES:
1. Video Resolution Scaling - Reduce frame dimensions while maintaining aspect ratio
2. Frame Rate Reduction - Process every Nth frame to reduce temporal redundancy
3. JPEG Quality Compression - Compress individual frames using JPEG encoding
4. H.264 Codec Optimization - Use efficient video codec with CRF quality control
5. Dynamic Frame Skipping - Skip frames without ArUco markers to reduce processing
6. Batch Processing Buffer - Process frames in batches for memory efficiency
7. Model Export Optimization - Convert model to ONNX for faster inference and smaller size

Requirements:
- OpenCV (cv2)
- NumPy  
- Ultralytics YOLO
- FFmpeg (optional, for advanced compression)

Author: Yash Sarang
Date: May 2025
"""

from collections import Counter
import os
import gc
import cv2
import time
import subprocess

import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2.aruco as aruco
from ultralytics import YOLO

from sklearn.cluster import DBSCAN


class CompressedFishLengthEstimator:
    def __init__(self, model_path, marker_size_cm=5.0, confidence_threshold=0.5,
                 use_optimized_model=True):
        """
        Initialize the Fish Length Estimator with compression optimizations

        Args:
            model_path (str): Path to the trained YOLO model (.pt file)
            marker_size_cm (float): Size of the ArUco marker in centimeters
            confidence_threshold (float): Minimum confidence for fish detection
            use_optimized_model (bool): Whether to export and use ONNX model for efficiency
        """
        # COMPRESSION TECHNIQUE 7: Model Export Optimization
        self.original_model_path = model_path
        self.optimized_model_path = None

        # Validate model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path.endswith('.pt'):
            raise ValueError(
                f"Model file must be a .pt file, got: {model_path}")

        # Load and optionally optimize the model
        try:
            if use_optimized_model:
                self.optimized_model_path = self._create_optimized_model(
                    model_path)
                self.fish_model = YOLO(self.optimized_model_path)
                print(
                    f"✓ Using optimized ONNX model: {self.optimized_model_path}")
            else:
                self.fish_model = YOLO(model_path)
                print(f"✓ Using original PyTorch model: {model_path}")

            # Display model information
            if hasattr(self.fish_model, 'names'):
                class_names = list(self.fish_model.names.values())
                print(f"✓ Model classes: {class_names}")

        except Exception as e:
            # Fallback to original model if optimization fails
            print(f"⚠ Model optimization failed, using original: {e}")
            self.fish_model = YOLO(model_path)

        # ArUco marker setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Parameters
        self.marker_size_cm = marker_size_cm
        self.confidence_threshold = confidence_threshold

        # COMPRESSION TECHNIQUE 6: Batch Processing Buffer
        self.frame_buffer = []
        # Load polynomial weight estimation model
        try:
            self.weight_model = joblib.load("weight_model_poly2.pkl")
            self.poly_transformer = joblib.load("poly_transformer2.pkl")
            print("✓ Weight estimation model loaded")
        except Exception as e:
            print(f"⚠ Failed to load weight model: {e}")
            self.weight_model = None
            self.poly_transformer = None

        self.buffer_size = 10  # Process frames in batches

        print(f"✓ ArUco detector initialized for {marker_size_cm}cm markers")
        print(f"✓ Confidence threshold set to {confidence_threshold}")

    def _create_optimized_model(self, model_path):
        """
        COMPRESSION TECHNIQUE 7: Model Export Optimization
        Create an optimized ONNX version of the model for faster inference and smaller size

        Args:
            model_path (str): Original .pt model path

        Returns:
            str: Path to optimized model
        """
        try:
            base_path = Path(model_path)
            onnx_path = base_path.parent / f"{base_path.stem}_optimized.onnx"

            # Check if optimized model already exists
            if onnx_path.exists():
                print(f"✓ Found existing optimized model: {onnx_path}")
                return str(onnx_path)

            # Load model and export to ONNX
            print("🔄 Creating optimized ONNX model...")
            temp_model = YOLO(model_path)

            # Export with optimization settings
            temp_model.export(
                format='onnx',
                optimize=True,     # Optimize for inference
                half=False,        # Keep full precision for accuracy
                simplify=True,     # Simplify the model graph
                dynamic=False,     # Static input shapes for better optimization
                opset=11          # ONNX opset version for compatibility
            )

            # Find the exported ONNX file
            exported_onnx = base_path.parent / f"{base_path.stem}.onnx"
            if exported_onnx.exists():
                # Rename to our optimized naming convention
                if onnx_path != exported_onnx:
                    exported_onnx.rename(onnx_path)

                # Compare file sizes
                original_size = os.path.getsize(model_path) / (1024*1024)  # MB
                optimized_size = os.path.getsize(onnx_path) / (1024*1024)  # MB
                print(f"✓ Model optimization complete:")
                print(f"  Original .pt: {original_size:.1f} MB")
                print(f"  Optimized ONNX: {optimized_size:.1f} MB")

                return str(onnx_path)
            else:
                raise Exception("ONNX export failed")

        except Exception as e:
            print(f"⚠ Model optimization failed: {e}")
            return model_path

    def detect_aruco_marker(self, frame):
        """
        Detect ArUco marker in the frame and calculate scale factor
        Optimized for compression pipeline
        """
        # Convert frame to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if len(corners) > 0:
            # Get the first detected marker
            marker_corners = corners[0][0]

            # Calculate marker width in pixels (distance between corner 0 and 1)
            width_pixels = np.linalg.norm(
                marker_corners[1] - marker_corners[0])

            # Calculate scale factor: pixels per centimeter
            scale_factor = width_pixels / self.marker_size_cm

            # Calculate marker center
            marker_center = np.mean(marker_corners, axis=0).astype(int)

            return scale_factor, marker_corners, marker_center

        return None, None, None

    def estimate_fish_length_bbox(self, bbox, scale_factor):
        """
        Estimate fish length using bounding box dimensions
        """
        x1, y1, x2, y2 = bbox

        # Calculate bounding box dimensions in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        diagonal_pixels = np.sqrt(width_pixels**2 + height_pixels**2)

        # Convert to centimeters
        width_cm = width_pixels / scale_factor
        height_cm = height_pixels / scale_factor
        diagonal_cm = diagonal_pixels / scale_factor

        # Use the longer side as the fish length estimate
        length_cm = max(width_cm, height_cm)

        return length_cm, min(width_cm, height_cm), diagonal_cm

    def compress_frame(self, frame, quality=70):
        """
        COMPRESSION TECHNIQUE 3: JPEG Quality Compression
        Compress individual frame using JPEG encoding

        Args:
            frame (numpy.ndarray): Input frame
            quality (int): JPEG quality 1-100

        Returns:
            numpy.ndarray: Compressed frame
        """
        if quality < 100:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return compressed_frame
        return frame

    def resize_frame(self, frame, max_width=1280, max_height=720):
        """
        COMPRESSION TECHNIQUE 1: Video Resolution Scaling
        Resize frame while maintaining aspect ratio

        Args:
            frame (numpy.ndarray): Input frame
            max_width (int): Maximum width
            max_height (int): Maximum height

        Returns:
            tuple: (resized_frame, scale_factor_applied)
        """
        height, width = frame.shape[:2]

        # Calculate scale factor to fit within max dimensions
        scale_w = max_width / width if width > max_width else 1.0
        scale_h = max_height / height if height > max_height else 1.0
        scale_factor = min(scale_w, scale_h)

        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_frame, scale_factor

        return frame, 1.0

    def draw_annotations(self, frame, detections, marker_info):
        """
        Draw all annotations (fish detections and ArUco marker) on frame
        """
        annotated_frame = frame.copy()

        # Draw ArUco marker if detected
        if marker_info['detected']:
            scale_factor, marker_corners = marker_info['scale_factor'], marker_info['corners']

            # Draw marker outline
            cv2.polylines(annotated_frame, [
                          marker_corners.astype(int)], True, (255, 0, 0), 3)

            # Draw marker center
            center = np.mean(marker_corners, axis=0).astype(int)
            cv2.circle(annotated_frame, tuple(center), 5, (255, 0, 0), -1)

            # Scale information
            scale_text = f'Scale: {scale_factor:.1f} px/cm'
            cv2.putText(annotated_frame, scale_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Draw fish detections
        for i, detection in enumerate(detections, 1):
            bbox = detection['bbox']
            length_cm = detection['length_cm']
            width_cm = detection['width_cm']
            diagonal_cm = detection['diagonal_cm']

            confidence = detection['confidence']
            class_name = detection.get('class_name', 'fish')

            x1, y1, x2, y2 = [int(x) for x in bbox]

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text
            text = f'{class_name} {i}: {length_cm:.1f}cm ({confidence:.2f})'
            text_y = max(y1 - 10, 20)

            # Text background for readability
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, text_y - text_height - 5),
                          (x1 + text_width, text_y + 5), (0, 255, 0), -1)

            # Draw text
            cv2.putText(annotated_frame, text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            weight_estimate = self.estimate_weight_from_length(length_cm)

            weight_text = f'~{weight_estimate:.1f}g'

            # Text for weight below the length text
            cv2.putText(annotated_frame, weight_text, (x1, text_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Fish count
        count_text = f'Fish detected: {len(detections)}'
        cv2.putText(annotated_frame, count_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return annotated_frame

    def process_frame(self, frame):
        """
        Process a single frame with compression optimizations
        """
        # Step 1: Detect ArUco marker for scale reference
        scale_factor, marker_corners, marker_center = self.detect_aruco_marker(
            frame)

        marker_info = {
            'detected': scale_factor is not None,
            'scale_factor': scale_factor,
            'corners': marker_corners,
            'center': marker_center
        }

        detections = []

        if scale_factor is None:
            # Return frame with warning if no marker detected
            warning_frame = frame.copy()
            cv2.putText(warning_frame, 'No ArUco marker detected!',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return warning_frame, detections, marker_info

        # Step 2: Detect fish using YOLO
        results = self.fish_model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Get class name if available
                    if hasattr(box, 'cls') and hasattr(self.fish_model, 'names'):
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.fish_model.names[class_id]
                    else:
                        class_name = "fish"

                    # Filter by confidence threshold
                    if confidence >= self.confidence_threshold:
                        # Estimate fish length
                        length_cm, width_cm, diagonal_cm = self.estimate_fish_length_bbox(
                            [x1, y1, x2, y2], scale_factor
                        )

                        # Store detection results
                        detection_data = {
                            'class_name': class_name,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'length_cm': length_cm,
                            'width_cm': width_cm,
                            'diagonal_cm': diagonal_cm,
                            'scale_factor': scale_factor
                        }
                        detections.append(detection_data)

        # Draw annotations
        annotated_frame = self.draw_annotations(frame, detections, marker_info)

        return annotated_frame, detections, marker_info

    def process_video_compressed(self, video_path, output_path=None,
                                 max_width=1280, max_height=720,
                                 output_fps=15, frame_skip=2,
                                 jpeg_quality=70, crf_quality=28,
                                 skip_no_marker_frames=True,
                                 show_video=True, batch_process=True):
        """
        MAIN COMPRESSION METHOD: Process video with all compression techniques

        Args:
            video_path (str): Input video path
            output_path (str): Output video path
            max_width (int): Maximum output width (COMPRESSION 1)
            max_height (int): Maximum output height (COMPRESSION 1)
            output_fps (int): Target output FPS (COMPRESSION 2)
            frame_skip (int): Process every Nth frame (COMPRESSION 2)
            jpeg_quality (int): JPEG compression quality 1-100 (COMPRESSION 3)
            crf_quality (int): H.264 CRF quality 18-30 (COMPRESSION 4)
            skip_no_marker_frames (bool): Skip frames without markers (COMPRESSION 5)
            show_video (bool): Display video during processing
            batch_process (bool): Use batch processing for memory efficiency (COMPRESSION 6)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")

        # Get original video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # COMPRESSION TECHNIQUE 1 & 2: Calculate output dimensions and FPS
        sample_frame = np.zeros(
            (original_height, original_width, 3), dtype=np.uint8)
        _, resolution_scale = self.resize_frame(
            sample_frame, max_width, max_height)

        output_width = int(original_width * resolution_scale)
        output_height = int(original_height * resolution_scale)
        actual_output_fps = min(output_fps, original_fps)

        # Display compression settings
        print(f"\n🗜️  COMPRESSION SETTINGS APPLIED:")
        print(
            f"1. Resolution Scaling: {original_width}x{original_height} → {output_width}x{output_height} ({resolution_scale:.2f}x)")
        print(
            f"2. Frame Rate Reduction: {original_fps} → {actual_output_fps} FPS")
        print(f"2. Frame Skipping: Processing every {frame_skip} frames")
        print(f"3. JPEG Quality: {jpeg_quality}%")
        print(f"4. H.264 CRF Quality: {crf_quality}")
        print(
            f"5. Dynamic Frame Skipping: {'Enabled' if skip_no_marker_frames else 'Disabled'}")
        print(
            f"6. Batch Processing: {'Enabled' if batch_process else 'Disabled'}")
        print(
            f"7. Model Optimization: {'ONNX' if self.optimized_model_path else 'PyTorch'}")

        # COMPRESSION TECHNIQUE 4: Setup H.264 video writer with CRF
        out = None
        if output_path:
            # Use H.264 codec for better compression
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, fourcc, actual_output_fps, (output_width, output_height))

        all_detections = []
        frame_count = 0
        processed_count = 0
        skipped_no_marker = 0
        start_time = time.time()

        # COMPRESSION TECHNIQUE 6: Batch processing variables
        if batch_process:
            self.frame_buffer = []
        # Load polynomial weight estimation model
        try:
            self.weight_model = joblib.load("weight_model_poly2.pkl")
            self.poly_transformer = joblib.load("poly_transformer2.pkl")
            print("✓ Weight estimation model loaded")
        except Exception as e:
            print(f"⚠ Failed to load weight model: {e}")
            self.weight_model = None
            self.poly_transformer = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # COMPRESSION TECHNIQUE 2: Frame Rate Reduction - Skip frames
                if frame_count % frame_skip != 0:
                    continue

                # COMPRESSION TECHNIQUE 1: Resolution Scaling
                resized_frame, _ = self.resize_frame(
                    frame, max_width, max_height)

                # COMPRESSION TECHNIQUE 5: Dynamic Frame Skipping - Quick marker check
                if skip_no_marker_frames:
                    scale_factor, _, _ = self.detect_aruco_marker(
                        resized_frame)
                    if scale_factor is None:
                        skipped_no_marker += 1
                        continue

                processed_count += 1

                # Process frame
                processed_frame, detections, marker_info = self.process_frame(
                    resized_frame)

                # COMPRESSION TECHNIQUE 3: JPEG Quality Compression
                if jpeg_quality < 100:
                    processed_frame = self.compress_frame(
                        processed_frame, jpeg_quality)

                # Add metadata to detections
                for detection in detections:
                    detection['frame'] = processed_count
                    detection['timestamp'] = processed_count / \
                        actual_output_fps
                    detection['original_frame'] = frame_count

                all_detections.extend(detections)

                # COMPRESSION TECHNIQUE 6: Batch Processing Buffer
                if batch_process:
                    self.frame_buffer.append(processed_frame)

                    if len(self.frame_buffer) >= self.buffer_size:
                        # Process buffer
                        for buffered_frame in self.frame_buffer:
                            if out is not None:
                                out.write(buffered_frame)

                        # Clear buffer and force garbage collection
                        self.frame_buffer.clear()
                        gc.collect()
                else:
                    # Direct write without buffering
                    if out is not None:
                        out.write(processed_frame)

                # Display video
                if show_video:
                    # Add compression info overlay
                    info_frame = processed_frame.copy()
                    compression_ratio = (
                        1 - (processed_count / frame_count)) * 100 if frame_count > 0 else 0

                    cv2.putText(info_frame, f'Frame: {frame_count} | Processed: {processed_count}',
                                (10, output_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(info_frame, f'Compression: {compression_ratio:.1f}% frames skipped',
                                (10, output_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(info_frame, f'Resolution: {output_width}x{output_height}',
                                (10, output_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow('Compressed Fish Length Detection', info_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                    elif key == ord('p'):
                        cv2.waitKey(0)

                # Progress update
                if processed_count % 50 == 0:
                    elapsed_time = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    fps_actual = processed_count / elapsed_time if elapsed_time > 0 else 0

                    print(f"Progress: {progress:.1f}% | Processed: {processed_count}/{frame_count} frames | "
                          f"FPS: {fps_actual:.1f} | Skipped (no marker): {skipped_no_marker}")

        finally:
            # COMPRESSION TECHNIQUE 6: Process remaining buffer
            if batch_process and self.frame_buffer:
                for buffered_frame in self.frame_buffer:
                    if out is not None:
                        out.write(buffered_frame)
                self.frame_buffer.clear()

            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            if show_video:
                cv2.destroyAllWindows()

            # Force garbage collection
            gc.collect()

        # Calculate and display compression statistics
        elapsed_time = time.time() - start_time
        frame_compression = (1 - (processed_count / frame_count)
                             ) * 100 if frame_count > 0 else 0

        print(f"\n📊 COMPRESSION RESULTS:")
        print(
            f"Total frames processed: {processed_count:,} out of {frame_count:,} ({frame_compression:.1f}% reduction)")
        print(f"Frames skipped (no marker): {skipped_no_marker:,}")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"Average processing FPS: {processed_count/elapsed_time:.1f}")
        print(f"Total fish detections: {len(all_detections):,}")

        # File size comparison if output was created
        if output_path and os.path.exists(output_path):
            original_size = os.path.getsize(video_path) / (1024*1024)
            compressed_size = os.path.getsize(output_path) / (1024*1024)
            size_reduction = (1 - (compressed_size / original_size)) * 100

            print(f"Original file size: {original_size:.1f} MB")
            print(f"Compressed file size: {compressed_size:.1f} MB")
            print(f"File size reduction: {size_reduction:.1f}%")

        return all_detections

    def process_webcam_compressed(self, max_width=1280, max_height=720,
                                  jpeg_quality=80, frame_skip=2):
        """
        Process live webcam feed with compression optimizations
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Could not open webcam")

        print("🎥 Starting compressed webcam processing...")
        print("Controls: 'q' to quit, 'p' to pause, 's' to save frame")

        frame_count = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam")
                    break

                frame_count += 1

                # COMPRESSION TECHNIQUE 2: Frame skipping for webcam
                if frame_count % frame_skip != 0:
                    continue

                processed_count += 1

                # COMPRESSION TECHNIQUE 1: Resolution scaling
                resized_frame, scale_factor = self.resize_frame(
                    frame, max_width, max_height)

                # Process frame
                processed_frame, detections, marker_info = self.process_frame(
                    resized_frame)

                # COMPRESSION TECHNIQUE 3: JPEG compression
                if jpeg_quality < 100:
                    processed_frame = self.compress_frame(
                        processed_frame, jpeg_quality)

                # Add real-time info overlay
                cv2.putText(processed_frame, f'Frames: {frame_count} | Processed: {processed_count}',
                            (10, processed_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(processed_frame, f'Quality: {jpeg_quality}% | Skip: 1/{frame_skip}',
                            (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(processed_frame, "Controls: 'q'=quit 'p'=pause 's'=save",
                            (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('Compressed Fish Detection - Webcam',
                           processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"webcam_frame_{processed_count:04d}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    print(f"Frame saved: {save_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _load_weight_model(self, weight_data_path: str = "../ICAR_Project/LengthWeightData.xlsx"):
        """Load and train the weight estimation model from length-weight data"""
        df = pd.read_excel(weight_data_path)
        df.columns = df.columns.str.strip().str.upper()
        self.weight_model = LinearRegression()
        self.weight_model.fit(df[['LENGHT(CM)']], df['WEIGHT(GM)'])
        print("✓ Weight estimation model trained")

    def estimate_weight_from_length(self, length_cm: float) -> float:
        """Estimate weight in grams from given length in centimeters"""
        if hasattr(self, 'poly_transformer') and hasattr(self, 'weight_model'):
            features = np.array([[length_cm]])
            poly_features = self.poly_transformer.transform(features)
            weight = self.weight_model.predict(poly_features)[0]
            return max(0.0, round(weight, 2))  # Ensure non-negative
        return 0.0

    def resize_summary_image_to_match_video(summary_img_path, target_width, target_height):
        img = cv2.imread(summary_img_path)
        if img is None:
            raise ValueError("Could not read summary image")

        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create white canvas and center the resized image
        canvas = np.ones((target_height, target_width, 3),
                         dtype=np.uint8) * 255
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h,
               x_offset:x_offset + new_w] = resized_img

        padded_path = os.path.splitext(summary_img_path)[0] + "_padded.jpg"
        cv2.imwrite(padded_path, canvas)
        return padded_path

    # Updated function to track unique fishes (max 3) and generate concise summary

    # Updated append_summary_to_video with outlier filtering and smarter clustering

    @staticmethod
    def append_summary_to_video(detections, input_video_path, output_video_path, estimator, duration_sec=2):
        import pandas as pd
        import numpy as np
        import cv2
        import subprocess
        import os

        def reduce_to_max_n_unique_fish(detections, max_n=3):
            df = pd.DataFrame(detections)
            if df.empty or not {'frame', 'length_cm', 'width_cm'}.issubset(df.columns):
                raise ValueError("Required detection keys are missing.")

            if 'Weight (g)' not in df.columns:
                df['Weight (g)'] = df['length_cm'].apply(
                    lambda l: estimator.estimate_weight_from_length(l))

            # Outlier removal based on unrealistic lengths
            df = df[(df['length_cm'] > 5) & (df['length_cm'] < 40)]

            # Smarter clustering using coarse bounding box centers
            df['bbox_center'] = df['bbox'].apply(lambda b: (
                int((b[0]+b[2])//20)*20,
                int((b[1]+b[3])//20)*20
            ))

            grouped = df.groupby('bbox_center').agg({
                'length_cm': 'max',
                'width_cm': 'max',
                'Weight (g)': 'max'
            }).reset_index()

            print(
                f"[Info] Total clusters detected: {len(grouped)} — limiting to top {max_n}")

            grouped = grouped.sort_values(by='Weight (g)', ascending=False).head(
                max_n).reset_index(drop=True)
            grouped['Fish ID'] = ['F' + str(i + 1)
                                  for i in range(len(grouped))]
            grouped.rename(columns={
                'length_cm': 'Length (cm)',
                'width_cm': 'Width (cm)',
                'Weight (g)': 'Weight (g)'
            }, inplace=True)

            return grouped

        grouped = reduce_to_max_n_unique_fish(detections, max_n=3)

        avg_weight = grouped['Weight (g)'].mean()
        total_weight = grouped['Weight (g)'].sum()

        img_height = 30 * (len(grouped) + 6)
        img = np.ones((img_height, 700, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        y = 40

        header = f"{'Fish ID':<10}{'Length (cm)':>15}{'Width (cm)':>15}{'Weight (g)':>15}"
        cv2.putText(img, header, (10, y), font, font_scale, (0, 0, 0), 2)
        y += 30

        for _, row in grouped.iterrows():
            line = f"{row['Fish ID']:<10}{row['Length (cm)']:>15.2f}{row['Width (cm)']:>15.2f}{row['Weight (g)']:>15.2f}"
            cv2.putText(img, line, (10, y), font, font_scale, (0, 0, 0), 1)
            y += 30

        cv2.putText(img, f"Total fishes detected: {len(grouped)}",
                    (10, y+20), font, font_scale, (0, 0, 255), 2)
        cv2.putText(img, f"Total weight: {total_weight:.2f} g",
                    (10, y+50), font, font_scale, (0, 0, 255), 2)
        cv2.putText(img, f"Average weight: {avg_weight:.2f} g",
                    (10, y+80), font, font_scale, (0, 0, 255), 2)

        summary_img_path = os.path.splitext(input_video_path)[
            0] + "_summary.jpg"
        cv2.imwrite(summary_img_path, img)

        cap = cv2.VideoCapture(input_video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        summary_img_path_padded = CompressedFishLengthEstimator.resize_summary_image_to_match_video(
            summary_img_path, video_width, video_height
        )

        cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-loop', '1',
            '-t', str(duration_sec),
            '-framerate', '30',
            '-i', summary_img_path_padded,
            '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_video_path
        ]

        subprocess.run(cmd, check=True)
        print(f"✅ Final video with summary saved to: {output_video_path}")

    @staticmethod
    def summarize_detections_and_generate_image(detections, estimator, input_video_path, duration_sec=2):
        df = pd.DataFrame(detections)
        if not {'frame', 'length_cm', 'width_cm'}.issubset(df.columns):
            raise ValueError("Required keys missing in detections.")

        # Add weight column
        df['Weight (g)'] = df['length_cm'].apply(
            estimator.estimate_weight_from_length)

        # Estimate number of distinct fishes using dynamic mode logic
        frame_fish_counts = df.groupby('frame').size().tolist()
        count_freq = Counter(frame_fish_counts)
        total_frames = len(frame_fish_counts)
        threshold = 0.6 * total_frames

        sorted_counts = sorted(
            count_freq.items(), key=lambda x: (-x[1], -x[0]))
        majority_count = next(
            (c for c, f in sorted_counts if f >= threshold), max(frame_fish_counts))

        # Select top-N fish based on average presence (length_cm descending as proxy)
        grouped = df.groupby('frame').agg({
            'length_cm': 'mean',
            'width_cm': 'mean',
            'Weight (g)': 'mean'
        }).reset_index()

        # For individual rows, sort by highest average length
        fish_stats = df.groupby(['frame']).agg({
            'length_cm': 'max',
            'width_cm': 'max',
            'Weight (g)': 'max'
        }).sort_values(by='length_cm', ascending=False).head(majority_count).reset_index()

        fish_stats['Shrimp ID'] = ['S' + str(i + 1)
                                   for i in range(len(fish_stats))]
        fish_stats = fish_stats[['Shrimp ID',
                                'length_cm', 'width_cm', 'Weight (g)']]

        total_weight = fish_stats["Weight (g)"].sum()
        average_weight = fish_stats["Weight (g)"].mean()

        # Generate summary image
        img_height = 30 * (len(fish_stats) + 6)
        img = np.ones((img_height, 720, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        line_height = 30
        y = 40

        cv2.putText(img, f"Total Fishes (n): {majority_count}",
                    (10, y), font, font_scale, (0, 0, 0), 2)
        y += line_height
        cv2.putText(img, f"Total Estimated Weight: {total_weight:.2f} g", (
            10, y), font, font_scale, (0, 0, 0), 2)
        y += line_height
        cv2.putText(img, f"Average Weight: {average_weight:.2f} g",
                    (10, y), font, font_scale, (0, 0, 0), 2)
        y += line_height * 2

        header = f"{'Shrimp ID':<10}{'Length (cm)':>15}{'Width (cm)':>15}{'Weight (g)':>15}"
        cv2.putText(img, header, (10, y), font, font_scale, (0, 0, 0), 2)
        y += line_height

        for _, row in fish_stats.iterrows():
            line = f"{row['Shrimp ID']:<10}{row['length_cm']:>15.2f}{row['width_cm']:>15.2f}{row['Weight (g)']:>15.2f}"
            cv2.putText(img, line, (10, y), font, font_scale, (0, 0, 0), 1)
            y += line_height

        # Save image
        summary_img_path = os.path.splitext(input_video_path)[
            0] + "_summary_fixed.jpg"
        cv2.imwrite(summary_img_path, img)
        return summary_img_path, majority_count


def main():
    """
    Main function with compression options
    """
    parser = argparse.ArgumentParser(
        description='Compressed Fish Length Estimation using YOLO and ArUco')

    # Model and input
    parser.add_argument('--model', required=True,
                        help='Path to YOLO fish detection model (.pt file)')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument(
        '--output', help='Path to save compressed output video')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam instead of video file')

    # Basic parameters
    parser.add_argument('--marker-size', type=float, default=5.0,
                        help='ArUco marker size in cm (default: 5.0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5)')

    # COMPRESSION PARAMETERS
    # Compression Technique 1: Resolution Scaling
    parser.add_argument('--max-width', type=int, default=1280,
                        help='Maximum output width (default: 1280)')
    parser.add_argument('--max-height', type=int, default=720,
                        help='Maximum output height (default: 720)')

    # Compression Technique 2: Frame Rate Reduction
    parser.add_argument('--output-fps', type=int, default=15,
                        help='Target output FPS (default: 15)')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every Nth frame (default: 2)')

    # Compression Technique 3: JPEG Quality
    parser.add_argument('--jpeg-quality', type=int, default=70,
                        help='JPEG quality 1-100 (default: 70)')

    # Compression Technique 4: H.264 CRF
    parser.add_argument('--crf-quality', type=int, default=28,
                        help='H.264 CRF quality 18-30 (default: 28)')

    # Compression Technique 5: Dynamic Frame Skipping
    parser.add_argument('--skip-no-marker', action='store_true', default=True,
                        help='Skip frames without ArUco markers (default: True)')

    # Compression Technique 6: Batch Processing
    parser.add_argument('--no-batch', action='store_true',
                        help='Disable batch processing')

    # Compression Technique 7: Model Optimization
    parser.add_argument('--no-optimize-model', action='store_true',
                        help='Disable model optimization to ONNX')

    # Display options
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display video during processing')

    # Compression presets
    parser.add_argument('--preset', choices=['quality', 'balanced', 'compression'],
                        help='Use compression preset (overrides individual settings)')

    args = parser.parse_args()

    # Apply compression presets
    if args.preset == 'quality':
        args.max_width, args.max_height = 1920, 1080
        args.output_fps = 30
        args.frame_skip = 1
        args.jpeg_quality = 90
        args.crf_quality = 20
        print("🎯 Using QUALITY preset")

    elif args.preset == 'balanced':
        args.max_width, args.max_height = 1280, 720
        args.output_fps = 20
        args.frame_skip = 2
        args.jpeg_quality = 75
        args.crf_quality = 25
        print("⚖️ Using BALANCED preset")

    elif args.preset == 'compression':
        args.max_width, args.max_height = 854, 480
        args.output_fps = 10
        args.frame_skip = 3
        args.jpeg_quality = 50
        args.crf_quality = 30
        print("🗜️ Using MAXIMUM COMPRESSION preset")

    # Validate arguments
    if not args.webcam and not args.video:
        print("❌ Error: Either --video or --webcam must be specified")
        return

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return

    if args.video and not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return

    # Initialize compressed fish length estimator
    try:
        estimator = CompressedFishLengthEstimator(
            model_path=args.model,
            marker_size_cm=args.marker_size,
            confidence_threshold=args.confidence,
            use_optimized_model=not args.no_optimize_model
        )
    except Exception as e:
        print(f"❌ Error initializing estimator: {e}")
        return

    # Process video or webcam with compression
    try:
        if args.webcam:
            estimator.process_webcam_compressed(
                max_width=args.max_width,
                max_height=args.max_height,
                jpeg_quality=args.jpeg_quality,
                frame_skip=args.frame_skip
            )
        else:
            detections = estimator.process_video_compressed(
                video_path=args.video,
                output_path=args.output,
                max_width=args.max_width,
                max_height=args.max_height,
                output_fps=args.output_fps,
                frame_skip=args.frame_skip,
                jpeg_quality=args.jpeg_quality,
                crf_quality=args.crf_quality,
                skip_no_marker_frames=args.skip_no_marker,
                show_video=not args.no_display,
                batch_process=not args.no_batch
            )

            # Print summary statistics
            if detections:
                lengths = [d['length_cm'] for d in detections]
                print(f"\n📈 FISH MEASUREMENT STATISTICS:")
                print(f"Average fish length: {np.mean(lengths):.2f} cm")
                print(f"Min fish length: {np.min(lengths):.2f} cm")
                print(f"Max fish length: {np.max(lengths):.2f} cm")
                print(f"Standard deviation: {np.std(lengths):.2f} cm")
                print(f"Total measurements: {len(lengths):,}")

                # ✨ Append 2-second summary
                if args.output:
                    output_with_summary = os.path.splitext(args.output)[
                        0] + "_final.mp4"
                    CompressedFishLengthEstimator.append_summary_to_video(
                        detections=detections,
                        input_video_path=args.output,
                        output_video_path=output_with_summary,
                        estimator=estimator,
                        duration_sec=2
                    )

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
