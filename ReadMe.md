For In depth understanding of the project and code,
Please refer to Project_Report.pdf and Code_Report.pdf.

# 🐟 Fish Metrics Estimation Pipeline

This repository hosts the full implementation of an AI-based computer vision pipeline designed to detect fish in videos, estimate their physical dimensions (length, width), and predict their weight using regression models. The system is modular and scalable, with future provisions for fish behavior analysis and visual freshness estimation.

---

## 📌 Project Highlights

- 🎯 **Object Detection** using YOLOv12s (single species - White Leg Shrimp)
- 📏 **Length/Width Estimation** using ArUco marker-based scaling
- ⚖️ **Weight Prediction** via polynomial regression
- 🎞️ **Multi-level Video Compression** for efficient processing
- 📊 **Summary Overlay** of processing stats on final output
- 🧱 **Modular design** for easy model and feature upgrades

---

## 🧠 Project Motivation

The fisheries sector often relies on manual fish measurement and grading techniques. This project aims to automate and streamline this process using AI—improving accuracy, saving time, and enabling further insights such as fish behavior and freshness from post-harvest videos.

---

## 📁 Repository Structure

```bash
project_root/
├── Build/
│ ├── FishEstimatorWithWeight_Modified.py # Main pipeline script
│ ├── best.pt # YOLOv12s trained model
│ ├── best_optimized.onnx # Auto-converted ONNX model
│ ├── poly_transformer2.pkl # Polynomial feature generator
│ ├── weight_model_poly2.pkl # Final regression model
├── Detection-Model-Training/
│ ├── fish_detection_train.py # Training script
│ ├── yolo12s.pt # Detection model (trained)
│ └── training_data.yolov12/ # Labeled dataset
├── Weight-Estimation-LR_model/
│ ├── LengthWeightData.xlsx # Input dataset for regression
│ ├── weight_model_poly2.pkl # Output model file
├── Training_data-videos/ # Raw input videos (ignored)
├── Final_outputs/ # Annotated results
├── Project_Report.pdf
├── Code_Report.pdf
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YashSarang/fish-estimation-pipeline.git
cd fish-estimation-pipeline
```

### 2. Install Dependencies

Use a Python environment (Python 3.8+) and install dependencies:

```bash
pip install -r requirements.txt
```

Make sure FFmpeg is installed and available in your system path.

### 3. Run the Pipeline

```bash
cd Build
python FishEstimatorWithWeight_Modified.py --model best.pt --output ../Final_outputs/output_video.mp4 --video ../Training_data-videos/input_video.mp4

Example:
python FishEstimatorWithWeight_Modified.py --model best.pt --output ../Final_outputs/output_video.mp4 --video ../Training_data-videos/C1.mp4
```

### ⚙️ Command-Line Arguments

```bash
Argument	        Type	Description	                        Default
--video	                str	Input video file path	                Required
--output	        str	Output video file path	                Optional
--max_width	        int	Width of resized output	                1280
--max_height	        int	Height of resized output	        720
--output_fps	        int	FPS of output video	                15
--frame_skip	        int	Process every Nth frame	                2
--jpeg_quality	        int	JPEG compression quality (1–100)	70
--crf_quality	        int	H.264 CRF comp(lower is better quality)	28
--skip_no_marker_frames	bool	Skip frames without ArUco marker	True
--show_video	        bool	Show video preview while processing	True
--batch_process	        bool	Enable memory-efficient processing	True
```

---

## 📈 Model Performance

```bash
Model	mAP@50	Precision	Recall
YOLOv12	82.6%	77.1%	83.8%
YOLOv8	79.4%	74.0%	85.6%

Weight Estimation (Degree 2 Polynomial Regression):
Weight ≈ 0.0129 + 1.0643 × Length + 0.00008 × Length²
MSE = 0.00085, R² = 0.99925
```

---

## 🧪 Sample Output

Annotated output videos include:

- Fish bounding boxes,
- Real-world length and weight,
- Summary clip all the fishes at the end (avg, total, individual weights, lengths and widths)

---

## 🚧 Limitations and Future Work

No object tracking yet (fish IDs not persistent across frames)
Video quality significantly impacts performance
RoboFlow-labeled data requires manual review
Behavior and freshness analytics are in future scope

---

## 📅 Roadmap

- Fish detection with YOLOv12
- ArUco-based length scaling
- Polynomial regression for weight estimation
- Output video with overlays
- Add object tracking (Deep SORT / ByteTrack)
- Integrate behavior trajectory analysis
- Visual freshness estimation module
- Web UI for video upload and download

---

## 🧑‍💻 Contributor

Yash Sarang ~ M.S. by Research – IIT Bombay.
yashsarang.com
📧 Contact for collaborations, research extensions, or implementation help.

---

## 📄 References

YOLOv12: Vision Transformer-based detection
OpenCV ArUco: Marker tracking and calibration
Scikit-Learn: Regression modeling
FFmpeg: Video encoding

---

## 📜 License

This project is licensed under the MIT License. See LICENSE for details.

---
