# Cheating Monitor: Autonomous Vision & Behaviour Understanding for Examination Halls

[![Challenge](https://img.shields.io/badge/Hackathon-Karunya%20Hacknex%202026-blue)](Images%20for%20reference/HNX26PSI07.pdf)
[![Problem Statement](https://img.shields.io/badge/Problem%20ID-HNX26PSI07-orange)](Images%20for%20reference/HNX26PSI07.pdf)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11%20Pose-purple)](https://github.com/ultralytics/ultralytics)

Autonomous CCTV video analytics engine designed for academic examination halls, test centers, and proctoring environments. Evaluates multi-camera CCTV feeds frame-by-frame using spatial landmark estimation, persistent object tracking, and geometric behavioral rule state machines without heavy model retraining overhead.

---

## Key Features & Capabilities

1. **Desk-Anchored Spatial Identification**:
   - Assigns unambiguous spatial coordinates (`Candidate_Desk_01`, `Candidate_Desk_02`, etc.) to candidates based on 2D seating polygons.
   - Eliminates multi-camera tracking ambiguities caused by identical school/college uniforms and varying viewing angles.
   - Includes **Dynamic Desk Anchoring** for automatic registration on real-world CCTV cameras of arbitrary resolution.

2. **The 3 Core Proctoring Behavioral Rules**:
   - **`UNAUTHORIZED_DESK_ABANDONMENT`**: Detects when a candidate vacates their assigned desk polygon into aisles/unauthorized areas for $> 5.0\text{s}$ ($60$ frames @ 12 FPS).
   - **`CANDIDATE_UNRESPONSIVE_SLUMP`**: Detects medical emergencies, fainting, or slumping flat on desk/floor ($\text{torso angle} < 25^\circ$) with complete stillness $> 20.0\text{s}$.
   - **`REPEATED_INTER_DESK_PEEKING`**: Monitors lateral leaning/turning towards adjacent desks. **Filters out single innocent glances/stretches** and triggers high-severity alerts **strictly upon $\ge 2$ repeated episodes within a $60$-second window**.

3. **Multi-Camera Perspective Fusion**:
   - Integrates Front and Rear camera feeds (`test/multi_camera_exam_sync.py`), cross-verifying candidate behavior across angles to eliminate occlusions and assign `HIGH (Multi-Camera Verified)` confidence scores.

4. **Hardware Optimized for Local Execution**:
   - Built to run efficiently on low-compute hardware (tested on Dual-Core Intel i5, 8 GB RAM) at ~20 FPS using YOLO11-Nano Pose, CPU thread pinning, and active memory garbage collection.

---

## Directory Structure

```text
Hacknex/
├── test/
│   ├── exam_analyzer.py                 # Core Exam Hall Vision & Behaviour Engine
│   ├── multi_camera_exam_sync.py        # Multi-Angle Feed Synchronizer & Fusion
│   ├── download_samples.py              # Asset downloader for models & videos
│   ├── generate_exam_hall_benchmark_video.py # Synthetic dual-camera test benchmark generator
│   ├── models/                          # AI weights (yolo11n-pose.pt)
│   ├── videos/                          # Raw test & IRL CCTV recordings
│   └── outputs/                         # Incident JSON logs and annotated MP4 videos
├── backup/                              # Archived legacy prototype files
├── rules                                # System development rules & constraints
├── logs                                 # Dual-instance execution and progress logs
├── TODO.md                              # Project roadmap & milestone tracking
└── README.md                            # Main project documentation
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jenisia-tech/Cheating-monitor.git
   cd Cheating-monitor
   ```

2. **Set up Python 3.11 virtual environment**:
   ```bash
   python3.11 -m venv central_env
   source central_env/bin/activate
   pip install torch torchvision ultralytics opencv-python lap yt-dlp
   ```

3. **Download Model Weights & Sample Videos**:
   ```bash
   python3.11 test/download_samples.py
   ```

---

## Quick Start & Usage

### 1. Analyze Single-Camera Exam Hall Footage
Run behavioral analysis on any CCTV recording (or real-life exam hall video):
```bash
python3.11 test/exam_analyzer.py --video irl_exam_hall_cctv.mp4 --stride 2
```

### 2. Run Multi-Camera Synchronized Proctoring
Run dual-angle cross-verification (Front + Rear cameras):
```bash
python3.11 test/multi_camera_exam_sync.py \
  --cam1 test/videos/exam_hall_cam1_front.mp4 \
  --cam2 test/videos/exam_hall_cam2_rear.mp4
```

### 3. Output Incident JSON Schema
Reports are structured with interval frame ranges, millisecond timestamps, and natural language explanations:
```json
{
  "event_type": "REPEATED_INTER_DESK_PEEKING",
  "severity": "HIGH",
  "candidate_id": "Candidate_Desk_02",
  "assigned_desk": "Desk_02",
  "camera_source": "Cam_01_ExamHall",
  "start_frame": 180,
  "start_timestamp": "10:00:15.000",
  "end_frame": 240,
  "end_timestamp": "10:00:20.000",
  "duration_seconds": 5.0,
  "reason": "Candidate_Desk_02 exhibited repeated inter-desk peeking (2 distinct episodes within 60s) towards Desk_01 (Cheating/Malpractice attempt)."
}
```

---

## Problem Statement Reference
- **Event**: Karunya Hacknex Qualifier 2026
- **Problem Statement ID**: `HNX26PSI07: Autonomous Vision & Behaviour Understanding`
- **Reference**: [`Images for reference/HNX26PSI07.pdf`](Images%20for%20reference/HNX26PSI07.pdf) (Page 11)
