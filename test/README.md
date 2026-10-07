# AI & Tracking Model Evaluation for Bank CCTV Surveillance

This directory tests and benchmarks candidate models and architectures against realistic bank surveillance constraints:
1. Low-resolution CCTV feeds (noisy, compressed, variable lighting).
2. Long-duration frame-by-frame processing without memory leaks / context explosion.
3. Cross-camera / Cross-video Person Re-Identification (ReID) and track continuity.
4. Real-time behavior and anomaly extraction for the 3 target bank use cases:
   - Unauthorized Zone Breach
   - After-Hours Intrusion
   - Fall / Incapacitation Detection

## Model Candidates Under Evaluation:
- **Pipeline A (Hybrid Edge-Grade - Recommended)**:
  - Detector & Pose: `YOLOv8n-pose` / `YOLO11n-pose`
  - Local Tracker: `ByteTrack` / `BoT-SORT`
  - Cross-Camera ReID: `OSNet` (Omni-Scale Feature Learning) + Spatio-Temporal Transition Matrix
  - Memory Footprint: Constant $O(1)$ memory per stream (~300MB RAM, 60+ FPS on Apple Silicon / CPU).
- **Pipeline B (Heavyweight VLM / End-to-End)**:
  - Models: Qwen2-VL / LLaVA / Florence-2
  - Bottleneck: Frame throughput (< 1-3 FPS), cannot handle continuous 30 FPS multi-camera CCTV streams locally without massive GPU clusters.
