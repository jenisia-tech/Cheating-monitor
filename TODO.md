# Hacknex HNX26PSI07 - Exam Hall Surveillance Roadmap & TODO List

**Problem Statement**: `HNX26PSI07: Autonomous Vision & Behaviour Understanding`  
**Domain**: Academic Examination Halls & Proctoring Video Analysis  
**Reference Document**: `Images for reference/HNX26PSI07.pdf` (Pages 3 & 11)  
**Host Target**: MacBook Pro (Dual-Core Intel Core i5, 8 GB RAM) — $O(1)$ RAM Hygiene

---

## 1. Core Vision & Multi-Camera Tracking Pipeline
- [x] **Lightweight Model Caching & Local Containment**: Downloaded `yolo11n-pose.pt` and `yolov8n-pose.pt` strictly into `test/models/`.
- [x] **Frame-by-Frame Person Detection**: Integrated YOLO-Pose keypoint extractor (17 COCO landmarks) at ~20 FPS locally on CPU.
- [x] **Persistent Multi-Object Tracking**: Integrated `ByteTrack` (`lap` solver) for persistent track assignment.
- [x] **Memory Hygiene & Garbage Collection**: Explicit `gc.collect()` and track-state clearing after every video.

---

## 2. Refined 3-Case Exam Hall Behaviour Engine
- [x] **Desk-Anchored Spatial Grid & Identity Mapping**:
  - [x] 2D Desk Polygon Map (`Desk_01`, `Desk_02`, ... `Desk_0N`, `Invigilator_Podium`, `Aisle_Zone`) with Dynamic Desk Anchoring for unmapped cameras.
  - [x] Desk-anchored candidate numbering (`Candidate_Desk_01`, `Candidate_Desk_02`) eliminating cross-angle uniform/clothing confusion.
- [x] **Case 1: Unauthorized Desk Abandonment / Wandering**:
  - [x] Ground-contact foot coordinate testing (`cv2.pointPolygonTest`) out of assigned desk polygon into aisles for $> 5$s ($60$ frames).
- [x] **Case 2: Candidate Unresponsive / Slump / Medical Emergency**:
  - [x] Head-down collapse on desk/floor ($\text{torso angle} < 25^\circ$) + complete stillness for $> 20$s ($240$ frames).
- [x] **Case 3: Repeated Inter-Desk Peeking / Malpractice Attempt**:
  - [x] Single glance/stretch ignored (prevents false alarms).
  - [x] Alerts trigger only when inter-desk leaning/lateral head turn towards an adjacent desk is **repeated ($\ge 2$ times within a $60$-second window)**.

---

## 3. Evidence, Explainability & Event Dispatcher (Judging Rules)
- [x] **Interval-Based Frame Logging**: Exact `[start_frame, end_frame]` grouping for every incident.
- [x] **Millisecond Timestamp Tracking**: `start_timestamp` and `end_timestamp` formatted in `HH:MM:SS.mmm`.
- [x] **Entity Attribution**: Explicit candidate desk binding (`Candidate_Desk_04`).
- [x] **Exam-Tailored Natural Language Reasons**: Generates detailed explanations (e.g. *"Candidate at Desk_02 exhibited repeated peeking (3 episodes in 45s) towards Desk_03"*).
- [x] **Automated JSON Export**: Output structured reports saved to `test/outputs/<video_name>_incidents.json`.

---

## 4. Test Footage & Multi-Angle Verification
- [x] **Asset Downloader**: `test/download_samples.py` saving all assets strictly inside `test/`.
- [x] **Exam Hall Benchmark Script**: `test/exam_analyzer.py` supporting single & multi-camera feeds with auto-desk anchoring.
- [x] **Multi-Angle Synchronizer**: `test/multi_camera_exam_sync.py` fusing front and rear camera perspectives.
- [x] **Benchmark Validation**: Verified on real-life CCTV exam footage (`irl_exam_hall_cctv.mp4`, `real_life_indoor_cctv.mp4`) and dual-camera synthetic benchmarks.

---

## 5. Security & Proctoring Command Center Dashboard (UI + Backend)
- [ ] **Backend Web Server (FastAPI / Python 3.11)**:
  - [ ] Video upload & analysis executor with WebSocket streaming.
  - [ ] Desk polygon configuration API (`/api/desks`).
  - [ ] Incident retrieval API (`/api/incidents`).
- [ ] **Frontend UI (React / Vite - Dark Glassmorphic Theme)**:
  - [ ] **Multi-Angle Video Player**: Timeline scrubber with colored incident markers.
  - [ ] **Interactive Desk Grid Overlay**: Click-to-draw/edit desk polygons over the canvas.
  - [ ] **Live Incident Feed**: Filterable by Severity (*Critical*, *High*, *Warning*) and Desk Number.
  - [ ] **Evidence Inspector Drawer**: Snapshots, exact timestamps `[from frame x to frame y]`, confidence, and plain-English reasons.
  - [ ] **One-Click Export**: Download structured JSON reports and annotated MP4 video clips.

---

## 6. Submission Deliverables (PDF Page 3 Guidelines)
- [ ] **Public Git Repository Setup**: Root `.gitignore` for clean packaging.
- [ ] **Comprehensive Setup & Reproducibility `README.md`**: Architecture diagrams, setup instructions, CLI execution guides, and scope documentation.
