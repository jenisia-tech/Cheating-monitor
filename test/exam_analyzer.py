"""
Exam Hall Autonomous Vision & Proctoring Anomaly Analyzer
HNX26PSI07: Autonomous Vision & Behaviour Understanding
Evaluates CCTV video files frame-by-frame using YOLO-Pose and ByteTrack.

Anomalies Evaluated:
  1. UNAUTHORIZED_DESK_ABANDONMENT: Candidate leaves assigned desk polygon for > 5s.
  2. CANDIDATE_UNRESPONSIVE_SLUMP: Candidate collapses or slumps flat on desk/floor (> 20s stillness).
  3. REPEATED_INTER_DESK_PEEKING: Leaning/turning towards adjacent desk (Alerts ONLY upon >= 2 repeated episodes within 60s).

Features:
  - Desk-Anchored Spatial Identification (eliminates uniform/angle ambiguities).
  - Multi-Camera / Single-Camera Ingestion Support.
  - Interval-based Frame Logging & Explainable JSON Reports.
  - Local Containment & Dual-Core Intel i5 Optimization (Rules 5, 7, 8).
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, List, Any, Tuple, Optional
from collections import deque

# Strict workspace containment (Rules 7 & 8)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(TEST_DIR)
MODELS_DIR = os.path.join(TEST_DIR, "models")
VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
OUTPUTS_DIR = os.path.join(TEST_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

os.environ["TORCH_HOME"] = MODELS_DIR
os.environ["YOLO_CONFIG_DIR"] = MODELS_DIR

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ExamHallAnomalyAnalyzer:
    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        desk_zones_config: Dict[str, List[List[int]]] = None,
        simulated_start_time: str = "10:00:00",
        fps_override: float = None,
        abandonment_seconds_threshold: float = 5.0,
        slump_seconds_threshold: float = 20.0,
        slump_torso_angle_threshold: float = 25.0,
        peeking_episode_seconds_threshold: float = 1.5,
        peeking_repeat_count_threshold: int = 2,
        peeking_window_seconds: float = 60.0,
        frame_stride: int = 1,
    ):
        self.model_name = model_name
        self.simulated_start_time = simulated_start_time
        self.fps_override = fps_override
        self.abandonment_seconds_threshold = abandonment_seconds_threshold
        self.slump_seconds_threshold = slump_seconds_threshold
        self.slump_torso_angle_threshold = slump_torso_angle_threshold
        self.peeking_episode_seconds_threshold = peeking_episode_seconds_threshold
        self.peeking_repeat_count_threshold = peeking_repeat_count_threshold
        self.peeking_window_seconds = peeking_window_seconds
        self.frame_stride = max(1, frame_stride)

        # Optimize PyTorch CPU threads for 2-core Intel i5 (Rule 5)
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass

        # Default Exam Hall Desk Layout (Polygonal 2D desk grids)
        self.desk_zones = desk_zones_config or {
            "Desk_01": [[80, 100], [280, 100], [280, 320], [80, 320]],
            "Desk_02": [[320, 100], [520, 100], [520, 320], [320, 320]],
            "Desk_03": [[80, 380], [280, 380], [280, 600], [80, 600]],
            "Desk_04": [[320, 380], [520, 380], [520, 600], [320, 600]],
            "Invigilator_Podium": [[600, 80], [740, 80], [740, 250], [600, 250]],
        }

        self.model = None

    def _load_model(self):
        if self.model is None:
            if YOLO is None:
                raise RuntimeError("Ultralytics YOLO is not installed. Please install it in central_env.")
            
            target_model_path = self.model_name
            if not os.path.exists(target_model_path):
                candidate = os.path.join(MODELS_DIR, self.model_name)
                if os.path.exists(candidate):
                    target_model_path = candidate

            print(f"[INFO] Loading YOLO-Pose model for Exam Surveillance: {target_model_path}...")
            try:
                self.model = YOLO(target_model_path)
            except Exception as e:
                fallback = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
                print(f"[WARN] Failed to load {target_model_path} ({e}), falling back to {fallback}")
                self.model = YOLO(fallback)

    def _frame_to_timestamp(self, frame_idx: int, fps: float) -> str:
        total_seconds = frame_idx / max(fps, 1.0)
        h, m, s = map(int, self.simulated_start_time.split(":"))
        start_total = h * 3600 + m * 60 + s
        curr_total = start_total + total_seconds
        
        cur_h = int((curr_total // 3600) % 24)
        cur_m = int((curr_total % 3600) // 60)
        cur_s = curr_total % 60
        return f"{cur_h:02d}:{cur_m:02d}:{cur_s:06.3f}"

    def _check_point_in_polygon(self, point: Tuple[float, float], polygon: List[List[int]]) -> bool:
        pts = np.array(polygon, dtype=np.int32)
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return dist >= 0

    def _calculate_torso_angle(self, keypoints: np.ndarray, bbox: List[float]) -> Tuple[float, bool]:
        """Calculates torso angle relative to horizontal ground (90 deg = upright, < 25 deg = slumped/fallen)."""
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        aspect_ratio = height / width

        torso_angle = 90.0
        has_kpts = False

        if keypoints is not None and len(keypoints) >= 17:
            ls, rs = keypoints[5][:2], keypoints[6][:2]
            lh, rh = keypoints[11][:2], keypoints[12][:2]

            if (ls[0] > 0 or rs[0] > 0) and (lh[0] > 0 or rh[0] > 0):
                shoulder_mid = (ls + rs) / 2.0 if (ls[0] > 0 and rs[0] > 0) else (ls if ls[0] > 0 else rs)
                hip_mid = (lh + rh) / 2.0 if (lh[0] > 0 and rh[0] > 0) else (lh if lh[0] > 0 else rh)

                dx = hip_mid[0] - shoulder_mid[0]
                dy = hip_mid[1] - shoulder_mid[1]

                angle_rad = math.atan2(abs(dy), abs(dx))
                torso_angle = math.degrees(angle_rad)
                has_kpts = True

        if not has_kpts and aspect_ratio < 0.7:
            torso_angle = 20.0

        return torso_angle, has_kpts

    def process_video(
        self,
        video_path: str,
        output_json_path: str = None,
        save_annotated_video: bool = True,
        output_video_path: str = None,
        camera_id: str = "Cam_01_ExamHall",
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processes an exam recording and produces explainable incident records.
        """
        self._load_model()

        # Auto-resolve video from VIDEOS_DIR
        resolved_video_path = video_path
        if not os.path.exists(resolved_video_path):
            candidate = os.path.join(VIDEOS_DIR, video_path)
            if os.path.exists(candidate):
                resolved_video_path = candidate
            else:
                raise FileNotFoundError(f"Input exam video not found: {video_path}")

        cap = cv2.VideoCapture(resolved_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 12.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frames threshold calculations
        abandonment_frames_limit = int(self.abandonment_seconds_threshold * fps)
        slump_frames_limit = int(self.slump_seconds_threshold * fps)
        peeking_episode_frames_limit = int(self.peeking_episode_seconds_threshold * fps)

        video_writer = None
        if save_annotated_video:
            if not output_video_path:
                base = os.path.splitext(os.path.basename(video_path))[0]
                output_video_path = os.path.join(OUTPUTS_DIR, f"{base}_exam_annotated.mp4")
            # Try mp4v with fallback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"[INFO] Analyzing Exam Video: {video_path} ({total_frames} frames @ {fps:.1f} FPS, {frame_width}x{frame_height})")

        # Entity State Tracking:
        # track_id -> assigned_desk_name
        track_to_desk: Dict[int, str] = {}
        # track_id -> deque of recent positions
        track_history: Dict[int, deque] = {}
        # track_id -> consecutive frames out of assigned desk
        out_of_desk_counters: Dict[int, int] = {}
        # track_id -> consecutive frames slumped
        slump_counters: Dict[int, int] = {}
        # track_id -> list of peeking episode timestamps [(start_time_sec, end_time_sec, target_desk)]
        peeking_episodes_history: Dict[int, List[Dict[str, Any]]] = {}
        # track_id -> current ongoing peeking frames counter
        current_peeking_counters: Dict[int, Tuple[int, str]] = {} # track_id -> (consecutive_frames, target_desk)

        # Active incident intervals: (entity_id, anomaly_type) -> current event dict
        active_intervals: Dict[Tuple[str, str], Dict[str, Any]] = {}
        completed_events: List[Dict[str, Any]] = []

        frame_idx = 0

        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_stride != 0:
                frame_idx += 1
                continue

            current_timestamp = self._frame_to_timestamp(frame_idx, fps)
            current_time_sec = frame_idx / max(fps, 1.0)

            # YOLO Pose + ByteTrack
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            current_active_incident_keys = set()
            frame_detections = []

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                keypoints_data = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

                # First pass: Bind tracks to desks if not already bound
                for i, track_id in enumerate(track_ids):
                    bbox = boxes[i].tolist()
                    foot_point = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
                    center_point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                    kpts = keypoints_data[i] if keypoints_data is not None else None

                    if track_id not in track_to_desk:
                        # Find which desk polygon this person started in
                        assigned = None
                        for desk_name, poly in self.desk_zones.items():
                            if self._check_point_in_polygon(foot_point, poly) or self._check_point_in_polygon(center_point, poly):
                                assigned = desk_name
                                break
                        
                        if assigned is None:
                            assigned = f"Desk_Candidate_{track_id:02d}"
                            # Dynamically anchor a desk zone around initial seated position
                            pad_x = max(35.0, (bbox[2] - bbox[0]) * 0.45)
                            pad_y = max(35.0, (bbox[3] - bbox[1]) * 0.45)
                            cx, cy = center_point
                            self.desk_zones[assigned] = [
                                [int(cx - pad_x), int(cy - pad_y)],
                                [int(cx + pad_x), int(cy - pad_y)],
                                [int(cx + pad_x), int(cy + pad_y + 15)],
                                [int(cx - pad_x), int(cy + pad_y + 15)]
                            ]

                        track_to_desk[track_id] = assigned
                        track_history[track_id] = deque(maxlen=60)
                        out_of_desk_counters[track_id] = 0
                        slump_counters[track_id] = 0
                        peeking_episodes_history[track_id] = []
                        current_peeking_counters[track_id] = (0, "")

                    assigned_desk = track_to_desk[track_id]
                    candidate_entity_id = f"Candidate_{assigned_desk}"

                    track_history[track_id].append({
                        "frame": frame_idx,
                        "center": center_point,
                        "foot": foot_point,
                        "bbox": bbox
                    })

                    frame_detections.append({
                        "track_id": int(track_id),
                        "entity_id": candidate_entity_id,
                        "assigned_desk": assigned_desk,
                        "bbox": bbox,
                        "foot": foot_point,
                        "center": center_point,
                        "kpts": kpts
                    })

                # Second pass: Evaluate 3 Core Exam Anomalies
                for det in frame_detections:
                    t_id = det["track_id"]
                    cand_id = det["entity_id"]
                    assigned_desk = det["assigned_desk"]
                    foot_point = det["foot"]
                    bbox = det["bbox"]
                    kpts = det["kpts"]

                    # Skip invigilator from student rules
                    if "Invigilator" in assigned_desk:
                        continue

                    # --- CASE 1: UNAUTHORIZED DESK ABANDONMENT / WANDERING ---
                    is_in_assigned_desk = False
                    if assigned_desk in self.desk_zones:
                        is_in_assigned_desk = self._check_point_in_polygon(foot_point, self.desk_zones[assigned_desk])

                    if not is_in_assigned_desk:
                        out_of_desk_counters[t_id] += self.frame_stride
                    else:
                        out_of_desk_counters[t_id] = max(0, out_of_desk_counters[t_id] - self.frame_stride * 2)

                    event_key_1 = (cand_id, "UNAUTHORIZED_DESK_ABANDONMENT")
                    if out_of_desk_counters[t_id] >= abandonment_frames_limit:
                        current_active_incident_keys.add(event_key_1)
                        if event_key_1 not in active_intervals:
                            active_intervals[event_key_1] = {
                                "event_type": "UNAUTHORIZED_DESK_ABANDONMENT",
                                "severity": "HIGH",
                                "candidate_id": cand_id,
                                "assigned_desk": assigned_desk,
                                "camera_source": camera_id,
                                "start_frame": frame_idx - out_of_desk_counters[t_id],
                                "start_timestamp": self._frame_to_timestamp(frame_idx - out_of_desk_counters[t_id], fps),
                                "end_frame": frame_idx,
                                "end_timestamp": current_timestamp,
                                "duration_seconds": round(out_of_desk_counters[t_id] / fps, 2),
                                "reason": f"{cand_id} abandoned assigned {assigned_desk} and is wandering in aisles/unauthorized areas for > {self.abandonment_seconds_threshold}s."
                            }
                        else:
                            active_intervals[event_key_1]["end_frame"] = frame_idx
                            active_intervals[event_key_1]["end_timestamp"] = current_timestamp
                            active_intervals[event_key_1]["duration_seconds"] = round((frame_idx - active_intervals[event_key_1]["start_frame"]) / fps, 2)

                    # --- CASE 2: CANDIDATE UNRESPONSIVE / SLUMP / MEDICAL EMERGENCY ---
                    torso_angle, _ = self._calculate_torso_angle(kpts, bbox)
                    
                    # Check stillness over recent frames
                    is_motionless = True
                    if len(track_history[t_id]) >= 15:
                        centers = [h["center"] for h in list(track_history[t_id])[-15:]]
                        disp = max(math.hypot(centers[-1][0] - c[0], centers[-1][1] - c[1]) for c in centers)
                        is_motionless = disp < 6.0

                    if torso_angle <= self.slump_torso_angle_threshold and is_motionless:
                        slump_counters[t_id] += self.frame_stride
                    else:
                        slump_counters[t_id] = max(0, slump_counters[t_id] - self.frame_stride)

                    event_key_2 = (cand_id, "CANDIDATE_UNRESPONSIVE_SLUMP")
                    if slump_counters[t_id] >= slump_frames_limit:
                        current_active_incident_keys.add(event_key_2)
                        if event_key_2 not in active_intervals:
                            active_intervals[event_key_2] = {
                                "event_type": "CANDIDATE_UNRESPONSIVE_SLUMP",
                                "severity": "CRITICAL",
                                "candidate_id": cand_id,
                                "assigned_desk": assigned_desk,
                                "camera_source": camera_id,
                                "start_frame": frame_idx - slump_counters[t_id],
                                "start_timestamp": self._frame_to_timestamp(frame_idx - slump_counters[t_id], fps),
                                "end_frame": frame_idx,
                                "end_timestamp": current_timestamp,
                                "torso_angle_degrees": round(torso_angle, 1),
                                "duration_seconds": round(slump_counters[t_id] / fps, 2),
                                "reason": f"{cand_id} collapsed or slumped flat onto desk (torso angle: {torso_angle:.1f}°) and remains stationary for > {self.slump_seconds_threshold}s (Potential Medical Emergency or Incapacitation)."
                            }
                        else:
                            active_intervals[event_key_2]["end_frame"] = frame_idx
                            active_intervals[event_key_2]["end_timestamp"] = current_timestamp
                            active_intervals[event_key_2]["duration_seconds"] = round((frame_idx - active_intervals[event_key_2]["start_frame"]) / fps, 2)

                    # --- CASE 3: REPEATED INTER-DESK PEEKING (CHEATING) ---
                    # Check if candidate leans or turns head towards any other desk
                    peeking_target_desk = None
                    if kpts is not None and len(kpts) >= 17:
                        nose = kpts[0][:2]
                        if nose[0] > 0 and nose[1] > 0:
                            for other_desk, poly in self.desk_zones.items():
                                if other_desk != assigned_desk and "Invigilator" not in other_desk:
                                    if self._check_point_in_polygon((nose[0], nose[1]), poly):
                                        peeking_target_desk = other_desk
                                        break

                    prev_count, prev_target = current_peeking_counters[t_id]
                    if peeking_target_desk is not None:
                        new_count = prev_count + self.frame_stride
                        current_peeking_counters[t_id] = (new_count, peeking_target_desk)

                        # Once an episode crosses threshold, log it as a discrete peeking episode
                        if new_count == peeking_episode_frames_limit:
                            peeking_episodes_history[t_id].append({
                                "start_time_sec": current_time_sec - self.peeking_episode_seconds_threshold,
                                "end_time_sec": current_time_sec,
                                "target_desk": peeking_target_desk,
                                "frame": frame_idx
                            })
                    else:
                        current_peeking_counters[t_id] = (0, "")

                    # Prune peeking episodes older than peeking_window_seconds
                    peeking_episodes_history[t_id] = [
                        ep for ep in peeking_episodes_history[t_id]
                        if (current_time_sec - ep["end_time_sec"]) <= self.peeking_window_seconds
                    ]

                    # Trigger incident ONLY if repeated episodes >= peeking_repeat_count_threshold
                    recent_episodes = peeking_episodes_history[t_id]
                    event_key_3 = (cand_id, "REPEATED_INTER_DESK_PEEKING")
                    if len(recent_episodes) >= self.peeking_repeat_count_threshold:
                        current_active_incident_keys.add(event_key_3)
                        target_desks_str = ", ".join(list(set(ep["target_desk"] for ep in recent_episodes)))
                        if event_key_3 not in active_intervals:
                            active_intervals[event_key_3] = {
                                "event_type": "REPEATED_INTER_DESK_PEEKING",
                                "severity": "HIGH",
                                "candidate_id": cand_id,
                                "assigned_desk": assigned_desk,
                                "target_desks": target_desks_str,
                                "repeated_episodes_count": len(recent_episodes),
                                "camera_source": camera_id,
                                "start_frame": recent_episodes[0]["frame"],
                                "start_timestamp": self._frame_to_timestamp(recent_episodes[0]["frame"], fps),
                                "end_frame": frame_idx,
                                "end_timestamp": current_timestamp,
                                "reason": f"{cand_id} exhibited repeated suspicious peeking/leaning ({len(recent_episodes)} distinct episodes in {self.peeking_window_seconds:.0f}s window) towards adjacent {target_desks_str} (Single glances were filtered)."
                            }
                        else:
                            active_intervals[event_key_3]["end_frame"] = frame_idx
                            active_intervals[event_key_3]["end_timestamp"] = current_timestamp
                            active_intervals[event_key_3]["repeated_episodes_count"] = len(recent_episodes)

            # Close ended intervals
            closed_keys = [k for k in active_intervals.keys() if k not in current_active_incident_keys]
            for k in closed_keys:
                evt = active_intervals.pop(k)
                completed_events.append(evt)

            # Optional annotated video rendering
            if video_writer is not None:
                # Draw Desk Polygons
                for desk_name, poly in self.desk_zones.items():
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    color = (255, 120, 0) if "Invigilator" in desk_name else (0, 180, 255)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                    cv2.putText(frame, desk_name, (poly[0][0] + 5, poly[0][1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw Time
                cv2.putText(frame, f"Exam Feed: {camera_id} | Time: {current_timestamp}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw Candidates & Alert Badges
                for det in frame_detections:
                    cand_id = det["entity_id"]
                    bbox = [int(v) for v in det["bbox"]]
                    has_alert = any(k[0] == cand_id for k in active_intervals.keys())
                    color = (0, 0, 255) if has_alert else (0, 255, 0)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                    alerts_str = ""
                    for k, v in active_intervals.items():
                        if k[0] == cand_id:
                            alerts_str += f" ! {v['event_type']}"

                    cv2.putText(frame, f"{cand_id}{alerts_str}", (bbox[0], max(bbox[1] - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                video_writer.write(frame)

            frame_idx += 1

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Annotated Exam video saved to: {output_video_path}")

        # Flush remaining intervals
        for k, evt in active_intervals.items():
            completed_events.append(evt)

        # Build report
        report = {
            "exam_surveillance_metadata": {
                "file_path": video_path,
                "camera_source": camera_id,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(total_frames / max(fps, 1.0), 2),
                "simulated_start_time": self.simulated_start_time,
                "desk_zones_configured": list(self.desk_zones.keys())
            },
            "summary": {
                "total_candidates_tracked": len(track_to_desk),
                "total_anomalies_detected": len(completed_events),
                "anomalies_by_type": {
                    "UNAUTHORIZED_DESK_ABANDONMENT": sum(1 for e in completed_events if e["event_type"] == "UNAUTHORIZED_DESK_ABANDONMENT"),
                    "CANDIDATE_UNRESPONSIVE_SLUMP": sum(1 for e in completed_events if e["event_type"] == "CANDIDATE_UNRESPONSIVE_SLUMP"),
                    "REPEATED_INTER_DESK_PEEKING": sum(1 for e in completed_events if e["event_type"] == "REPEATED_INTER_DESK_PEEKING")
                }
            },
            "incidents": completed_events
        }

        if output_json_path:
            with open(output_json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[INFO] Exam Incident JSON report saved to: {output_json_path}")

        # Explicit garbage collection for RAM hygiene (Rule 5 & 8)
        import gc
        del track_to_desk
        del track_history
        del active_intervals
        gc.collect()

        return report


def main():
    parser = argparse.ArgumentParser(description="Exam Hall Autonomous Vision & Proctoring Analyzer")
    parser.add_argument("--video", type=str, required=True, help="Path to input exam video (or filename inside test/videos/)")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save output JSON incident log")
    parser.add_argument("--output_video", type=str, default=None, help="Path to save annotated MP4 video")
    parser.add_argument("--start_time", type=str, default="10:00:00", help="Exam start timestamp (HH:MM:SS)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO model name")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default 1 = every frame)")
    parser.add_argument("--camera_id", type=str, default="Cam_01_ExamHall", help="Camera source identifier")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    analyzer = ExamHallAnomalyAnalyzer(
        model_name=args.model,
        simulated_start_time=args.start_time,
        frame_stride=args.stride
    )

    # Resolve video path
    video_input = args.video
    if not os.path.exists(video_input):
        candidate = os.path.join(VIDEOS_DIR, video_input)
        if os.path.exists(candidate):
            video_input = candidate

    video_basename = os.path.splitext(os.path.basename(video_input))[0]
    out_json = args.output_json or os.path.join(OUTPUTS_DIR, f"{video_basename}_exam_incidents.json")
    out_video = args.output_video or os.path.join(OUTPUTS_DIR, f"{video_basename}_exam_annotated.mp4")

    report = analyzer.process_video(
        video_path=video_input,
        output_json_path=out_json,
        save_annotated_video=True,
        output_video_path=out_video,
        camera_id=args.camera_id,
        max_frames=args.max_frames
    )

    print("\n" + "="*55)
    print("EXAM SURVEILLANCE INCIDENT SUMMARY:")
    print("="*55)
    print(json.dumps(report["summary"], indent=2))
    print(f"\nDetailed incidents written to: {out_json}")
    print(f"Annotated exam video saved to: {out_video}")


if __name__ == "__main__":
    main()
