import os
import sys
import json
import math
import argparse
from typing import Dict, List, Any, Tuple

# Enforce strict local directory containment & organization (Rules 7 & 8)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(WORKSPACE_DIR, "test")
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


class BankAnomalyAnalyzer:
    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        zones_config: Dict[str, List[List[int]]] = None,
        operating_hours: Tuple[str, str] = ("09:00:00", "17:00:00"),
        simulated_start_time: str = "10:00:00",
        fps_override: float = None,
        fall_consecutive_frames_threshold: int = 30, # ~1 sec at 30 fps
        fall_torso_angle_threshold: float = 30.0,   # degrees relative to ground horizontal
        fall_motion_threshold_pixels: float = 8.0,
        frame_stride: int = 1,                      # Default 1: process every single frame
    ):
        self.model_name = model_name
        self.operating_hours = operating_hours
        self.simulated_start_time = simulated_start_time
        self.fps_override = fps_override
        self.fall_consecutive_frames_threshold = fall_consecutive_frames_threshold
        self.fall_torso_angle_threshold = fall_torso_angle_threshold
        self.fall_motion_threshold_pixels = fall_motion_threshold_pixels
        self.frame_stride = max(1, frame_stride)

        # Optimize PyTorch CPU threads for 2-core Intel i5
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass

        # Default Bank Polygonal Restricted Zones (normalized or absolute pixel coords)
        # E.g. Vault area, Behind Teller Counter
        self.zones_config = zones_config or {
            "Vault_Restricted_Zone": [
                [50, 50],
                [350, 50],
                [350, 300],
                [50, 300]
            ],
            "Behind_Teller_Counter": [
                [450, 50],
                [900, 50],
                [900, 220],
                [450, 220]
            ]
        }

        # Initialize YOLO model lazily
        self.model = None

    def _load_model(self):
        if self.model is None:
            if YOLO is None:
                raise RuntimeError("Ultralytics YOLO is not installed. Please run: pip install ultralytics")
            
            # Check local models directory first
            target_model_path = self.model_name
            if not os.path.exists(target_model_path):
                local_candidate = os.path.join(MODELS_DIR, self.model_name)
                if os.path.exists(local_candidate):
                    target_model_path = local_candidate

            print(f"[INFO] Loading YOLO-Pose model: {target_model_path}...")
            try:
                self.model = YOLO(target_model_path)
            except Exception as e:
                fallback = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
                print(f"[WARN] Failed to load {target_model_path} ({e}), falling back to {fallback}")
                self.model = YOLO(fallback)

    def _frame_to_timestamp(self, frame_idx: int, fps: float) -> str:
        """Converts frame index + simulated start time into HH:MM:SS.mmm string."""
        total_seconds = frame_idx / max(fps, 1.0)
        h, m, s = map(int, self.simulated_start_time.split(":"))
        start_total = h * 3600 + m * 60 + s
        curr_total = start_total + total_seconds
        
        cur_h = int((curr_total // 3600) % 24)
        cur_m = int((curr_total % 3600) // 60)
        cur_s = curr_total % 60
        return f"{cur_h:02d}:{cur_m:02d}:{cur_s:06.3f}"

    def _is_after_hours(self, timestamp_str: str) -> bool:
        """Checks if HH:MM:SS timestamp is outside operating hours (e.g. 09:00 - 17:00)."""
        time_part = timestamp_str.split(".")[0]
        curr_time = list(map(int, time_part.split(":")))
        curr_sec = curr_time[0] * 3600 + curr_time[1] * 60 + curr_time[2]

        open_time = list(map(int, self.operating_hours[0].split(":")))
        open_sec = open_time[0] * 3600 + open_time[1] * 60 + open_time[2]

        close_time = list(map(int, self.operating_hours[1].split(":")))
        close_sec = close_time[0] * 3600 + close_time[1] * 60 + close_time[2]

        if open_sec <= close_sec:
            return not (open_sec <= curr_sec <= close_sec)
        else:
            # Over midnight wrap
            return not (curr_sec >= open_sec or curr_sec <= close_sec)

    def _check_point_in_polygon(self, point: Tuple[float, float], polygon: List[List[int]]) -> bool:
        pts = np.array(polygon, dtype=np.int32)
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return dist >= 0

    def _calculate_pose_orientation_and_fall(
        self,
        keypoints: np.ndarray,
        bbox: List[float],
        track_history: List[Dict[str, Any]]
    ) -> Tuple[bool, float, str]:
        """
        Calculates torso angle relative to horizontal ground and determines fall state.
        Keypoint indices in COCO 17-point format:
          0: nose, 5: left_shoulder, 6: right_shoulder,
          11: left_hip, 12: right_hip, 15: left_ankle, 16: right_ankle
        """
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        aspect_ratio = height / width

        torso_angle = 90.0 # default vertical
        has_hip_shoulder = False

        if keypoints is not None and len(keypoints) >= 17:
            # Shoulders midpoint
            ls, rs = keypoints[5][:2], keypoints[6][:2]
            lh, rh = keypoints[11][:2], keypoints[12][:2]

            if (ls[0] > 0 or rs[0] > 0) and (lh[0] > 0 or rh[0] > 0):
                shoulder_mid = (ls + rs) / 2.0 if (ls[0] > 0 and rs[0] > 0) else (ls if ls[0] > 0 else rs)
                hip_mid = (lh + rh) / 2.0 if (lh[0] > 0 and rh[0] > 0) else (lh if lh[0] > 0 else rh)

                dx = hip_mid[0] - shoulder_mid[0]
                dy = hip_mid[1] - shoulder_mid[1] # downward in image coords

                # Angle relative to horizontal (0 deg = flat on ground, 90 deg = standing upright)
                angle_rad = math.atan2(abs(dy), abs(dx))
                torso_angle = math.degrees(angle_rad)
                has_hip_shoulder = True

        # Fall posture criteria:
        # 1. Torso angle is low (< fall_torso_angle_threshold) OR aspect ratio < 0.85
        is_horizontal_posture = (torso_angle <= self.fall_torso_angle_threshold) or (aspect_ratio < 0.8 and not has_hip_shoulder)
        
        # Check motion stillness over recent history
        is_still = False
        if len(track_history) >= 10:
            recent_centers = [h["center"] for h in track_history[-10:]]
            max_displacement = max(
                math.hypot(recent_centers[-1][0] - c[0], recent_centers[-1][1] - c[1])
                for c in recent_centers
            )
            is_still = max_displacement < self.fall_motion_threshold_pixels
        else:
            is_still = True

        is_fall = is_horizontal_posture and is_still
        reason = f"Torso angle: {torso_angle:.1f}° relative to ground (threshold <= {self.fall_torso_angle_threshold}°), height/width aspect ratio: {aspect_ratio:.2f}, motion stillness confirmed."
        return is_fall, torso_angle, reason

    def process_video(
        self,
        video_path: str,
        output_json_path: str = None,
        save_annotated_video: bool = True,
        output_video_path: str = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end video analysis and produces explainable incident records.
        """
        self._load_model()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = None
        if save_annotated_video:
            if not output_video_path:
                base, ext = os.path.splitext(video_path)
                output_video_path = f"{base}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"[INFO] Processing video: {video_path} ({total_frames} frames @ {fps:.1f} FPS, {frame_width}x{frame_height})")

        # Track history: track_id -> list of frame states
        tracks_state: Dict[int, List[Dict[str, Any]]] = {}
        # Active active anomaly intervals: (track_id, anomaly_type) -> current event dict
        active_intervals: Dict[Tuple[int, str], Dict[str, Any]] = {}
        completed_events: List[Dict[str, Any]] = []

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frame stride skipping for Intel Dual-Core CPU efficiency
            if frame_idx % self.frame_stride != 0:
                frame_idx += 1
                continue

            current_timestamp = self._frame_to_timestamp(frame_idx, fps)
            after_hours_flag = self._is_after_hours(current_timestamp)

            # YOLO inference with ByteTrack
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            current_frame_detections = []
            current_active_keys_in_frame = set()

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                keypoints_data = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

                for i, track_id in enumerate(track_ids):
                    bbox = boxes[i].tolist()
                    conf = float(confs[i])
                    kpts = keypoints_data[i] if keypoints_data is not None else None

                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    foot_point = ((x1 + x2) / 2.0, y2) # bottom center (ground contact)

                    if track_id not in tracks_state:
                        tracks_state[track_id] = []

                    state = {
                        "frame_idx": frame_idx,
                        "timestamp": current_timestamp,
                        "bbox": bbox,
                        "center": center,
                        "foot_point": foot_point,
                        "conf": conf,
                    }
                    tracks_state[track_id].append(state)

                    # --- EVALUATE CASE 1: FORBIDDEN / RESTRICTED ZONE BREACH ---
                    for zone_name, polygon in self.zones_config.items():
                        is_in_zone = self._check_point_in_polygon(foot_point, polygon)
                        event_key = (int(track_id), f"RESTRICTED_ZONE_BREACH:{zone_name}")

                        if is_in_zone:
                            current_active_keys_in_frame.add(event_key)
                            if event_key not in active_intervals:
                                active_intervals[event_key] = {
                                    "event_type": "UNAUTHORIZED_ZONE_BREACH",
                                    "severity": "CRITICAL",
                                    "track_id": int(track_id),
                                    "zone": zone_name,
                                    "start_frame": frame_idx,
                                    "start_timestamp": current_timestamp,
                                    "end_frame": frame_idx,
                                    "end_timestamp": current_timestamp,
                                    "foot_coordinates": [round(foot_point[0], 1), round(foot_point[1], 1)],
                                    "reason": f"Person (Track #{track_id}) entered forbidden security zone '{zone_name}' at ground coordinates ({foot_point[0]:.1f}, {foot_point[1]:.1f}) without authorization."
                                }
                            else:
                                active_intervals[event_key]["end_frame"] = frame_idx
                                active_intervals[event_key]["end_timestamp"] = current_timestamp

                    # --- EVALUATE CASE 2: AFTER-HOURS INTRUSION ---
                    if after_hours_flag:
                        event_key = (int(track_id), "AFTER_HOURS_INTRUSION")
                        current_active_keys_in_frame.add(event_key)
                        if event_key not in active_intervals:
                            active_intervals[event_key] = {
                                "event_type": "AFTER_HOURS_INTRUSION",
                                "severity": "HIGH",
                                "track_id": int(track_id),
                                "zone": "Bank_Interior",
                                "start_frame": frame_idx,
                                "start_timestamp": current_timestamp,
                                "end_frame": frame_idx,
                                "end_timestamp": current_timestamp,
                                "operating_hours": f"{self.operating_hours[0]} to {self.operating_hours[1]}",
                                "reason": f"Person (Track #{track_id}) detected inside bank facility outside operating hours at timestamp {current_timestamp} (Authorized hours: {self.operating_hours[0]} - {self.operating_hours[1]})."
                            }
                        else:
                            active_intervals[event_key]["end_frame"] = frame_idx
                            active_intervals[event_key]["end_timestamp"] = current_timestamp

                    # --- EVALUATE CASE 3: FALL / DOWN & UNMOVING ---
                    is_fall, torso_angle, fall_reason = self._calculate_pose_orientation_and_fall(
                        kpts, bbox, tracks_state[track_id]
                    )
                    
                    event_key = (int(track_id), "FALL_AND_UNRESPONSIVE")
                    if is_fall:
                        current_active_keys_in_frame.add(event_key)
                        if event_key not in active_intervals:
                            active_intervals[event_key] = {
                                "event_type": "FALL_AND_UNRESPONSIVE",
                                "severity": "CRITICAL",
                                "track_id": int(track_id),
                                "zone": "Floor_Area",
                                "start_frame": frame_idx,
                                "start_timestamp": current_timestamp,
                                "end_frame": frame_idx,
                                "end_timestamp": current_timestamp,
                                "torso_angle_degrees": round(torso_angle, 1),
                                "reason": f"Person (Track #{track_id}) collapsed to horizontal posture ({fall_reason}) and remains unmoving."
                            }
                        else:
                            active_intervals[event_key]["end_frame"] = frame_idx
                            active_intervals[event_key]["end_timestamp"] = current_timestamp

            # Close any active intervals that ended in this frame
            closed_keys = [k for k in active_intervals.keys() if k not in current_active_keys_in_frame]
            for k in closed_keys:
                evt = active_intervals.pop(k)
                # Filter out single-frame momentary blips if needed (> 3 frames duration)
                if evt["end_frame"] - evt["start_frame"] >= 2:
                    completed_events.append(evt)

            # Draw visual debug annotations if video writer enabled
            if video_writer is not None:
                # Draw Zones
                for zone_name, polygon in self.zones_config.items():
                    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.putText(frame, f"[ZONE] {zone_name}", (polygon[0][0] + 5, polygon[0][1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw Time & Operating Hours
                time_color = (0, 0, 255) if after_hours_flag else (0, 255, 0)
                cv2.putText(frame, f"Time: {current_timestamp} | Hours: {self.operating_hours[0]}-{self.operating_hours[1]}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)

                # Draw Track Bounding Boxes
                if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                    for i, track_id in enumerate(results[0].boxes.id.int().cpu().numpy()):
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].int().cpu().numpy()
                        # Check if this track currently has an active alert
                        has_alert = any(k[0] == int(track_id) for k in active_intervals.keys())
                        box_color = (0, 0, 255) if has_alert else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        alert_label = ""
                        for k, v in active_intervals.items():
                            if k[0] == int(track_id):
                                alert_label += f" ! {v['event_type']}"

                        cv2.putText(frame, f"Track #{track_id}{alert_label}", (x1, max(y1 - 8, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                video_writer.write(frame)

            frame_idx += 1

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Annotated video saved to: {output_video_path}")

        # Flush any remaining active intervals at video end
        for k, evt in active_intervals.items():
            completed_events.append(evt)

        # Build final structured JSON output
        report = {
            "video_metadata": {
                "file_path": video_path,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(total_frames / max(fps, 1.0), 2),
                "operating_hours_configured": f"{self.operating_hours[0]} to {self.operating_hours[1]}",
                "simulated_start_time": self.simulated_start_time
            },
            "summary": {
                "total_unique_tracks": len(tracks_state),
                "total_anomalies_detected": len(completed_events),
                "anomalies_by_type": {
                    "UNAUTHORIZED_ZONE_BREACH": sum(1 for e in completed_events if e["event_type"] == "UNAUTHORIZED_ZONE_BREACH"),
                    "AFTER_HOURS_INTRUSION": sum(1 for e in completed_events if e["event_type"] == "AFTER_HOURS_INTRUSION"),
                    "FALL_AND_UNRESPONSIVE": sum(1 for e in completed_events if e["event_type"] == "FALL_AND_UNRESPONSIVE")
                }
            },
            "incidents": completed_events
        }

        if output_json_path:
            with open(output_json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[INFO] Anomaly JSON report saved to: {output_json_path}")

        # Explicit garbage collection to ensure fresh memory state for next video run
        import gc
        del tracks_state
        del active_intervals
        gc.collect()

        return report


def main():
    parser = argparse.ArgumentParser(description="Bank Vision Security Anomaly Analyzer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save JSON incident report")
    parser.add_argument("--output_video", type=str, default=None, help="Path to save annotated MP4 video")
    parser.add_argument("--start_time", type=str, default="10:00:00", help="Simulated video start time (HH:MM:SS)")
    parser.add_argument("--open_hours", type=str, default="09:00:00,17:00:00", help="Operating hours start,end")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO model path or name")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default 1 = every single frame)")
    args = parser.parse_args()

    open_h, close_h = args.open_hours.split(",")
    analyzer = BankAnomalyAnalyzer(
        model_name=args.model,
        operating_hours=(open_h.strip(), close_h.strip()),
        simulated_start_time=args.start_time,
        frame_stride=args.stride
    )

    # Resolve input video path
    video_input = args.video
    if not os.path.exists(video_input):
        candidate = os.path.join(VIDEOS_DIR, video_input)
        if os.path.exists(candidate):
            video_input = candidate

    video_basename = os.path.splitext(os.path.basename(video_input))[0]
    out_json = args.output_json or os.path.join(OUTPUTS_DIR, f"{video_basename}_incidents.json")
    out_video = args.output_video or os.path.join(OUTPUTS_DIR, f"{video_basename}_annotated.mp4")

    report = analyzer.process_video(
        video_path=video_input,
        output_json_path=out_json,
        save_annotated_video=True,
        output_video_path=out_video
    )

    print("\n" + "="*50)
    print("ANOMALY DETECTION REPORT SUMMARY:")
    print("="*50)
    print(json.dumps(report["summary"], indent=2))
    print(f"\nDetailed incidents written to: {out_json}")
    print(f"Annotated video saved to: {out_video}")


if __name__ == "__main__":
    main()
