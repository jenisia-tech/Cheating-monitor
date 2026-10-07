"""
Multi-Camera Exam Hall Incident Fusion & Synchronizer
HNX26PSI07: Autonomous Vision & Behaviour Understanding

Ingests two or more camera feeds (e.g. Cam 1: Front View, Cam 2: Rear View),
runs the ExamHallAnomalyAnalyzer on each feed, and merges the detections into
a single unified, multi-perspective exam incident report.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(TEST_DIR)
VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
OUTPUTS_DIR = os.path.join(TEST_DIR, "outputs")
MODELS_DIR = os.path.join(TEST_DIR, "models")

sys.path.insert(0, TEST_DIR)
from exam_analyzer import ExamHallAnomalyAnalyzer


class MultiCameraExamSynchronizer:
    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        simulated_start_time: str = "10:00:00",
        frame_stride: int = 1
    ):
        self.model_name = model_name
        self.simulated_start_time = simulated_start_time
        self.frame_stride = frame_stride

    def process_dual_camera_session(
        self,
        cam1_video_path: str,
        cam2_video_path: str,
        cam1_desk_zones: Dict[str, List[List[int]]] = None,
        cam2_desk_zones: Dict[str, List[List[int]]] = None,
        output_json_path: str = None
    ) -> Dict[str, Any]:
        """
        Runs analysis across both camera angles and fuses incidents by physical Desk ID.
        """
        # Resolve full video paths
        if not os.path.exists(cam1_video_path):
            candidate = os.path.join(VIDEOS_DIR, cam1_video_path)
            if os.path.exists(candidate):
                cam1_video_path = candidate

        if not os.path.exists(cam2_video_path):
            candidate = os.path.join(VIDEOS_DIR, cam2_video_path)
            if os.path.exists(candidate):
                cam2_video_path = candidate

        print("\n" + "="*60)
        print("STARTING DUAL-CAMERA EXAM SESSION ANALYSIS")
        print(f"Camera 1 (Front View): {cam1_video_path}")
        print(f"Camera 2 (Rear View) : {cam2_video_path}")
        print("="*60)

        # 1. Analyze Camera 1
        analyzer_cam1 = ExamHallAnomalyAnalyzer(
            model_name=self.model_name,
            desk_zones_config=cam1_desk_zones,
            simulated_start_time=self.simulated_start_time,
            frame_stride=self.frame_stride
        )
        report_cam1 = analyzer_cam1.process_video(
            video_path=cam1_video_path,
            save_annotated_video=True,
            camera_id="Cam_01_Front"
        )

        # 2. Analyze Camera 2
        analyzer_cam2 = ExamHallAnomalyAnalyzer(
            model_name=self.model_name,
            desk_zones_config=cam2_desk_zones,
            simulated_start_time=self.simulated_start_time,
            frame_stride=self.frame_stride
        )
        report_cam2 = analyzer_cam2.process_video(
            video_path=cam2_video_path,
            save_annotated_video=True,
            camera_id="Cam_02_Rear"
        )

        # 3. Fuse Incidents across Cameras by Candidate Desk ID & Timestamp
        fused_incidents: List[Dict[str, Any]] = []
        raw_all = report_cam1["incidents"] + report_cam2["incidents"]

        # Deduplicate incidents for the same Candidate and Anomaly Type that overlap in time
        raw_all.sort(key=lambda x: (x["candidate_id"], x["event_type"], x["start_frame"]))

        for inc in raw_all:
            matched = False
            for existing in fused_incidents:
                # Same candidate & same anomaly
                if (existing["candidate_id"] == inc["candidate_id"] and
                    existing["event_type"] == inc["event_type"]):
                    # If time intervals overlap or are within 3 seconds
                    if (inc["start_frame"] <= existing["end_frame"] + 36 and
                        inc["end_frame"] >= existing["start_frame"] - 36):
                        # Merge multi-camera confirmation
                        if "verified_by_cameras" not in existing:
                            existing["verified_by_cameras"] = [existing["camera_source"]]
                        if inc["camera_source"] not in existing["verified_by_cameras"]:
                            existing["verified_by_cameras"].append(inc["camera_source"])
                        existing["end_frame"] = max(existing["end_frame"], inc["end_frame"])
                        existing["end_timestamp"] = max(existing["end_timestamp"], inc["end_timestamp"])
                        existing["confidence_score"] = "HIGH (Multi-Camera Verified)"
                        matched = True
                        break

            if not matched:
                inc_copy = dict(inc)
                inc_copy["verified_by_cameras"] = [inc["camera_source"]]
                inc_copy["confidence_score"] = "CONFIRMED"
                fused_incidents.append(inc_copy)

        fused_report = {
            "multi_camera_session_metadata": {
                "cam1_source": cam1_video_path,
                "cam2_source": cam2_video_path,
                "total_fused_candidates": len(set(i["candidate_id"] for i in fused_incidents)),
                "total_fused_anomalies": len(fused_incidents),
                "simulated_start_time": self.simulated_start_time,
            },
            "summary": {
                "total_incidents": len(fused_incidents),
                "by_type": {
                    "UNAUTHORIZED_DESK_ABANDONMENT": sum(1 for e in fused_incidents if e["event_type"] == "UNAUTHORIZED_DESK_ABANDONMENT"),
                    "CANDIDATE_UNRESPONSIVE_SLUMP": sum(1 for e in fused_incidents if e["event_type"] == "CANDIDATE_UNRESPONSIVE_SLUMP"),
                    "REPEATED_INTER_DESK_PEEKING": sum(1 for e in fused_incidents if e["event_type"] == "REPEATED_INTER_DESK_PEEKING")
                },
                "multi_angle_cross_verified_count": sum(1 for e in fused_incidents if len(e.get("verified_by_cameras", [])) > 1)
            },
            "fused_incidents": fused_incidents
        }

        if not output_json_path:
            output_json_path = os.path.join(OUTPUTS_DIR, "multi_camera_fused_exam_incidents.json")

        with open(output_json_path, "w") as f:
            json.dump(fused_report, f, indent=2)

        print("\n" + "="*60)
        print("DUAL-CAMERA FUSED INCIDENT REPORT SUMMARY:")
        print("="*60)
        print(json.dumps(fused_report["summary"], indent=2))
        print(f"\nFinal fused incident JSON saved to: {output_json_path}")

        return fused_report


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Exam Incident Synchronizer")
    parser.add_argument("--cam1", type=str, default="cctv_people_surveillance.mp4", help="Camera 1 video path")
    parser.add_argument("--cam2", type=str, default="multi_person_traffic.mp4", help="Camera 2 video path")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    sync = MultiCameraExamSynchronizer()
    sync.process_dual_camera_session(
        cam1_video_path=args.cam1,
        cam2_video_path=args.cam2,
        output_json_path=args.output_json
    )


if __name__ == "__main__":
    main()
