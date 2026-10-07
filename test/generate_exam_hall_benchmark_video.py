"""
Synthetic Exam Hall Dual-Angle CCTV Benchmark Video Generator
Creates realistic exam hall CCTV recordings for benchmarking the 3 core cases:
  1. Candidate 2: Repeated Inter-Desk Peeking towards Desk 1 (2 distinct episodes in 20s).
  2. Candidate 3: Slumps flat on desk and stays unresponsive (> 20s stillness).
  3. Candidate 4: Abandons assigned desk and walks into aisle.
  4. Candidate 1: Normal focused student.

Generates:
  - test/videos/exam_hall_cam1_front.mp4 (Front/Overhead view)
  - test/videos/exam_hall_cam2_rear.mp4 (Rear/Angle view)
"""

import os
import cv2
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

def generate_exam_hall_video(
    output_path: str,
    view_type: str = "front",
    duration_sec: int = 25,
    fps: int = 12
):
    width, height = 800, 700
    total_frames = duration_sec * fps
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try codecs supported on macOS
    writer = None
    for codec in ["avc1", "mp4v", "MJPG", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            break

    if not writer or not writer.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {output_path}")
        return

    # Desk Polygons: [x1, y1, x2, y2]
    desks = {
        "Desk_01": (80, 100, 280, 320),
        "Desk_02": (320, 100, 520, 320),
        "Desk_03": (80, 380, 280, 600),
        "Desk_04": (320, 380, 520, 600),
        "Invigilator_Podium": (600, 80, 740, 250),
    }

    for f in range(total_frames):
        t_sec = f / float(fps)
        # Background floor
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Draw Tiles
        for x in range(0, width, 60):
            cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 1)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (60, 60, 60), 1)

        # Draw Desk Zones
        for name, (x1, y1, x2, y2) in desks.items():
            color = (80, 40, 20) if "Invigilator" in name else (35, 50, 65)
            border_color = (255, 140, 0) if "Invigilator" in name else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(frame, name, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 1)

            # Desk table outline
            cv2.rectangle(frame, (x1 + 20, y1 + 50), (x2 - 20, y1 + 110), (90, 80, 70), -1)

        # --- CANDIDATE 1 (Desk_01: Normal Student) ---
        c1_x, c1_y = 180, 240
        # Normal seated posture (head upright, body facing table)
        cv2.circle(frame, (c1_x, c1_y - 70), 16, (220, 220, 220), -1) # Head
        cv2.line(frame, (c1_x, c1_y - 54), (c1_x, c1_y - 10), (220, 220, 220), 8) # Upright Torso
        cv2.line(frame, (c1_x, c1_y - 10), (c1_x - 12, c1_y + 25), (200, 200, 200), 6) # Legs
        cv2.line(frame, (c1_x, c1_y - 10), (c1_x + 12, c1_y + 25), (200, 200, 200), 6)

        # --- CANDIDATE 2 (Desk_02: Repeated Peeking towards Desk 01) ---
        c2_base_x, c2_base_y = 420, 240
        # Episode 1: Peeking at t = 3s to 6s
        # Episode 2: Peeking at t = 10s to 13s
        is_peeking_ep1 = (3.0 <= t_sec <= 6.0)
        is_peeking_ep2 = (10.0 <= t_sec <= 13.0)
        is_peeking = is_peeking_ep1 or is_peeking_ep2

        if is_peeking:
            # Leaning left towards Desk 01
            c2_head_x = c2_base_x - 70
            c2_head_y = c2_base_y - 65
            cv2.circle(frame, (c2_head_x, c2_head_y), 16, (180, 180, 255), -1)
            cv2.line(frame, (c2_head_x + 10, c2_head_y + 10), (c2_base_x, c2_base_y - 10), (180, 180, 255), 8) # Angled Torso
            cv2.line(frame, (c2_base_x, c2_base_y - 10), (c2_base_x - 12, c2_base_y + 25), (200, 200, 200), 6)
            cv2.line(frame, (c2_base_x, c2_base_y - 10), (c2_base_x + 12, c2_base_y + 25), (200, 200, 200), 6)
        else:
            # Normal upright
            cv2.circle(frame, (c2_base_x, c2_base_y - 70), 16, (220, 220, 220), -1)
            cv2.line(frame, (c2_base_x, c2_base_y - 54), (c2_base_x, c2_base_y - 10), (220, 220, 220), 8)
            cv2.line(frame, (c2_base_x, c2_base_y - 10), (c2_base_x - 12, c2_base_y + 25), (200, 200, 200), 6)
            cv2.line(frame, (c2_base_x, c2_base_y - 10), (c2_base_x + 12, c2_base_y + 25), (200, 200, 200), 6)

        # --- CANDIDATE 3 (Desk_03: Slump / Unresponsive on Desk) ---
        c3_x, c3_y = 180, 520
        if t_sec < 4.0:
            # Upright
            cv2.circle(frame, (c3_x, c3_y - 70), 16, (220, 220, 220), -1)
            cv2.line(frame, (c3_x, c3_y - 54), (c3_x, c3_y - 10), (220, 220, 220), 8)
            cv2.line(frame, (c3_x, c3_y - 10), (c3_x - 12, c3_y + 25), (200, 200, 200), 6)
            cv2.line(frame, (c3_x, c3_y - 10), (c3_x + 12, c3_y + 25), (200, 200, 200), 6)
        else:
            # Slumped flat on desk surface (Torso horizontal, angle ~ 15 deg)
            head_slump_x = c3_x + 20
            head_slump_y = c3_y - 85 # Head down on table
            cv2.circle(frame, (head_slump_x, head_slump_y), 16, (255, 180, 180), -1)
            cv2.line(frame, (head_slump_x - 10, head_slump_y + 5), (c3_x, c3_y - 10), (255, 180, 180), 8)
            cv2.line(frame, (c3_x, c3_y - 10), (c3_x - 12, c3_y + 25), (200, 200, 200), 6)
            cv2.line(frame, (c3_x, c3_y - 10), (c3_x + 12, c3_y + 25), (200, 200, 200), 6)

        # --- CANDIDATE 4 (Desk_04: Desk Abandonment / Wandering) ---
        c4_base_x, c4_base_y = 420, 520
        if t_sec < 5.0:
            # Inside Desk 04
            c4_cur_x, c4_cur_y = c4_base_x, c4_base_y
        else:
            # Walks out of Desk 04 into center aisle towards Invigilator Podium
            walk_progress = min((t_sec - 5.0) / 12.0, 1.0)
            c4_cur_x = int(c4_base_x + walk_progress * 220)
            c4_cur_y = int(c4_base_y - walk_progress * 300)

        cv2.circle(frame, (c4_cur_x, c4_cur_y - 70), 16, (200, 255, 200), -1)
        cv2.line(frame, (c4_cur_x, c4_cur_y - 54), (c4_cur_x, c4_cur_y - 10), (200, 255, 200), 8)
        cv2.line(frame, (c4_cur_x, c4_cur_y - 10), (c4_cur_x - 10, c4_cur_y + 25), (200, 200, 200), 6)
        cv2.line(frame, (c4_cur_x, c4_cur_y - 10), (c4_cur_x + 10, c4_cur_y + 25), (200, 200, 200), 6)

        # Camera Header
        cam_title = f"CCTV CAM 01 [FRONT] - TIME: 10:{int(t_sec//60):02d}:{int(t_sec%60):02d}" if view_type == "front" else f"CCTV CAM 02 [REAR] - TIME: 10:{int(t_sec//60):02d}:{int(t_sec%60):02d}"
        cv2.putText(frame, cam_title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"[SUCCESS] Exam CCTV Benchmark Video Created: {output_path} ({total_frames} frames)")

def main():
    front_video = os.path.join(VIDEOS_DIR, "exam_hall_cam1_front.mp4")
    rear_video = os.path.join(VIDEOS_DIR, "exam_hall_cam2_rear.mp4")

    print("Generating Dual-Angle Exam Hall CCTV Benchmark Videos...")
    generate_exam_hall_video(front_video, view_type="front", duration_sec=25, fps=12)
    generate_exam_hall_video(rear_video, view_type="rear", duration_sec=25, fps=12)
    print("\nBenchmark videos ready in:", VIDEOS_DIR)

if __name__ == "__main__":
    main()
