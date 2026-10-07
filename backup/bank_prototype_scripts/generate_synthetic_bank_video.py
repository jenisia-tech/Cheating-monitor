"""
Synthetic Bank CCTV Video Generator
Creates a realistic test video for benchmarking the 3 anomaly scenarios:
  1. Person walking into Vault zone (Restricted Area Breach)
  2. Person collapsing and staying stationary on floor (Fall Detection)
  3. Normal customer walking in lobby
"""

import os
import cv2
import numpy as np

def generate_bank_test_video(output_path: str = "test/bank_cctv_test.mp4", duration_sec: int = 10, fps: int = 30):
    width, height = 1280, 720
    total_frames = duration_sec * fps
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Zone coordinates
    vault_polygon = np.array([[50, 50], [350, 50], [350, 300], [50, 300]], np.int32)
    teller_polygon = np.array([[450, 50], [900, 50], [900, 220], [450, 220]], np.int32)

    for f in range(total_frames):
        # Create background bank floor
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw grid floor tiles
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (55, 55, 55), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (55, 55, 55), 1)

        # Draw Vault zone
        cv2.fillPoly(frame, [vault_polygon], (20, 20, 80))
        cv2.polylines(frame, [vault_polygon], True, (0, 0, 220), 2)
        cv2.putText(frame, "RESTRICTED VAULT", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw Teller counter zone
        cv2.fillPoly(frame, [teller_polygon], (20, 60, 20))
        cv2.polylines(frame, [teller_polygon], True, (0, 200, 0), 2)
        cv2.putText(frame, "TELLER DESK", (460, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Entity 1: Normal Customer (Walking left to right across lobby) ---
        p1_x = int(100 + (f / total_frames) * 800)
        p1_y = 520
        # Draw realistic human-like stick figure (head, torso, limbs)
        cv2.circle(frame, (p1_x, p1_y - 80), 15, (220, 220, 220), -1) # Head
        cv2.line(frame, (p1_x, p1_y - 65), (p1_x, p1_y - 20), (220, 220, 220), 8) # Torso
        cv2.line(frame, (p1_x, p1_y - 50), (p1_x - 15, p1_y - 30), (220, 220, 220), 4) # Arms
        cv2.line(frame, (p1_x, p1_y - 50), (p1_x + 15, p1_y - 30), (220, 220, 220), 4)
        cv2.line(frame, (p1_x, p1_y - 20), (p1_x - 10, p1_y), (220, 220, 220), 5) # Legs
        cv2.line(frame, (p1_x, p1_y - 20), (p1_x + 10, p1_y), (220, 220, 220), 5)

        # --- Entity 2: Intruder (Starts outside, walks straight into Vault Zone at frame 90) ---
        p2_x = 200
        # Walk from y=450 into Vault at y=200
        p2_y = int(450 - min(f / 150.0, 1.0) * 260)
        cv2.circle(frame, (p2_x, p2_y - 80), 15, (180, 180, 240), -1)
        cv2.line(frame, (p2_x, p2_y - 65), (p2_x, p2_y - 20), (180, 180, 240), 8)
        cv2.line(frame, (p2_x, p2_y - 20), (p2_x - 10, p2_y), (180, 180, 240), 5)
        cv2.line(frame, (p2_x, p2_y - 20), (p2_x + 10, p2_y), (180, 180, 240), 5)

        # --- Entity 3: Person Falling down (Walks then collapses at frame 100) ---
        p3_x = 750
        p3_y = 520
        if f < 90:
            # Standing
            cv2.circle(frame, (p3_x, p3_y - 80), 15, (200, 240, 200), -1)
            cv2.line(frame, (p3_x, p3_y - 65), (p3_x, p3_y - 20), (200, 240, 200), 8)
            cv2.line(frame, (p3_x, p3_y - 20), (p3_x - 10, p3_y), (200, 240, 200), 5)
            cv2.line(frame, (p3_x, p3_y - 20), (p3_x + 10, p3_y), (200, 240, 200), 5)
        else:
            # Horizontal / Fallen on floor
            cv2.circle(frame, (p3_x - 45, p3_y - 10), 15, (200, 240, 200), -1) # Head on floor
            cv2.line(frame, (p3_x - 30, p3_y - 10), (p3_x + 30, p3_y - 10), (200, 240, 200), 8) # Horizontal torso
            cv2.line(frame, (p3_x + 30, p3_y - 10), (p3_x + 60, p3_y - 10), (200, 240, 200), 5) # Horizontal legs

        # Add CCTV camera timestamp watermark
        cv2.putText(frame, f"CAM-01 [LOBBY & VAULT] FRAME: {f:04d}", (width - 450, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"[SUCCESS] Synthetic bank test video created: {output_path} ({total_frames} frames)")

if __name__ == "__main__":
    generate_bank_test_video()
