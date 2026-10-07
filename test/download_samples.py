"""
Sample Video and Model Downloader for Bank Anomaly Test Suite
Strictly downloads all test videos and YOLO model weights into the `test/` folder only (Rule 7).
"""

import os
import sys
import ssl
import urllib.request

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(WORKSPACE_DIR, "test")
MODELS_DIR = os.path.join(TEST_DIR, "models")
VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
OUTPUTS_DIR = os.path.join(TEST_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Pinned storage containment
os.environ["TORCH_HOME"] = MODELS_DIR
os.environ["YOLO_CONFIG_DIR"] = MODELS_DIR

# Handle macOS SSL certificate verification
ssl_context = ssl._create_unverified_context()

def download_file(url: str, dest_path: str, description: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[EXISTS] {description} already present at: {dest_path}")
        return

    print(f"[DOWNLOADING] {description} from {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[SUCCESS] Saved to {dest_path} ({os.path.getsize(dest_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[ERROR] Failed downloading {description}: {e}")

def main():
    print("="*60)
    print("DOWNLOADING BENCHMARK SAMPLES INTO ORGANIZED SUBDIRECTORIES")
    print("="*60)

    # 1. Download YOLO-Pose Model directly into test/models/
    model_dest = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    download_file(model_url, model_dest, "YOLO11-Nano Pose Model Weights")

    # Fallback model weights
    fallback_dest = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
    fallback_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt"
    download_file(fallback_url, fallback_dest, "YOLOv8-Nano Pose Model Weights")

    # 2. Download Real CCTV / Indoor People Surveillance Video (Intel IoT Benchmark)
    surveillance_video_dest = os.path.join(VIDEOS_DIR, "cctv_people_surveillance.mp4")
    surveillance_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    download_file(surveillance_url, surveillance_video_dest, "Indoor CCTV Surveillance Test Video")

    # 3. Download Real-Life Indoor Walking / Seated Demographic Surveillance Footage
    walking_video_dest = os.path.join(VIDEOS_DIR, "real_life_indoor_cctv.mp4")
    walking_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
    download_file(walking_url, walking_video_dest, "Real-Life Indoor Surveillance CCTV Video")

    # 4. Download Street / Indoor Multi-Person Traffic Video
    multi_person_dest = os.path.join(VIDEOS_DIR, "multi_person_traffic.mp4")
    multi_person_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    download_file(multi_person_url, multi_person_dest, "Multi-Target Walking & Zone Test Video")

    print("\n[COMPLETE] All test assets organized in:", TEST_DIR)

if __name__ == "__main__":
    main()
