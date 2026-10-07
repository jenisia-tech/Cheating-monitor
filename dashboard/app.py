"""
ProctorVision AI - Autonomous Exam Hall Surveillance Web Server & API
FastAPI Backend with Local Containment & Dual-Core Intel i5 Optimization
"""

import os
import sys
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Workspace Paths
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(DASHBOARD_DIR)
TEST_DIR = os.path.join(WORKSPACE_DIR, "test")
VIDEOS_DIR = os.path.join(TEST_DIR, "videos")
OUTPUTS_DIR = os.path.join(TEST_DIR, "outputs")
STATIC_DIR = os.path.join(DASHBOARD_DIR, "static")

sys.path.insert(0, TEST_DIR)
from exam_analyzer import ExamHallAnomalyAnalyzer

app = FastAPI(title="ProctorVision AI Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>ProctorVision Dashboard Loaded</h1>")


@app.get("/media/video/{filename}")
async def serve_video(filename: str):
    # Check outputs first (annotated video), then videos
    annotated = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(annotated):
        return FileResponse(annotated, media_type="video/mp4")
    raw = os.path.join(VIDEOS_DIR, filename)
    if os.path.exists(raw):
        return FileResponse(raw, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


@app.post("/api/analyze")
async def analyze_uploaded_video(video: UploadFile = File(...)):
    """Uploads video, runs frame-by-frame anomaly detection, returns structured incident JSON."""
    try:
        # Sanitize filename
        safe_filename = os.path.basename(video.filename)
        dest_video_path = os.path.join(VIDEOS_DIR, safe_filename)

        with open(dest_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        base_name = os.path.splitext(safe_filename)[0]
        out_json = os.path.join(OUTPUTS_DIR, f"{base_name}_exam_incidents.json")
        out_video = os.path.join(OUTPUTS_DIR, f"{base_name}_exam_annotated.mp4")

        # Run Analyzer
        analyzer = ExamHallAnomalyAnalyzer(frame_stride=1)
        report = analyzer.process_video(
            video_path=dest_video_path,
            output_json_path=out_json,
            save_annotated_video=True,
            output_video_path=out_video
        )

        return JSONResponse({
            "status": "success",
            "filename": safe_filename,
            "video_url": f"/media/video/{base_name}_exam_annotated.mp4",
            "report": report
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze_preset")
async def analyze_preset_video(filename: str = Query(...)):
    """Runs anomaly detection on an existing video in test/videos/."""
    try:
        dest_video_path = os.path.join(VIDEOS_DIR, filename)
        if not os.path.exists(dest_video_path):
            raise HTTPException(status_code=404, detail=f"Preset video {filename} not found.")

        base_name = os.path.splitext(filename)[0]
        out_json = os.path.join(OUTPUTS_DIR, f"{base_name}_exam_incidents.json")
        out_video = os.path.join(OUTPUTS_DIR, f"{base_name}_exam_annotated.mp4")

        analyzer = ExamHallAnomalyAnalyzer(frame_stride=2 if "irl_exam_hall" in filename else 1)
        report = analyzer.process_video(
            video_path=dest_video_path,
            output_json_path=out_json,
            save_annotated_video=True,
            output_video_path=out_video,
            max_frames=600 if "irl_exam_hall" in filename else None
        )

        return JSONResponse({
            "status": "success",
            "filename": filename,
            "video_url": f"/media/video/{base_name}_exam_annotated.mp4",
            "report": report
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(port: int = 8000):
    print(f"[INFO] Launching ProctorVision Server on http://localhost:{port}...")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False, app_dir=DASHBOARD_DIR)


if __name__ == "__main__":
    run_server(8000)
