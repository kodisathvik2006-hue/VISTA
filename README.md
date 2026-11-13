# VISTA (NEUROVISION) — Video Intelligence & Smart Tracking Analytics

**VISTA** (Neurovision) is an end-to-end cognitive video intelligence platform designed for automated multi-format video ingestion, neural object detection, multi-target trajectory tracking, and automated spatial-temporal incident anomaly scoring.

- **Phase 1 (Completed)**: Production-grade video stream ingestion, OpenCV container verification, frame-0 thumbnail generation, and metadata extraction.
- **Phase 2 (Completed)**: Neural object detection with YOLOv8/v11, multi-target trajectory tracking via ByteTrack with Kalman filtering, background analysis jobs, annotated MP4 video rendering, and an interactive React detection explorer.

---

## 🎯 Project Objectives

### Primary Objective
To build a high-throughput, real-time cognitive video intelligence platform that automates video stream ingestion, neural object detection, multi-target trajectory tracking, and automated spatial-temporal incident anomaly scoring for public and industrial safety.

### Specific Objectives
1. **Resilient Multi-Format Ingestion**: Stream video uploads in chunked buffers supporting `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm` formats with strict payload capping (200MB) and empty-file guards.
2. **Deep Container Inspection**: Probe stream properties directly at the binary container and frame-0 levels using OpenCV (`cv2.VideoCapture`), rejecting corrupted or disguised non-video files.
3. **Dynamic Snapshot Generation**: Extract and encode the first valid frame as a high-quality JPEG thumbnail, serving it via path-isolated, UUID-governed endpoints.
4. **Neural Object Detection (Phase 2)**: Deploy quantized YOLOv8 models to identify vehicles (`car`, `truck`, `bus`, `motorcycle`, `bicycle`), pedestrians (`person`), and traffic objects with low inference latency.
5. **Multi-Target Trajectory Association (Phase 2)**: Implement ByteTrack with Kalman filtering to maintain persistent track IDs across visual occlusions.
6. **Memory-Efficient Stream Processing (Phase 2)**: Stream video frames sequentially ($O(1)$ memory footprint), generating annotated MP4 output and evidence frames without loading entire videos into RAM.
7. **Background Job Engine (Phase 2)**: Non-blocking asynchronous job orchestration with real-time progress percentage, frame counters, metrics summary, and cleanup.
8. **Cognitive Incident Scoring (Phase 3)**: Compute kinematic velocity vectors, collision courses, lane violations, and wrong-way motions, persisting structured incident logs into relational databases (PostgreSQL/SQLite).

---

## 🔬 AI Model & Tracking Details (Phase 2)

### 1. Detection Model Architecture
- **Model**: `YOLOv8n` (Nano) or `YOLOv8s` / `YOLO11n` (configurable via `YOLO_MODEL` in `.env`).
- **Framework**: PyTorch (`torch>=2.0.0`) via official `ultralytics` library.
- **Weights Size**: ~6.2 MB (`yolov8n.pt`).
- **Pretrained Dataset**: COCO dataset (80 object classes).
- **Core Targeted Classes**:
  - Vehicles: `car` (2), `truck` (7), `bus` (5), `motorcycle` (3), `bicycle` (1)
  - Pedestrians: `person` (0)
  - Traffic signs & signals: `traffic light` (9), `stop sign` (11)
- **License**: AGPL-3.0 (Ultralytics Open Source).

### 2. Multi-Target Tracking (ByteTrack)
- **Tracker**: `ByteTrack` (`bytetrack.yaml` tracker configuration).
- **Association Method**: Two-stage bipartite matching associating both high-confidence and low-confidence detection boxes to recover tracks under motion blur and occlusions.
- **Motion Prediction**: Kalman filter estimating spatial bounding box velocity vectors.
- **Outputs**: Integer persistent `track_id` assigned to each detection across consecutive frames.

### 3. Model Downloads & Weight Storage
- **Automatic Download**: Ultralytics automatically downloads `yolov8n.pt` into the backend directory upon first run.
- **Manual Pre-Download** (optional, for air-gapped systems):
  ```powershell
  cd backend
  .\.venv\Scripts\python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  ```
- **Git Security**: Model weight files (`*.pt`, `*.pth`, `*.onnx`, `*.engine`) and runtime outputs are ignored by `.gitignore`.

---

## 💻 Hardware Setup (CPU vs GPU)

The platform automatically detects available compute devices (`DEVICE=auto`).

### CPU Mode (Default for standard PCs)
PyTorch CPU wheels are installed by default, requiring no external CUDA drivers:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics lap
```
- **Performance**: ~15–30 FPS on modern multi-core x86_64 CPUs with `yolov8n`.
- **RAM Footprint**: Under 300 MB during active inference.

### GPU Mode (NVIDIA CUDA Acceleration)
For high-throughput or batch video analysis on NVIDIA GPUs:
1. Ensure NVIDIA Drivers and CUDA Toolkit (11.8 or 12.1+) are installed.
2. Install PyTorch with CUDA support:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics lap
   ```
3. Set `DEVICE=cuda` (or leave as `DEVICE=auto`) in `backend/.env`.

---

## 📁 Repository Layout

```
vista/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py         # Endpoints for health, videos, and jobs
│   │   ├── services/
│   │   │   ├── annotator.py      # Bounding boxes, pills, and MP4 generation
│   │   │   ├── detector.py       # YOLOv8 + ByteTrack pipeline wrapper
│   │   │   ├── jobs.py           # Background JobManager & worker thread
│   │   │   ├── storage.py        # Path isolation, UUID validation, and cleanup
│   │   │   └── video.py          # OpenCV metadata probing & thumbnail
│   │   ├── config.py             # Pydantic settings & path resolvers
│   │   ├── exceptions.py         # Standardized HTTP application errors
│   │   ├── main.py               # FastAPI application factory & CORS
│   │   └── schemas.py            # Pydantic request/response schemas
│   ├── data/                     # Dedicated storage (ignored by git)
│   │   ├── uploads/              # Ingested raw videos ({uuid}.ext)
│   │   ├── frames/               # First-frame preview thumbnails ({uuid}.jpg)
│   │   ├── jobs/                 # Annotated videos & results JSON
│   │   └── evidence/             # Extracted keyframe evidence JPEGs
│   ├── tests/
│   │   ├── test_health.py        # Health endpoint checks
│   │   ├── test_analyze.py       # Phase 1 video ingestion & validation
│   │   ├── test_frames.py        # Path traversal & preview security
│   │   ├── test_jobs_mocked.py   # Deterministic mock tests for job pipeline
│   │   └── test_inference_real.py# Real YOLOv8 + ByteTrack inference tests
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts            # Typed client API services
│   │   ├── types/
│   │   │   └── api.ts            # TypeScript interfaces
│   │   ├── App.tsx               # Dashboard with Video Ingest & AI Tracking
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
└── README.md
```

---

## 🚀 Getting Started (Windows PowerShell)

### Backend Setup

```powershell
cd backend
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health check: `http://127.0.0.1:8000/api/health`
- Supported classes: `http://127.0.0.1:8000/api/models/classes`

### Frontend Setup

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` calls directly to `http://127.0.0.1:8000`.

### Running Tests

```powershell
cd backend
.\.venv\Scripts\python -m pytest tests
```

---

## 📡 API Specification

| Method | Path | Phase | Purpose |
| --- | --- | :---: | --- |
| `GET` | `/api/health` | 1 & 2 | Service status, version, OpenCV status, AI tracking readiness, and compute device (`CPU`/`CUDA`) |
| `POST` | `/api/videos/analyze` | 1 | Multipart `file` upload; validates readability, extracts duration/FPS/resolution, generates preview frame |
| `GET` | `/api/videos/{video_id}/preview` | 1 | Path-isolated JPEG preview frame |
| `GET` | `/api/models/classes` | 2 | Returns 80 COCO classes and default vehicle/pedestrian tracking classes |
| `POST` | `/api/jobs` | 2 | Submits an asynchronous detection & tracking job (`video_id`, `confidence_threshold`, `classes`) |
| `GET` | `/api/jobs` | 2 | Lists all analysis jobs |
| `GET` | `/api/jobs/{job_id}` | 2 | Polls job status, progress percentage, current frame, and summary metrics |
| `GET` | `/api/jobs/{job_id}/results` | 2 | Returns complete frame-by-frame detection records with timestamps, bboxes, and track IDs |
| `GET` | `/api/jobs/{job_id}/annotated-video` | 2 | Streams/downloads the rendered annotated MP4 video |
| `GET` | `/api/jobs/{job_id}/evidence/{frame_index}` | 2 | Retrieves an evidence keyframe image |
| `DELETE` | `/api/jobs/{job_id}` | 2 | Cancels an ongoing job and purges associated output files |
