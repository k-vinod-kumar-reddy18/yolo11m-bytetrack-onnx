YOLO11m + ByteTrack + ONNX
YOLO11m object detection and ByteTrack multi-object tracking with PyTorch and ONNX performance benchmarking.
Features
- YOLO11m object detection
- ByteTrack object tracking
- PyTorch inference
- ONNX conversion and inference
- FPS and latency measurement
- P95 latency
- Detection and tracking metrics
- CPU, GPU and RAM monitoring
- Output video generation
- JSON and LOG performance reports
Project Structure
yolo11m-bytetrack-onnx/
├── 01_model/
├── 02_input/
├── 03_bytetrack/
├── 04_onnx/
├── 05_results/
│   ├── pytorch/
│   ├── onnx/
│   └── reports/
└── requirements.txt

Run PyTorch
.\.venv\Scripts\python.exe .\03_bytetrack\run_bytetrack.py

Convert to ONNX
.\.venv\Scripts\python.exe .\04_onnx\export_onnx.py

Run ONNX
.\.venv\Scripts\python.exe .\04_onnx\onnx_bytetrack.py

Metrics
The generated reports contain:
- Total Processing Time
- Average FPS
- Average Latency
- P95 Latency
- Total Frames
- Total Detections
- Unique Track IDs
- Average Active Tracks
- Maximum Active Tracks
- Average Track Lifetime
- Longest Track Lifetime
- Track Losses
- ID Switches
- Average/Peak CPU
- Average/Peak GPU
- Average/Peak RAM
- Peak GPU Memory
- Output Video Size
Outputs
05_results/
├── pytorch/
│   └── bytetrack_output.mp4
├── onnx/
│   └── bytetrack_onnx_output.mp4
└── reports/
    ├── bytetrack_pytorch_report.json
    ├── bytetrack_pytorch_report.log
    ├── bytetrack_onnx_report.json
    └── bytetrack_onnx_report.log

Technologies
Python | YOLO11m | ByteTrack | ONNX | ONNX Runtime | OpenCV | PyTorch | NumPy | psutil