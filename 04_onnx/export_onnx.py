from ultralytics import YOLO
from pathlib import Path

# ==========================================
# YOLO11m -> ONNX
# ==========================================

MODEL_PATH = "../01_model/yolo11m.pt"

print("Loading YOLO11m...")

model = YOLO(MODEL_PATH)

print("YOLO11m loaded successfully")
print("Starting ONNX export...")

# Export YOLO11m to ONNX
onnx_path = model.export(
    format="onnx",
    opset=12,
    simplify=True
)

print()
print("=" * 50)
print("ONNX Export Completed")
print("=" * 50)
print(f"ONNX Model: {onnx_path}")
print("=" * 50)