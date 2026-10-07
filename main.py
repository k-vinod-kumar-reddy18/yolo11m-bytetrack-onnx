import os
import runpy


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# EXISTING INFERENCE FILES
# ============================================================

PYTORCH_FILE = os.path.join(
    PROJECT_ROOT,
    "03_bytetrack",
    "run_bytetrack.py"
)

ONNX_FILE = os.path.join(
    PROJECT_ROOT,
    "04_onnx",
    "onnx_bytetrack.py"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(PYTORCH_FILE):

    print("ERROR: PyTorch inference file not found:")
    print(PYTORCH_FILE)
    raise SystemExit(1)


if not os.path.exists(ONNX_FILE):

    print("ERROR: ONNX inference file not found:")
    print(ONNX_FILE)
    raise SystemExit(1)


# ============================================================
# MODEL SELECTION
# ============================================================

print()
print("=" * 60)
print("       YOLO11m + ByteTrack Inference")
print("=" * 60)

print()
print("Select Inference Model")
print()
print("1. PyTorch")
print("2. ONNX")
print()

while True:

    choice = input(
        "Enter your choice (1/2): "
    ).strip()

    if choice == "1":

        print()
        print("=" * 60)
        print("Running PyTorch YOLO11m + ByteTrack")
        print("=" * 60)
        print()

        runpy.run_path(
            PYTORCH_FILE,
            run_name="__main__"
        )

        break


    elif choice == "2":

        print()
        print("=" * 60)
        print("Running ONNX YOLO11m + ByteTrack")
        print("=" * 60)
        print()

        runpy.run_path(
            ONNX_FILE,
            run_name="__main__"
        )

        break


    else:

        print(
            "Invalid choice. Please enter 1 or 2."
        )