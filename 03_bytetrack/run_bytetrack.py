from ultralytics import YOLO
from pathlib import Path
import cv2
import time
import os
import json
import statistics
import psutil


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "./01_model/yolo11m.pt"
VIDEO_PATH = "./02_input/videos/PNNL_Parking_LOT(1).mp4"

OUTPUT_DIR = Path("./05_results/pytorch")
REPORT_DIR = Path("./05_results/reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "bytetrack_output.mp4"
REPORT_JSON = REPORT_DIR / "bytetrack_pytorch_report.json"
REPORT_LOG = REPORT_DIR / "bytetrack_pytorch_report.log"


# ============================================================
# SYSTEM
# ============================================================

process = psutil.Process(os.getpid())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024)


def get_video_size_mb(path):
    if Path(path).exists():
        return Path(path).stat().st_size / (1024 * 1024)
    return None


def get_gpu_info():
    """
    Detect CUDA availability.

    This system is currently expected to run on CPU.
    GPU utilization is therefore reported as N/A if CUDA
    is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return {
                "available": True,
                "name": gpu_name
            }

    except Exception:
        pass

    return {
        "available": False,
        "name": None
    }


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11m + ByteTrack PERFORMANCE REPORT")
print("=" * 70)

print("\nLoading YOLO11m...")

model_load_start = time.perf_counter()

model = YOLO(MODEL_PATH)

model_load_time = time.perf_counter() - model_load_start

print("YOLO11m loaded successfully")


# ============================================================
# GPU INFORMATION
# ============================================================

gpu_info = get_gpu_info()

if gpu_info["available"]:
    print(f"GPU: {gpu_info['name']}")
else:
    print("GPU: Not available")
    print("Using CPU")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


input_fps = cap.get(cv2.CAP_PROP_FPS)
total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print("\nVideo Information")
print("-" * 70)
print(f"Resolution       : {width} x {height}")
print(f"Input FPS        : {input_fps:.2f}")
print(f"Total Frames     : {total_input_frames}")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    input_fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video.")


# ============================================================
# METRIC STORAGE
# ============================================================

frame_count = 0

latencies = []

cpu_values = []
ram_values = []

gpu_values = []
gpu_memory_values = []

total_detections = 0

unique_track_ids = set()

active_tracks_per_frame = []

# Track lifetime:
# track_id -> number of frames where it was active
track_lifetimes = {}

# Previous active IDs
previous_active_ids = set()

# Number of times a previously active track disappears
track_loss_events = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print("\nStarting ByteTrack processing...")
print("Press Q to stop early.\n")

processing_start = time.perf_counter()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------------------------------
    # Start frame timer
    # --------------------------------------------------------

    frame_start = time.perf_counter()

    # --------------------------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------------------------

    results = model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.25,
        iou=0.5,
        verbose=False
    )

    # --------------------------------------------------------
    # End frame timer
    # --------------------------------------------------------

    frame_end = time.perf_counter()

    latency_ms = (frame_end - frame_start) * 1000

    latencies.append(latency_ms)

    # --------------------------------------------------------
    # CPU / RAM
    # --------------------------------------------------------

    cpu_percent = process.cpu_percent(interval=None)
    ram_mb = process.memory_info().rss / (1024 * 1024)

    cpu_values.append(cpu_percent)
    ram_values.append(ram_mb)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if gpu_info["available"]:

        try:
            import torch

            gpu_memory_mb = (
                torch.cuda.memory_allocated(0)
                / (1024 * 1024)
            )

            gpu_memory_values.append(gpu_memory_mb)

            # GPU utilization cannot be reliably obtained
            # from torch alone.
            gpu_values.append(None)

        except Exception:
            pass

    # --------------------------------------------------------
    # Extract detection / tracking information
    # --------------------------------------------------------

    result = results[0]

    boxes = result.boxes

    frame_detection_count = 0
    current_active_ids = set()

    if boxes is not None:

        frame_detection_count = len(boxes)

        total_detections += frame_detection_count

        # ----------------------------------------------------
        # Track IDs
        # ----------------------------------------------------

        if boxes.id is not None:

            ids = boxes.id.int().cpu().tolist()

            for track_id in ids:

                track_id = int(track_id)

                unique_track_ids.add(track_id)

                current_active_ids.add(track_id)

                # Track lifetime
                track_lifetimes[track_id] = (
                    track_lifetimes.get(track_id, 0) + 1
                )

    # --------------------------------------------------------
    # Active tracks
    # --------------------------------------------------------

    active_track_count = len(current_active_ids)

    active_tracks_per_frame.append(active_track_count)

    # --------------------------------------------------------
    # Track loss events
    #
    # A track loss event is counted when an ID that was active
    # in the previous frame is absent in the current frame.
    #
    # This is an operational tracking-loss measurement,
    # NOT a formal ground-truth ID-switch metric.
    # --------------------------------------------------------

    lost_tracks = previous_active_ids - current_active_ids

    track_loss_events += len(lost_tracks)

    previous_active_ids = current_active_ids.copy()

    # --------------------------------------------------------
    # Draw / save result
    # --------------------------------------------------------

    annotated_frame = result.plot()

    writer.write(annotated_frame)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_frame = cv2.resize(
        annotated_frame,
        (960, 540)
    )

    cv2.putText(
        display_frame,
        f"FPS: {1 / (latency_ms / 1000):.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"CPU: {cpu_percent:.1f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"RAM: {ram_mb:.1f} MB",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"Active Tracks: {active_track_count}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO11m + ByteTrack",
        display_frame
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_count % 50 == 0:

        elapsed = time.perf_counter() - processing_start

        current_fps = frame_count / elapsed

        print(
            f"Frame {frame_count}/{total_input_frames} | "
            f"FPS: {current_fps:.2f} | "
            f"Active Tracks: {active_track_count}"
        )

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("\nProcessing stopped by user.")
        break


# ============================================================
# CLEANUP
# ============================================================

processing_end = time.perf_counter()

total_processing_time = processing_end - processing_start

cap.release()
writer.release()
cv2.destroyAllWindows()


# ============================================================
# FINAL METRICS
# ============================================================

if frame_count > 0:

    average_fps = frame_count / total_processing_time

    average_latency = statistics.mean(latencies)

    if len(latencies) >= 2:
        p95_latency = statistics.quantiles(
            latencies,
            n=100
        )[94]
    else:
        p95_latency = latencies[0]

    average_cpu = statistics.mean(cpu_values)
    peak_cpu = max(cpu_values)

    average_ram = statistics.mean(ram_values)
    peak_ram = max(ram_values)

else:

    average_fps = 0
    average_latency = 0
    p95_latency = 0

    average_cpu = 0
    peak_cpu = 0

    average_ram = 0
    peak_ram = 0


# ============================================================
# TRACKING METRICS
# ============================================================

unique_id_count = len(unique_track_ids)

if active_tracks_per_frame:

    average_active_tracks = statistics.mean(
        active_tracks_per_frame
    )

    maximum_active_tracks = max(
        active_tracks_per_frame
    )

else:

    average_active_tracks = 0
    maximum_active_tracks = 0


if track_lifetimes:

    average_track_lifetime = statistics.mean(
        track_lifetimes.values()
    )

    longest_track_lifetime = max(
        track_lifetimes.values()
    )

else:

    average_track_lifetime = 0
    longest_track_lifetime = 0


# ============================================================
# ID SWITCHES
# ============================================================

# Formal ID switches require ground-truth tracking annotations.
# Therefore we do NOT invent an ID-switch value.

id_switches = "N/A - ground-truth tracking annotations not available"


# ============================================================
# GPU METRICS
# ============================================================

if gpu_info["available"]:

    average_gpu = "N/A - GPU utilization API not configured"
    peak_gpu = "N/A - GPU utilization API not configured"

    if gpu_memory_values:
        peak_gpu_memory = max(gpu_memory_values)
    else:
        peak_gpu_memory = "N/A"

else:

    average_gpu = "N/A - CPU-only system"
    peak_gpu = "N/A - CPU-only system"
    peak_gpu_memory = "N/A - CPU-only system"


# ============================================================
# OUTPUT VIDEO SIZE
# ============================================================

output_video_size_mb = get_video_size_mb(
    OUTPUT_VIDEO
)


# ============================================================
# REPORT DATA
# ============================================================

report = {

    "experiment": {
        "model": "YOLO11m",
        "format": "PyTorch",
        "tracker": "ByteTrack",
        "input_video": VIDEO_PATH,
        "output_video": str(OUTPUT_VIDEO)
    },

    "model": {
        "model_size_mb": round(
            get_model_size_mb(MODEL_PATH),
            2
        ),

        "model_load_time_sec": round(
            model_load_time,
            3
        )
    },

    "video": {
        "resolution": f"{width}x{height}",

        "input_fps": round(
            input_fps,
            2
        ),

        "total_frames": frame_count
    },

    "performance": {

        "total_processing_time_sec": round(
            total_processing_time,
            2
        ),

        "average_fps": round(
            average_fps,
            2
        ),

        "average_latency_ms": round(
            average_latency,
            2
        ),

        "p95_latency_ms": round(
            p95_latency,
            2
        )
    },

    "detections": {

        "total_detections": total_detections,

        "average_detections_per_frame": round(
            total_detections / frame_count,
            2
        ) if frame_count else 0
    },

    "tracking": {

        "unique_track_ids": unique_id_count,

        "average_active_tracks": round(
            average_active_tracks,
            2
        ),

        "maximum_active_tracks": maximum_active_tracks,

        "average_track_lifetime_frames": round(
            average_track_lifetime,
            2
        ),

        "longest_track_lifetime_frames": longest_track_lifetime,

        "id_switches": id_switches,

        "track_loss_events": track_loss_events
    },

    "system": {

        "average_cpu_percent": round(
            average_cpu,
            2
        ),

        "peak_cpu_percent": round(
            peak_cpu,
            2
        ),

        "average_gpu_percent": average_gpu,

        "peak_gpu_percent": peak_gpu,

        "average_ram_mb": round(
            average_ram,
            2
        ),

        "peak_ram_mb": round(
            peak_ram,
            2
        ),

        "peak_gpu_memory_mb": peak_gpu_memory
    },

    "output": {

        "video_size_mb": round(
            output_video_size_mb,
            2
        ) if output_video_size_mb is not None else None,

        "video_path": str(
            OUTPUT_VIDEO
        ),

        "json_report": str(
            REPORT_JSON
        ),

        "log_file": str(
            REPORT_LOG
        )
    },

    "notes": [

        "CPU percentages may exceed 100% because psutil reports process utilization across CPU cores.",

        "GPU metrics are N/A because this system is running without CUDA GPU.",

        "ID switches require ground-truth tracking annotations and are therefore not calculated.",

        "Track loss events represent disappearance of an active track between consecutive frames and are not equivalent to formal ID-switch evaluation.",

        "Track lifetime is measured in frames."
    ]
}


# ============================================================
# SAVE JSON REPORT
# ============================================================

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# SAVE HUMAN-READABLE LOG
# ============================================================

with open(
    REPORT_LOG,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("YOLO11m + ByteTrack PERFORMANCE REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL\n")
    f.write("-" * 70 + "\n")
    f.write("Model              : YOLO11m\n")
    f.write("Format             : PyTorch\n")
    f.write("Tracker            : ByteTrack\n")
    f.write(
        f"Model Size         : "
        f"{report['model']['model_size_mb']} MB\n"
    )
    f.write(
        f"Model Load Time    : "
        f"{report['model']['model_load_time_sec']} sec\n\n"
    )

    f.write("VIDEO\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Resolution         : "
        f"{report['video']['resolution']}\n"
    )
    f.write(
        f"Input FPS          : "
        f"{report['video']['input_fps']}\n"
    )
    f.write(
        f"Total Frames       : "
        f"{report['video']['total_frames']}\n\n"
    )

    f.write("PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Total Processing Time : "
        f"{report['performance']['total_processing_time_sec']} sec\n"
    )
    f.write(
        f"Average FPS           : "
        f"{report['performance']['average_fps']}\n"
    )
    f.write(
        f"Average Latency       : "
        f"{report['performance']['average_latency_ms']} ms\n"
    )
    f.write(
        f"P95 Latency           : "
        f"{report['performance']['p95_latency_ms']} ms\n\n"
    )

    f.write("DETECTIONS\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Total Detections      : "
        f"{report['detections']['total_detections']}\n"
    )
    f.write(
        f"Average Detections/Frame : "
        f"{report['detections']['average_detections_per_frame']}\n\n"
    )

    f.write("TRACKING\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Unique Track IDs      : "
        f"{report['tracking']['unique_track_ids']}\n"
    )
    f.write(
        f"Average Active Tracks : "
        f"{report['tracking']['average_active_tracks']}\n"
    )
    f.write(
        f"Maximum Active Tracks : "
        f"{report['tracking']['maximum_active_tracks']}\n"
    )
    f.write(
        f"Average Track Lifetime: "
        f"{report['tracking']['average_track_lifetime_frames']} frames\n"
    )
    f.write(
        f"Longest Track Lifetime: "
        f"{report['tracking']['longest_track_lifetime_frames']} frames\n"
    )
    f.write(
        f"ID Switches           : "
        f"{report['tracking']['id_switches']}\n"
    )
    f.write(
        f"Track Loss Events     : "
        f"{report['tracking']['track_loss_events']}\n\n"
    )

    f.write("SYSTEM\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Average CPU           : "
        f"{report['system']['average_cpu_percent']}%\n"
    )
    f.write(
        f"Peak CPU              : "
        f"{report['system']['peak_cpu_percent']}%\n"
    )
    f.write(
        f"Average GPU           : "
        f"{report['system']['average_gpu_percent']}\n"
    )
    f.write(
        f"Peak GPU              : "
        f"{report['system']['peak_gpu_percent']}\n"
    )
    f.write(
        f"Average RAM           : "
        f"{report['system']['average_ram_mb']} MB\n"
    )
    f.write(
        f"Peak RAM              : "
        f"{report['system']['peak_ram_mb']} MB\n"
    )
    f.write(
        f"Peak GPU Memory       : "
        f"{report['system']['peak_gpu_memory_mb']}\n\n"
    )

    f.write("OUTPUT\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Output Video Size     : "
        f"{report['output']['video_size_mb']} MB\n"
    )
    f.write(
        f"Output Video          : "
        f"{report['output']['video_path']}\n"
    )
    f.write(
        f"JSON Report           : "
        f"{report['output']['json_report']}\n"
    )
    f.write(
        f"Log File              : "
        f"{report['output']['log_file']}\n"
    )

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 70 + "\n")


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("YOLO11m + ByteTrack REPORT COMPLETED")
print("=" * 70)

print(f"Model                  : YOLO11m")
print(f"Format                 : PyTorch")
print(f"Tracker                : ByteTrack")

print(f"Model Size             : {report['model']['model_size_mb']} MB")
print(f"Model Load Time        : {report['model']['model_load_time_sec']} sec")

print(f"Total Frames           : {frame_count}")
print(f"Total Detections       : {total_detections}")
print(f"Unique Track IDs       : {unique_id_count}")

print(
    f"Average Active Tracks  : "
    f"{average_active_tracks:.2f}"
)

print(
    f"Maximum Active Tracks  : "
    f"{maximum_active_tracks}"
)

print(
    f"Average Track Lifetime : "
    f"{average_track_lifetime:.2f} frames"
)

print(
    f"Longest Track Lifetime : "
    f"{longest_track_lifetime} frames"
)

print(
    f"Track Loss Events      : "
    f"{track_loss_events}"
)

print(
    f"ID Switches            : "
    f"{id_switches}"
)

print(
    f"Total Processing Time  : "
    f"{total_processing_time:.2f} sec"
)

print(
    f"Average FPS            : "
    f"{average_fps:.2f}"
)

print(
    f"Average Latency        : "
    f"{average_latency:.2f} ms"
)

print(
    f"P95 Latency            : "
    f"{p95_latency:.2f} ms"
)

print(
    f"Average CPU            : "
    f"{average_cpu:.2f}%"
)

print(
    f"Peak CPU               : "
    f"{peak_cpu:.2f}%"
)

print(
    f"Average RAM            : "
    f"{average_ram:.2f} MB"
)

print(
    f"Peak RAM               : "
    f"{peak_ram:.2f} MB"
)

print(
    f"Average GPU            : "
    f"{average_gpu}"
)

print(
    f"Peak GPU               : "
    f"{peak_gpu}"
)

print(
    f"Peak GPU Memory        : "
    f"{peak_gpu_memory}"
)

print(
    f"Output Video Size      : "
    f"{output_video_size_mb:.2f} MB"
    if output_video_size_mb is not None
    else "N/A"
)

print("\nOutput Video:")
print(OUTPUT_VIDEO)

print("\nJSON Report:")
print(REPORT_JSON)

print("\nLog File:")
print(REPORT_LOG)

print("=" * 70)