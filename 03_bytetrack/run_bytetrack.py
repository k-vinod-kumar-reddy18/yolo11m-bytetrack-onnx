import os
import json
import time
import statistics

import cv2
import psutil

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "./01_model/yolo11m.pt"

VIDEO_PATH = "./02_input/videos/PNNL_Parking_LOT(1).mp4"

OUTPUT_VIDEO = "./05_results/pytorch/bytetrack_output.mp4"

OUTPUT_JSON = "./05_results/reports/bytetrack_pytorch_report.json"

OUTPUT_LOG = "./05_results/reports/bytetrack_pytorch_report.log"


# ============================================================
# CONFIGURATION
# ============================================================

TRACKER = "bytetrack.yaml"

CONF = 0.25

IOU = 0.5

DEVICE = "cpu"


# ============================================================
# LIVE WINDOW
# ============================================================

SHOW_WINDOW = True

WINDOW_NAME = "YOLO11m + ByteTrack - PyTorch"

DISPLAY_WIDTH = 1280

DISPLAY_HEIGHT = 720


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    "./05_results/pytorch",
    exist_ok=True
)

os.makedirs(
    "./05_results/reports",
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def get_file_size_mb(path):

    if os.path.exists(path):

        return round(
            os.path.getsize(path)
            / (1024 * 1024),
            2
        )

    return 0.0


# ============================================================
# START PIPELINE
# ============================================================

pipeline_start = time.perf_counter()


print("=" * 70)

print("YOLO11m + ByteTrack - PYTORCH")

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print("Loading YOLO11m PyTorch model...")


model_load_start = time.perf_counter()


model = YOLO(
    MODEL_PATH
)


model_load_time = (
    time.perf_counter()
    - model_load_start
)


print(
    "YOLO11m loaded successfully"
)


print(
    f"Model Load Time: "
    f"{model_load_time:.3f} sec"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Unable to open video: {VIDEO_PATH}"
    )


fps_input = cap.get(
    cv2.CAP_PROP_FPS
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


total_frames_expected = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print()

print(
    f"Video Resolution: "
    f"{width}x{height}"
)


print(
    f"Input FPS: "
    f"{fps_input}"
)


print(
    f"Total Frames: "
    f"{total_frames_expected}"
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps_input,
    (width, height)
)


if not out.isOpened():

    cap.release()

    raise RuntimeError(
        f"Unable to create output video: "
        f"{OUTPUT_VIDEO}"
    )


# ============================================================
# METRIC STORAGE
# ============================================================

latencies = []

process_cpu_values = []

system_cpu_values = []

ram_values = []


total_detections = 0

unique_track_ids = set()

active_track_counts = []


track_first_frame = {}

track_last_frame = {}


previous_active_ids = set()

track_loss_events = 0


frame_count = 0


# ============================================================
# PROCESS MONITOR
# ============================================================

process = psutil.Process(
    os.getpid()
)

# Initialize process CPU measurement.
# This must be done ONCE before the frame loop.
process.cpu_percent(
    interval=None
)

# Initialize system CPU measurement.
psutil.cpu_percent(
    interval=None
)


# ============================================================
# PROCESS VIDEO
# ============================================================

processing_start = time.perf_counter()


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_start = time.perf_counter()


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(

        frame,

        tracker=TRACKER,

        persist=True,

        conf=CONF,

        iou=IOU,

        device=DEVICE,

        verbose=False

    )


    result = results[0]


    current_ids = set()

    detection_count = 0


    # ========================================================
    # DETECTIONS + TRACK IDs
    # ========================================================

    if result.boxes is not None:

        boxes = result.boxes


        detection_count = len(
            boxes
        )


        total_detections += (
            detection_count
        )


        if boxes.id is not None:

            ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )


            for track_id in ids:

                track_id = int(
                    track_id
                )


                current_ids.add(
                    track_id
                )


                unique_track_ids.add(
                    track_id
                )


                if (
                    track_id
                    not in track_first_frame
                ):

                    track_first_frame[
                        track_id
                    ] = frame_count


                track_last_frame[
                    track_id
                ] = frame_count


    # ========================================================
    # ACTIVE TRACKS
    # ========================================================

    active_track_counts.append(
        len(current_ids)
    )


    # ========================================================
    # TRACK LOSS EVENTS
    # ========================================================

    lost_ids = (
        previous_active_ids
        - current_ids
    )


    track_loss_events += len(
        lost_ids
    )


    previous_active_ids = (
        current_ids
    )


    # ========================================================
    # RESOURCE MONITORING
    # ========================================================

    process_cpu = process.cpu_percent(
        interval=None
    )


    system_cpu = psutil.cpu_percent(
        interval=None
    )


    ram_mb = (
        process.memory_info().rss
        / (1024 * 1024)
    )


    process_cpu_values.append(
        process_cpu
    )


    system_cpu_values.append(
        system_cpu
    )


    ram_values.append(
        ram_mb
    )


    # ========================================================
    # FRAME LATENCY
    # ========================================================

    frame_latency = (
        time.perf_counter()
        - frame_start
    )


    latencies.append(
        frame_latency
    )


    frame_count += 1


    # ========================================================
    # DRAW OUTPUT
    # ========================================================

    annotated_frame = (
        result.plot()
    )


    # ========================================================
    # SAVE OUTPUT VIDEO
    # ========================================================

    out.write(
        annotated_frame
    )


    # ========================================================
    # LIVE WINDOW
    # ========================================================

    if SHOW_WINDOW:

        display_frame = cv2.resize(

            annotated_frame,

            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )

        )


        cv2.imshow(

            WINDOW_NAME,

            display_frame

        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == ord("q"):

            print()

            print(
                "Q pressed. "
                "Stopping processing..."
            )

            break


# ============================================================
# END PROCESSING
# ============================================================

processing_end = time.perf_counter()


total_processing_time = (
    processing_end
    - processing_start
)


pipeline_wall_time = (
    time.perf_counter()
    - pipeline_start
)


cap.release()

out.release()

cv2.destroyAllWindows()


# ============================================================
# PERFORMANCE
# ============================================================

average_fps = (

    frame_count
    / total_processing_time

    if total_processing_time > 0

    else 0

)


average_latency_ms = (

    statistics.mean(latencies)
    * 1000

    if latencies

    else 0

)


# ============================================================
# P95 LATENCY
# ============================================================

if latencies:

    sorted_latencies = sorted(
        latencies
    )


    p95_index = int(
        0.95
        * len(sorted_latencies)
    )


    if p95_index >= len(
        sorted_latencies
    ):

        p95_index = (
            len(sorted_latencies)
            - 1
        )


    p95_latency_ms = (

        sorted_latencies[
            p95_index
        ]

        * 1000

    )

else:

    p95_latency_ms = 0


# ============================================================
# TRACKING STATISTICS
# ============================================================

average_active_tracks = (

    statistics.mean(
        active_track_counts
    )

    if active_track_counts

    else 0

)


maximum_active_tracks = (

    max(
        active_track_counts
    )

    if active_track_counts

    else 0

)


# ============================================================
# TRACK LIFETIMES
# ============================================================

track_lifetimes = []


for track_id in unique_track_ids:

    first_frame = (
        track_first_frame.get(
            track_id,
            0
        )
    )


    last_frame = (
        track_last_frame.get(
            track_id,
            first_frame
        )
    )


    lifetime = (
        last_frame
        - first_frame
        + 1
    )


    track_lifetimes.append(
        lifetime
    )


average_track_lifetime = (

    statistics.mean(
        track_lifetimes
    )

    if track_lifetimes

    else 0

)


longest_track_lifetime = (

    max(
        track_lifetimes
    )

    if track_lifetimes

    else 0

)


# ============================================================
# RESOURCE STATISTICS
# ============================================================

average_process_cpu = (

    statistics.mean(
        process_cpu_values
    )

    if process_cpu_values

    else 0

)


peak_process_cpu = (

    max(
        process_cpu_values
    )

    if process_cpu_values

    else 0

)


average_system_cpu = (

    statistics.mean(
        system_cpu_values
    )

    if system_cpu_values

    else 0

)


peak_system_cpu = (

    max(
        system_cpu_values
    )

    if system_cpu_values

    else 0

)


average_ram_mb = (

    statistics.mean(
        ram_values
    )

    if ram_values

    else 0

)


peak_ram_mb = (

    max(
        ram_values
    )

    if ram_values

    else 0

)


# ============================================================
# JSON REPORT
# ============================================================

report = {

    "project": {

        "name":
            "YOLO11m + ByteTrack Object Tracking",

        "tracker":
            "ByteTrack"

    },


    "model": {

        "name":
            "YOLO11m",

        "format":
            "pt",

        "path":
            os.path.abspath(
                MODEL_PATH
            )

    },


    "configuration": {

        "confidence_threshold":
            CONF,

        "iou_threshold":
            IOU,

        "tracker_config":
            "ByteTrack",

        "device":
            "CPU"

    },


    "video": {

        "input":
            os.path.abspath(
                VIDEO_PATH
            ),

        "width":
            width,

        "height":
            height,

        "fps":
            round(
                fps_input,
                2
            ),

        "total_frames":
            frame_count

    },


    "tracking_statistics": {

        "frame_count":
            frame_count,

        "total_detections":
            total_detections,

        "unique_track_ids":
            len(
                unique_track_ids
            ),

        "average_active_tracks":
            round(
                average_active_tracks,
                10
            ),

        "maximum_active_tracks":
            maximum_active_tracks,

        "average_track_lifetime":
            round(
                average_track_lifetime,
                10
            ),

        "longest_track_lifetime":
            longest_track_lifetime,

        "average_fps":
            round(
                average_fps,
                10
            ),

        "average_latency_ms":
            round(
                average_latency_ms,
                10
            ),

        "p95_latency_ms":
            round(
                p95_latency_ms,
                10
            ),

        "total_processing_time_seconds":
            round(
                total_processing_time,
                10
            )

    },


    "resources": {

        "process": {

            "average_cpu_percent":
                round(
                    average_process_cpu,
                    10
                ),

            "peak_cpu_percent":
                round(
                    peak_process_cpu,
                    10
                ),

            "average_ram_mb":
                round(
                    average_ram_mb,
                    10
                ),

            "peak_ram_mb":
                round(
                    peak_ram_mb,
                    10
                )

        },


        "system": {

            "average_cpu_percent":
                round(
                    average_system_cpu,
                    10
                ),

            "peak_cpu_percent":
                round(
                    peak_system_cpu,
                    10
                )

        }

    },


    "outputs": {

        "video":
            os.path.abspath(
                OUTPUT_VIDEO
            ),

        "json":
            os.path.abspath(
                OUTPUT_JSON
            ),

        "log":
            os.path.abspath(
                OUTPUT_LOG
            )

    }

}


# ============================================================
# SAVE JSON
# ============================================================

with open(

    OUTPUT_JSON,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        report,

        f,

        indent=4

    )


# ============================================================
# LOG REPORT
# ============================================================

report_text = f"""

======================================================================
YOLO11m + ByteTrack OBJECT TRACKING REPORT
======================================================================

PROJECT
----------------------------------------------------------------------

Name                 : YOLO11m + ByteTrack Object Tracking

Tracker              : ByteTrack


MODEL
----------------------------------------------------------------------

Name                 : YOLO11m

Format               : PyTorch

Model Path           : {os.path.abspath(MODEL_PATH)}

Model Load Time      : {model_load_time:.6f} sec


CONFIGURATION
----------------------------------------------------------------------

Confidence Threshold : {CONF}

IoU Threshold        : {IOU}

Tracker Config       : ByteTrack

Device               : CPU


VIDEO
----------------------------------------------------------------------

Input                : {os.path.abspath(VIDEO_PATH)}

Resolution           : {width}x{height}

FPS                  : {fps_input:.2f}

Total Frames         : {frame_count}


TRACKING STATISTICS
----------------------------------------------------------------------

Frame Count              : {frame_count}

Total Detections         : {total_detections}

Unique Track IDs         : {len(unique_track_ids)}

Average Active Tracks    : {average_active_tracks:.10f}

Maximum Active Tracks    : {maximum_active_tracks}

Average Track Lifetime   : {average_track_lifetime:.10f}

Longest Track Lifetime   : {longest_track_lifetime}

Track Loss Events        : {track_loss_events}


PERFORMANCE
----------------------------------------------------------------------

Average FPS              : {average_fps:.10f}

Average Latency          : {average_latency_ms:.10f} ms

P95 Latency              : {p95_latency_ms:.10f} ms

Total Processing Time    : {total_processing_time:.10f} sec

Pipeline Wall Time       : {pipeline_wall_time:.10f} sec


RESOURCES - PROCESS
----------------------------------------------------------------------

Average CPU              : {average_process_cpu:.10f}%

Peak CPU                 : {peak_process_cpu:.10f}%

Average RAM              : {average_ram_mb:.10f} MB

Peak RAM                 : {peak_ram_mb:.10f} MB


RESOURCES - SYSTEM
----------------------------------------------------------------------

Average CPU              : {average_system_cpu:.10f}%

Peak CPU                 : {peak_system_cpu:.10f}%


OUTPUTS
----------------------------------------------------------------------

Output Video             : {os.path.abspath(OUTPUT_VIDEO)}

JSON Report              : {os.path.abspath(OUTPUT_JSON)}

Log File                 : {os.path.abspath(OUTPUT_LOG)}


======================================================================
END OF REPORT
======================================================================
"""


# ============================================================
# PRINT REPORT
# ============================================================

print()

print(
    report_text
)


# ============================================================
# SAVE LOG
# ============================================================

with open(

    OUTPUT_LOG,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        report_text
    )


# ============================================================
# COMPLETION
# ============================================================

print()

print(
    "Processing completed successfully."
)

print()

print(
    f"Output video: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"JSON report: "
    f"{OUTPUT_JSON}"
)

print(
    f"Log file: "
    f"{OUTPUT_LOG}"
)