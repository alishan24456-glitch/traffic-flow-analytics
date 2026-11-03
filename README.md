# Traffic Flow Analytics

Multi-class vehicle counting with speed estimation, built with OpenCV
(and YOLOv8 for real footage): vehicles crossing a virtual count line on
a road are tallied per class — car, truck, bus, motorcycle — and per
direction (eastbound / westbound), and each vehicle's speed is estimated
from how fast its tracked centroid moves across the frame.

## Run it (CMD)

```
pip install -r requirements.txt
python traffic_flow_analytics.py --simulate
python traffic_flow_analytics.py --simulate --output demo.mp4 --duration 28
python traffic_flow_analytics.py --source pick
python traffic_flow_analytics.py
python traffic_flow_analytics.py --source 0
python traffic_flow_analytics.py --source traffic.mp4 --output counted.mp4
```

Running with no arguments (or `--source pick`) opens a file-picker
dialog to choose a video. `yolov8n.pt` downloads automatically on first
real-mode run. Use `--line-x 0.4` to move the count line, and
`--px-per-metre` to calibrate speed to your camera (see below).

## How it works

One pipeline serves both modes: detect → class-gated centroid track →
line-crossing count per class and direction → speed from tracked
displacement → overlay (boxes, IDs, live km/h, HUD totals).

- **Real mode** detects with YOLOv8 (COCO classes car, truck, bus,
  motorcycle), tracked by ByteTrack by default (`--tracker centroid`
  switches to the built-in tracker).
- **Simulation mode** renders a top-down four-lane road and reads the
  vehicles back out of the rendered pixels with an HSV colour/shape
  detector. It never reads the scene's vehicle list: counts come only
  from detected pixels, exactly as real-mode counts come only from
  model output. Simulation needs only opencv-python + numpy.

## Demo

- Simulation demo (28 s, rendered four-lane road, judged by the pixel
  detector): `files/project-demos/traffic-flow-analytics-demo.mp4`
- LinkedIn-ready copy (H.264, same demo):
  `files/project-demos/linkedin-h264/traffic-flow-analytics-demo.mp4`
- Verification: `py_compile` clean; demo read back and frame-inspected;
  end-of-run counters below are the run's own output, not estimates.
  - Simulation demo (30 Oct 2026): 840 frames, 39 vehicles counted —
    18 eastbound, 21 westbound, peak 10 simultaneous; car 27 (mean
    59.9 km/h), truck 6 (60.7 km/h), bus 2 (56.5 km/h), motorcycle 4
    (56.0 km/h).

## Honest notes

- Speed is an **estimate**: it converts tracked pixels per second with
  `--px-per-metre` (default 0.05 m/px, i.e. a ~100 px car is ~5 m long).
  It is only as accurate as that calibration for your camera's angle
  and height — measure a known-length vehicle in your own footage and
  set the flag to match. Do not quote these speeds as radar-grade.
- Simulation classes are told apart by body colour + shape, because a
  rendered scene has no photographic texture for a model to read; the
  real-mode classes come from YOLOv8, not from colour.
- Counts need a clean crossing: a vehicle that stops on the line, or
  two vehicles merged into one blob in heavy congestion, can be missed
  or counted once instead of twice. The detector also rejects blobs
  wider than 260 px, so a long traffic jam reads as one blob, not many
  vehicles.
- The simulation demo shows the pipeline working; on real highway
  footage, YOLOv8 + ByteTrack is the mode that matters, and its
  accuracy depends on camera angle, occlusion and the model used.
