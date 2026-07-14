# Person Detection & Tracking with YOLOv8 + ByteTrack

## Overview
This project detects and tracks people in a video using **YOLOv8** for object detection and **ByteTrack** (via the `supervision` library) for multi-object tracking. Each tracked person is assigned a unique ID and color, and their movement path is drawn frame-by-frame across the video.

## Features
- Detects people (COCO class `0`) in each frame using YOLOv8n (`yolov8n.pt`)
- Assigns a persistent tracker ID to each detected person using ByteTrack
- Draws a bounding box and ID label for each tracked person
- Assigns a unique random color per tracker ID
- Draws a movement trail (path) showing where each person has walked over time
- Outputs an annotated video with all tracking overlays
- Re-encodes the output video with `ffmpeg`/H.264 for browser-friendly playback
- Displays the final video inline in the notebook

## Requirements
Install the following packages (as done in the notebook):
```bash
pip install ultralytics
pip install opencv-python numpy
pip install supervision
```
You will also need `ffmpeg` installed on your system/environment for the video re-encoding step.

## Project Structure / Workflow
1. **Setup** — Install dependencies and import libraries (`cv2`, `numpy`, `ultralytics.YOLO`, `supervision`, `uuid`, `collections.defaultdict`, `random`).
2. **Model loading** — Load the pretrained YOLOv8 nano model: `YOLO('yolov8n.pt')`.
3. **Tracking utilities** — A `trackers` dictionary stores the path history per tracker ID, and `generate_random_color()` assigns each new ID a distinct color.
4. **Core function — `process_video(input_path, output_path)`**:
   - Opens the input video with OpenCV
   - Runs YOLOv8 inference per frame
   - Filters detections to the `person` class only
   - Updates a `ByteTrack` tracker with the person detections
   - For each tracked person:
     - Assigns/retrieves a color for its ID
     - Records the bottom-center point of its bounding box (used as a "foot position")
     - Draws the bounding box, ID label, and a connected-line movement trail
   - Writes each annotated frame to the output video
5. **Run the pipeline** — Call `process_video("Track2.mp4", "final2_Persons_Tracking_video.mp4")`.
6. **Post-processing** — Re-encode the output with `ffmpeg` (H.264 / `libx264`) so it plays correctly in-browser/notebook.
7. **Display** — Read the final video, base64-encode it, and render it inline via `IPython.display.HTML`.

## Inputs & Outputs
| Item | Description |
|---|---|
| `Track2.mp4` | Input video containing people to track |
| `final2_Persons_Tracking_video.mp4` | Raw annotated output (mp4v codec) |
| `new_final2_Persons_Tracking_video.mp4` | Re-encoded, browser-playable output (H.264) |

## How to Run
1. Place your input video (e.g. `Track2.mp4`) in the working directory (`/content` if using Colab).
2. Run all cells in order.
3. The annotated video will be saved, re-encoded, and displayed inline at the end of the notebook.

## Notes / Possible Improvements
- Currently only tracks the `person` class (COCO class ID `0`); could be extended to other classes.
- Uses `yolov8n.pt` (nano) for speed — a larger model (`yolov8s/m/l/x`) could improve detection accuracy at the cost of speed.
- Movement trails currently persist for the whole video; could add trail length limits or fading effects.
- No frame-skipping/batching optimizations — processing is done frame-by-frame, which may be slow for long videos.
