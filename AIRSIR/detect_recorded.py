import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from glob import glob
print("YOLO Detection on AirSim Recordings")
airsim_base = Path.home() / "Documents" / "AirSim"
if not airsim_base.exists():
    airsim_base = Path.home() / "OneDrive" / "Documents" / "AirSim"
print(f"\nLooking for recordings in: {airsim_base}")
recording_folders = sorted(
    [d for d in airsim_base.iterdir() if d.is_dir() and d.name[0].isdigit()],
    key=lambda x: x.name,
    reverse=True
)
if not recording_folders:
    print(" No recording folders found!")
    print("   Make sure to press 'R' in AirSim to record first.")
    exit()

print(f"\nFound {len(recording_folders)} recording(s):")
for i, folder in enumerate(recording_folders[:5]):
    images_dir = folder / "images"
    num_images = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    print(f"   {i+1}. {folder.name} ({num_images} images)")

print(f"\nUsing most recent: {recording_folders[0].name}")
images_folder = recording_folders[0] / "images"
image_files = sorted(glob(str(images_folder / "*.png")))
print(f"   Found {len(image_files)} images")

if not image_files:
    print("No images found in this recording!")
    exit()
output_dir = Path("airsim_detections")
output_dir.mkdir(exist_ok=True)
(output_dir / "annotated").mkdir(exist_ok=True)

print("\nLoading YOLOv8x model (Best Accuracy)...")
model = YOLO('yolov8x.pt')
print("YOLOv8x model loaded!")
all_detections = []
object_counts = {}
frame_num = 0
first_img = cv2.imread(image_files[0])
h, w = first_img.shape[:2]
video_path = output_dir / f"detection_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
video_writer = cv2.VideoWriter(
    str(video_path),
    cv2.VideoWriter_fourcc(*'mp4v'),
    10, (w, h)
)
print(f"\nProcessing {len(image_files)} images...")
print("-"*60)

for i, img_path in enumerate(image_files):
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()
    detected = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        detected.append(name)
        object_counts[name] = object_counts.get(name, 0) + 1
        all_detections.append({
            'frame': frame_num,
            'image': os.path.basename(img_path),
            'class': name,
            'confidence': round(conf, 3)
        })
    cv2.rectangle(annotated, (5, 5), (300, 60), (0, 0, 0), -1)
    cv2.putText(annotated, f"Frame {frame_num+1}/{len(image_files)}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detected: {len(detected)} objects", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    video_writer.write(annotated)
    if len(detected) >= 3 or frame_num % 20 == 0:
        cv2.imwrite(str(output_dir / "annotated" / f"frame_{frame_num:05d}.jpg"), annotated)
    if (i + 1) % 50 == 0 or i == len(image_files) - 1:
        print(f"   Processed {i+1}/{len(image_files)} | Detections so far: {len(all_detections)}")
    frame_num += 1

video_writer.release()
report = {
    'source_folder': str(recording_folders[0]),
    'total_frames': len(image_files),
    'total_detections': len(all_detections),
    'unique_objects': list(object_counts.keys()),
    'object_counts': object_counts,
    'sample_detections': all_detections[:100]
}

report_path = output_dir / f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("DETECTION RESULTS")
print(f"   Frames processed: {len(image_files)}")
print(f"   Total detections: {len(all_detections)}")
print(f"   Unique object types: {len(object_counts)}")

if object_counts:
    print("\nObjects Found:")
    for obj, count in sorted(object_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * min(count // 5, 20)
        print(f"      {obj:20s}: {count:4d} {bar}")

print(" OUTPUT FILES")
print(f"    Video: {video_path}")
print(f"    Report: {report_path}")
print(f"     Images: {output_dir / 'annotated'}/")

