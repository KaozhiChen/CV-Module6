"""
CV Module 6 – Frame extraction script.

This script opens an input video file, seeks to a specific frame index,
and saves two consecutive frames as PNG images. These frames are later
used for manual motion tracking and theoretical validation.

Usage:
    python frame.py

Notes:
- The input video file path and frame index are hard-coded.
- Video and image files are expected under the `images_and_videos`
  subfolder in this project.
- Modify the `cv2.VideoCapture` path and `cv2.CAP_PROP_POS_FRAMES` value
  if you want to use a different video or frame index.
"""

import cv2

cap = cv2.VideoCapture("images_and_videos/video1.mp4")

# Seek to the selected starting frame index (here: 125)
cap.set(cv2.CAP_PROP_POS_FRAMES, 125)

ret, frame125 = cap.read()
ret, frame126 = cap.read()

cv2.imwrite("images_and_videos/video1_frame125.png", frame125)
cv2.imwrite("images_and_videos/video1_frame126.png", frame126)

cap.release()

print("Frames saved.")