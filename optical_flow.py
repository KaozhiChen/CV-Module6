"""
CV Module 6 – Dense optical flow visualization script.

This script computes dense optical flow for an input video using the
Farneback algorithm and visualizes the motion field as a color-coded
video in HSV space (encoded as BGR for saving).

Usage:
    python optical_flow.py

Outputs:
- A video file (default: `images_and_videos/optical_flow_video1.mp4`)
  where hue encodes motion direction and value/brightness encodes motion
  magnitude.

Notes:
- The input video path is currently hard-coded to
  `images_and_videos/video1.mp4`.
- Adjust the Farneback parameters if you want to experiment with the
  smoothness and sensitivity of the optical flow field.
"""

import cv2
import numpy as np

# Open input video (stored in the images_and_videos subfolder)
cap = cv2.VideoCapture("images_and_videos/video1.mp4")

# Query basic video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer for optical-flow visualization
out = cv2.VideoWriter(
    "images_and_videos/optical_flow_video1.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height),
)

# Read first frame and convert to grayscale as optical-flow reference
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# HSV image used for color visualization of optical flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow between consecutive grayscale frames
    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        next,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    # Convert flow vectors (u,v) to polar coordinates (magnitude, angle)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Encode direction as hue and magnitude as value (brightness)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Write color-encoded optical flow frame
    out.write(rgb)

    # Current frame becomes previous frame for the next iteration
    prvs = next

cap.release()
out.release()

print("Optical flow video saved.")