"""
SfM Boundary Rectification (Planar Homography Demo)

This script reconstructs a rectified (top-down) view of a planar object by:
1) Manually clicking 4 corresponding corner points on each image.
2) Computing a homography that maps each view (2.jpg/3.jpg/4.jpg) to the
   reference image (1.jpg).
3) Warping each view into the reference image coordinate system.
4) Saving the rectified results.

Input images (required):
  images_and_videos/1.jpg
  images_and_videos/2.jpg
  images_and_videos/3.jpg
  images_and_videos/4.jpg

Click order (must be the SAME in every image):
  1. top-left
  2. top-right
  3. bottom-right
  4. bottom-left

Run:
  python sfm_auto.py
"""

import cv2
import numpy as np
import os

# Temporary storage for clicked points (pixel coordinates).
current_points = []

def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        # Mark the clicked location for visual confirmation.
        cv2.circle(param['img'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(param['window_name'], param['img'])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images_and_videos')

def get_points(img_name):
    global current_points
    # Clear previously clicked points before loading the next image.
    current_points = []
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not find image: {img_path}")
        return None
        
    img_draw = img.copy()
    window_name = f'Click 4 corners of {img_name} (Press ANY KEY when done)'
    
    # Allow window resizing to avoid cropping when images are large.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_draw)
    
    print(f"\nProcessing: {img_name} ...")
    print("Please click 4 corners in order:")
    print("  1) top-left -> 2) top-right -> 3) bottom-right -> 4) bottom-left")
    print("Tip: After clicking all 4 points, press ANY KEY (e.g., Space/Enter) to continue.")
    
    cv2.setMouseCallback(window_name, click_event, param={'img': img_draw, 'window_name': window_name})
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(current_points, dtype="float32")

# ==========================================
# Main program
# ==========================================

print("SfM boundary rectification started.")

# 1) Collect 4 corner points from the reference (top-down) image.
pts_ref = get_points('1.jpg')

if pts_ref is not None and len(pts_ref) == 4:
    img_ref_path = os.path.join(IMAGES_DIR, '1.jpg')
    img_ref = cv2.imread(img_ref_path)
    if img_ref is None:
        raise FileNotFoundError(f"Could not read reference image: {img_ref_path}")
    height, width = img_ref.shape[:2]
    
    # 2) Automatically process 2.jpg, 3.jpg, 4.jpg.
    for i in range(2, 5):
        img_name = f'{i}.jpg'
        pts_view = get_points(img_name)
        
        if pts_view is not None and len(pts_view) == 4:
            # Compute homography H (view -> reference).
            H, status = cv2.findHomography(pts_view, pts_ref)
            if H is None:
                print(f"Warning: Homography could not be computed for {img_name}. Skipping.")
                continue
            
            print(f"\nHomography matrix H for mapping {img_name} -> 1.jpg:")
            print(H)
            
            # Warp the view image into the reference coordinate system.
            img_view_path = os.path.join(IMAGES_DIR, img_name)
            img_view = cv2.imread(img_view_path)
            if img_view is None:
                print(f"Warning: Could not read {img_view_path}. Skipping.")
                continue
            reconstructed = cv2.warpPerspective(img_view, H, (width, height))
            
            # Save the rectified result.
            out_name = os.path.join(IMAGES_DIR, f'reconstructed_from_view{i}.jpg')
            cv2.imwrite(out_name, reconstructed)
            print(f"Saved rectified image to: {out_name}\n")
            print("-" * 50)
        else:
            print(f"Warning: {img_name} does not have exactly 4 clicked points. Skipping this image.")
else:
    print("Warning: 1.jpg did not result in exactly 4 clicked points. Exiting.")

print("All done. Check the output images in the `images_and_videos/` folder.")