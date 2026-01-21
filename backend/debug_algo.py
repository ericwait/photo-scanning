import cv2
import numpy as np
import os

def debug_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 4. HSV Saturation (tends to highlight color photos on white/gray backgrounds)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:,:,1]
    _, thresh_s = cv2.threshold(s_channel, 20, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug_hsv_s.jpg", thresh_s)

    # Use thresh_s and find contours with hierarchy
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_s = cv2.dilate(thresh_s, kernel_small, iterations=1)
    
    contours, hierarchy = cv2.findContours(dilated_s, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = image.copy()
    print(f"Total contours found (HSVs): {len(contours)}")
    
    img_h, img_w = image.shape[:2]
    full_area = img_h * img_w
    
    found_count = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            area = cv2.contourArea(cnt)
            # Filter for photo-sized objects (4x5 print at high DPI)
            if area < 200000 or area > full_area * 0.5:
                continue
                
            found_count += 1
            cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 3)
            print(f"Photo Candidate {found_count}: Area={area}, %={area/full_area:.2f}")

    cv2.imwrite("debug_contours_hsv.jpg", debug_img)
    print(f"Kept {found_count} candidates. Result saved to debug_contours_hsv.jpg")

if __name__ == "__main__":
    debug_detection("../test_scanner_bed.png")
