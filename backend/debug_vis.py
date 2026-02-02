
import cv2
import numpy as np
import os

def debug_steps():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scan_dir = os.path.abspath(os.path.join(base_dir, "..", "scans"))
    input_filename = "scan_20260126_113048.tiff"
    input_path = os.path.join(scan_dir, input_filename)
    output_dir = os.path.join(base_dir, "output", "debug_step_vis")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing {input_path}")
    
    # Load Image (Scale down for debug speed/viewing)
    image = cv2.imread(input_path)
    if image is None: return

    # Scale for visualization
    vis_scale = 0.2
    def save_vis(name, img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * vis_scale), int(h * vis_scale)))
        cv2.imwrite(os.path.join(output_dir, name), small)
        print(f"Saved {name}")

    # Params
    sensitivity = 250
    ignore_black_bg = True
    bg_cutoff = 40 # The value we are testing

    # 1. HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    save_vis("01_s_channel.jpg", s)
    save_vis("02_v_channel.jpg", v)

    # 2. Masks
    _, s_thresh = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
    save_vis("03_s_thresh.jpg", s_thresh)

    # V Thresh (Inv)
    _, v_thresh = cv2.threshold(v, sensitivity, 255, cv2.THRESH_BINARY_INV)
    save_vis("04_v_thresh_inv.jpg", v_thresh)

    save_vis("04_v_thresh_inv.jpg", v_thresh)

    # Combine
    combined_mask = cv2.bitwise_or(s_thresh, v_thresh)
    save_vis("07_combined_pre_filter.jpg", combined_mask)

    # Black BG Filter (Apply to EVERYTHING)
    if ignore_black_bg:
        _, bg_thresh = cv2.threshold(v, bg_cutoff, 255, cv2.THRESH_BINARY)
        save_vis("05_bg_thresh_gt40.jpg", bg_thresh)
        
        combined_mask = cv2.bitwise_and(combined_mask, bg_thresh)
        save_vis("07_combined_filtered.jpg", combined_mask)

    # 3. Morph
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    save_vis("08_morphed.jpg", morphed)
    
    eroded = cv2.erode(morphed, kernel, iterations=2)
    save_vis("09_eroded.jpg", eroded)

    # 4. Contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original
    debug_cnt = image.copy()
    cv2.drawContours(debug_cnt, contours, -1, (0, 0, 255), 5)
    
    img_h, img_w = image.shape[:2]
    full_area = img_h * img_w
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        ratio = area / full_area
        cx, cy = 0, 0
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
        color = (0, 0, 255)
        if 0.05 < ratio < 0.90:
            color = (0, 255, 0)
            print(f"Contour {i}: Valid (Ratio {ratio:.3f})")
        else:
            print(f"Contour {i}: Invalid (Ratio {ratio:.3f})")
            
        cv2.drawContours(debug_cnt, [cnt], -1, color, 5)
        cv2.putText(debug_cnt, f"{i}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)

    save_vis("10_contours.jpg", debug_cnt)

if __name__ == "__main__":
    debug_steps()
