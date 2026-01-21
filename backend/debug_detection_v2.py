
import cv2
import numpy as np
import os
from processor_service import ProcessorService

def debug_pipeline():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "test_scanner_bed.png"))
    output_dir = os.path.join(base_dir, "output")
    
    print(f"DEBUG: Input Image: {input_path}")
    print(f"DEBUG: Output Dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Image
    image = cv2.imread(input_path)
    if image is None:
        print("ERROR: Could not load image!")
        return

    # 1. HSV Conversion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    cv2.imwrite(os.path.join(output_dir, "debug_01_saturation.png"), s)
    cv2.imwrite(os.path.join(output_dir, "debug_02_value.png"), v)

    # 2. Thresholding (Exactly as in processor_service)
    # Saturation Threshold > 30
    _, s_thresh = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "debug_03_s_thresh.png"), s_thresh)

    # Value Threshold < 210 (Dark objects) - Updated to match processor_service
    _, v_thresh = cv2.threshold(v, 210, 255, cv2.THRESH_BINARY_INV) # Inverted so dark is white
    cv2.imwrite(os.path.join(output_dir, "debug_04_v_thresh.png"), v_thresh)

    # Combined Mask
    combined_mask = cv2.bitwise_or(s_thresh, v_thresh)
    cv2.imwrite(os.path.join(output_dir, "debug_05_combined_mask.png"), combined_mask)

    # 3. Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(morphed, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "debug_06_dilated_morph.png"), dilated)

    # 4. Find Contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"DEBUG: Found {len(contours)} contours")

    # Draw ALL contours
    debug_all_contours = image.copy()
    cv2.drawContours(debug_all_contours, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "debug_07_all_contours.png"), debug_all_contours)

    # Filter Logic Visualization
    total_area = image.shape[0] * image.shape[1]
    print(f"DEBUG: Total Image Area: {total_area} px")

    debug_filtered = image.copy()
    valid_count = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        ratio = area / total_area
        
        # Current Filter Logic: 0.05 < ratio < 0.90
        is_valid = 0.05 < ratio < 0.90
        
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        status = "KEEP" if is_valid else "DROP"
        
        print(f"Contour {i}: Area={area:.0f} ({ratio*100:.2f}%) -> {status}")
        
        if is_valid:
            valid_count += 1
            # Draw valid in Green, thick
            cv2.drawContours(debug_filtered, [c], -1, (0, 255, 0), 3)
            # Label it
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(debug_filtered, f"#{i} OK", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # Draw invalid in Red, thin
             cv2.drawContours(debug_filtered, [c], -1, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(output_dir, "debug_08_filtered_contours.png"), debug_filtered)
    print(f"DEBUG: Total Valid Photos: {valid_count}")

    # 5. Run Actual Service Method
    print("\n--- Running ProcessorService.detect_and_crop ---")
    service = ProcessorService(output_dir=output_dir)
    # Mock dirs used by service if needed (but it takes image directly)
    # call detect_and_crop
    # We pass the image path directly
    try:
        results = service.detect_and_crop(input_path)
        print(f"Service returned {len(results)} paths:")
        for r in results:
            print(f" - {r}")
    except Exception as e:
        print(f"Service crashed: {e}")

if __name__ == "__main__":
    debug_pipeline()
