
import cv2
import numpy as np
import os
from processor_service import ProcessorService

def test_service_black_bg_v2():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scan_dir = os.path.abspath(os.path.join(base_dir, "..", "scans"))
    output_dir = os.path.join(base_dir, "output", "debug_service_test_v2")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processor = ProcessorService(output_dir)

    files_to_test = [
        "scan_20260126_120632.tiff"
    ]

    for f in files_to_test:
        path = os.path.join(scan_dir, f)
        if not os.path.exists(path):
            print(f"Skipping {f} (Not Found)")
            continue
            
        print(f"\nProcessing {f}...")
        try:
            # Emulate the recommendation: Ignore Black BG = True, Sensitivity = 250
            # Grid = 2 rows x 1 col = 2 total.
            
            results = processor.detect_and_crop(
                path, 
                output_subfolder="debug_test", 
                sensitivity=250, 
                crop_margin=10, 
                ignore_black_background=True,
                grid_rows=2,
                grid_cols=1
            )
            
            print(f"  Found {len(results)} photos.")
            for i, r in enumerate(results):
                print(f"    Photo {i}: {r['path']}")
                if not r['points']:
                    print("    (Placeholder - Detection Failed)")
                    
            # Also save debug mask logic locally to see WHY it failed if it did
            # Re-implement partial logic here for visualization
            img = cv2.imread(path)
            if img is not None:
                # Resize for speed
                h, w = img.shape[:2]
                scale = 0.2
                small = cv2.resize(img, (int(w*scale), int(h*scale)))
                
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                s = hsv[:,:,1]
                v = hsv[:,:,2]
                
                _, s_thresh = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)
                _, v_thresh = cv2.threshold(v, 250, 255, cv2.THRESH_BINARY_INV)
                _, bg_thresh = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY)
                
                combined = cv2.bitwise_or(s_thresh, v_thresh)
                filtered = cv2.bitwise_and(combined, bg_thresh)
                
                cv2.imwrite(os.path.join(output_dir, "debug_manual_s.jpg"), s_thresh)
                cv2.imwrite(os.path.join(output_dir, "debug_manual_v.jpg"), v_thresh)
                cv2.imwrite(os.path.join(output_dir, "debug_manual_bg.jpg"), bg_thresh)
                cv2.imwrite(os.path.join(output_dir, "debug_manual_combined.jpg"), combined)
                cv2.imwrite(os.path.join(output_dir, "debug_manual_filtered.jpg"), filtered)

                # Morph
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
                eroded = cv2.erode(morphed, kernel, iterations=2)
                cv2.imwrite(os.path.join(output_dir, "debug_manual_eroded.jpg"), eroded)

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_service_black_bg_v2()
