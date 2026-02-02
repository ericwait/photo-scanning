
import cv2
import numpy as np
import os
from smart_detector import SmartDetector

def debug_pipeline():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "test_scanner_bed.png"))
    output_dir = os.path.join(base_dir, "output")
    
    print(f"DEBUG: Input Image: {input_path}")
    print(f"DEBUG: Output Dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detector = SmartDetector()
    
    # Assume 400 DPI for testing (based on image size analysis)
    # Adjust as needed if the fit is terrible
    dpi = 400 
    print(f"Testing with DPI={dpi}")
    
    # 1. Visualize Texture Map
    image = cv2.imread(input_path)
    if image is None:
        print("Error loading image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_map = detector._calculate_texture_map(gray)
    cv2.imwrite(os.path.join(output_dir, "smart_01_texture.png"), texture_map)
    print("Saved texture map")
    
    print(f"Texture Map Stats: Mean={np.mean(texture_map):.2f}, Max={np.max(texture_map)}")
    
    # 2. Candidates
    candidates = detector._find_candidates(texture_map, dpi)
    # Sort by area
    candidates.sort(key=lambda x: x['rect'][1][0] * x['rect'][1][1], reverse=True)
    
    debug_cand = image.copy()
    for i, c in enumerate(candidates[:5]):
        rect = c['rect']
        box = np.int64(cv2.boxPoints(rect))
        (w, h) = rect[1]
        print(f"Candidate {i}: Center={rect[0]}, Size={w:.0f}x{h:.0f}, Angle={rect[2]:.1f}")
        cv2.drawContours(debug_cand, [box], 0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "smart_02_candidates.png"), debug_cand)
    print(f"Found {len(candidates)} candidates")
    
    # 3. Final Detect
    # debug_detection_v2 loads image from file, so we pass it
    results = detector.detect(image, dpi)
    print(f"Final Detection found {len(results)} photos")
    
    debug_final = image.copy()
    for i, res in enumerate(results):
        box = np.array(res.box)
        cv2.drawContours(debug_final, [box], 0, (0, 255, 0), 3)
        
        # Draw label
        cx, cy = np.mean(box, axis=0).astype(int)
        label = f"{res.size_label} ({res.confidence:.2f})"
        cv2.putText(debug_final, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(f" - #{i} {label} {box.tolist()}")

    cv2.imwrite(os.path.join(output_dir, "smart_03_final.png"), debug_final)
    print("Done")

if __name__ == "__main__":
    debug_pipeline()
