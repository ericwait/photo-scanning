
import cv2
import numpy as np
import os

def check_stats():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "test_scanner_bed.png"))
    
    image = cv2.imread(input_path)
    if image is None:
        print("Error loading image")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    print(f"V Channel - Min: {np.min(v)}, Max: {np.max(v)}, Mean: {np.mean(v):.2f}")
    print(f"S Channel - Min: {np.min(s)}, Max: {np.max(s)}, Mean: {np.mean(s):.2f}")
    
    # Histogram of V to see the background peak
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    # Print the top 5 most common V values (likely background)
    indices = np.argsort(hist_v.flatten())[-5:]
    print("Most common V values:", indices)
    
if __name__ == "__main__":
    check_stats()
