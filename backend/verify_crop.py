
import cv2
import numpy as np
import os
from processor_service import ProcessorService

def verify_crop():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(base_dir, "..", "test_scanner_bed.png"))
    output_dir = os.path.join(base_dir, "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Testing detection/crop on: {input_path}")
    
    service = ProcessorService(output_dir=output_dir)
    try:
        results = service.detect_and_crop(input_path)
        print(f"Service returned {len(results)} paths:")
        for r in results:
            print(f" - {r}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_crop()
