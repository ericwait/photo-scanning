import cv2
import numpy as np
import os
from typing import List, Tuple

class ProcessorService:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def detect_and_crop(self, image_path: str) -> List[str]:
        """
        Detects multiple photos in a single scanned page and crops them.
        Returns a list of paths to the cropped photos.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle potential brightness variations
        # Assuming photos are darker/different from the high-key scanner background/lid
        # Epson lid is usually white or black. Assuming white for now.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary thresholding - invert if the background is white
        # We try to find the "objects" which are the photos
        _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to close gaps and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cropped_images_paths = []
        img_base_name = os.path.basename(image_path).split('.')[0]
        
        photo_idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out small noise - adjust threshold based on DPI
            # 4x6 photo at 300DPI is ~1200x1800 = 2.16M pixels
            # Let's set a conservative minimum area of 100k pixels
            if area < 100000:
                continue
            
            # Get the rotated bounding box
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Straighten and crop
            cropped = self._get_rotated_crop(image, rect)
            
            output_name = f"{img_base_name}_photo_{photo_idx}.jpg"
            output_path = os.path.join(self.output_dir, output_name)
            cv2.imwrite(output_path, cropped)
            cropped_images_paths.append(output_path)
            photo_idx += 1
            
        return cropped_images_paths

    def _get_rotated_crop(self, image: np.ndarray, rect: Tuple) -> np.ndarray:
        """
        Crops a rotated rectangle from an image and straightens it.
        """
        center, size, angle = rect
        
        # Correct for OpenCV angle behavior
        # In newer OpenCV, angle is in [0, 90]. 
        # In older version it was [-90, 0].
        # We want to ensure the width is greater than height or vice versa depending on orientation
        width, height = size
        if width < height:
            angle += 90
            width, height = height, width
            
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation on the whole image (with padding to avoid clipping)
        # Actually, for just cropping, we can be more efficient, but let's keep it simple
        h, w = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h))
        
        # Crop the straightened area
        x_start = int(center[0] - width / 2)
        y_start = int(center[1] - height / 2)
        
        # Ensure within bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        
        cropped = rotated[y_start:y_start+int(height), x_start:x_start+int(width)]
        return cropped

if __name__ == "__main__":
    # Internal test logic
    processor = ProcessorService("output")
    # processor.detect_and_crop("scans/test_page.jpg")
    print("ProcessorService initialized.")
