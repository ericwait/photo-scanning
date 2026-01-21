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

        # Convert to grayscale and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection for better boundary detection
        edged = cv2.Canny(blurred, 30, 150)
        
        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cropped_images_paths = []
        img_base_name = os.path.basename(image_path).split('.')[0]
        
        # Sort contours by area to optionally skip the "full page" if it persists
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        img_h, img_w = image.shape[:2]
        full_area = img_h * img_w
        
        photo_idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Skip if too small (noise) or too large (likely the whole background/bed)
            if area < 50000: # Adjust based on expected min photo size
                continue
            if area > full_area * 0.95:
                continue
            
            # Get the rotated bounding box
            rect = cv2.minAreaRect(cnt)
            
            # Extract and straighten using perspective transform
            try:
                cropped = self._get_rotated_crop(image, rect)
                
                output_name = f"{img_base_name}_photo_{photo_idx}.jpg"
                output_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(output_path, cropped)
                cropped_images_paths.append(output_path)
                photo_idx += 1
            except Exception as e:
                print(f"Failed to process contour {photo_idx}: {e}")
            
        return cropped_images_paths

    def _get_rotated_crop(self, image: np.ndarray, rect: Tuple) -> np.ndarray:
        """
        Crops a rotated rectangle from an image and straightens it using perspective transform.
        """
        # Get 4 corners of the rotated box
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        # Sort points: top-left, top-right, bottom-right, bottom-left
        # Sum of coords: min = TL, max = BR
        # Diff of coords: min = TR, max = BL
        s = box.sum(axis=1)
        tl = box[np.argmin(s)]
        br = box[np.argmax(s)]
        
        diff = np.diff(box, axis=1)
        tr = box[np.argmin(diff)]
        bl = box[np.argmax(diff)]

        # Calculate side lengths
        width1 = np.linalg.norm(br - bl)
        width2 = np.linalg.norm(tr - tl)
        maxWidth = max(int(width1), int(width2))

        height1 = np.linalg.norm(tr - br)
        height2 = np.linalg.norm(tl - bl)
        maxHeight = max(int(height1), int(height2))

        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Source points in correct order
        src = np.array([tl, tr, br, bl], dtype="float32")

        # Get transform matrix and warp
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

if __name__ == "__main__":
    # Internal test logic
    processor = ProcessorService("output")
    # processor.detect_and_crop("scans/test_page.jpg")
    print("ProcessorService initialized.")
