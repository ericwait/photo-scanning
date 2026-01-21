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
        Uses a combination of thresholding and contour analysis.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Pre-processing: Convert to grayscale and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 2. Thresholding: We want to find the "non-white" parts
        # Since photos have varying colors, we use an inverse threshold
        # Assuming background is white (255), we threshold anything lower than 240
        _, thresh = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # 4. Find all contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process
        img_h, img_w = image.shape[:2]
        full_area = img_h * img_w
        cropped_images_paths = []
        img_base_name = os.path.basename(image_path).split('.')[0]
        
        photo_idx = 0
        # Sort by area to process larger items first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Skip noise and the background itself
            if area < (full_area * 0.05): # Min 5% of scanner bed
                continue
            if area > (full_area * 0.95):
                continue
            
            # Approximate the contour to a rectangle
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # If not 4 corners, use the minAreaRect instead
            if len(approx) == 4:
                # Use the approx points for perspective transform
                pts = approx.reshape(4, 2)
            else:
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect)

            try:
                # Extract and straighten
                cropped = self._get_warped_crop(image, pts)
                
                output_name = f"{img_base_name}_photo_{photo_idx}.jpg"
                output_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(output_path, cropped)
                cropped_images_paths.append(output_path)
                photo_idx += 1
            except Exception as e:
                print(f"Failed to process contour {photo_idx}: {e}")
            
        return cropped_images_paths

    def _get_warped_crop(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Extracts a straightened crop using a 4-point perspective transform.
        """
        # Rectify point order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # tl
        rect[2] = pts[np.argmax(s)] # br
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # tr
        rect[3] = pts[np.argmax(diff)] # bl
        
        (tl, tr, br, bl) = rect

        # Compute width and height
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

if __name__ == "__main__":
    # Internal test logic
    processor = ProcessorService("output")
    # processor.detect_and_crop("scans/test_page.jpg")
    print("ProcessorService initialized.")
