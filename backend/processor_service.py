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
        Uses HSV saturation thresholding to isolate color prints from white backgrounds.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Convert to HSV and extract Saturation channel
        # Color photos have high saturation compared to white/gray backgrounds
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:,:,1]
        
        # 2. Threshold Saturation
        # A higher threshold ensures we pick up strong colors only, ignoring "off-white" paper
        _, thresh = cv2.threshold(s_channel, 45, 255, cv2.THRESH_BINARY)
        
        # 3. Morphological cleanup: Close small gaps but don't expand too much
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(morphed, kernel, iterations=1)

        # 4. Find all contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filter for photo-sized objects
        img_h, img_w = image.shape[:2]
        full_area = img_h * img_w
        
        # We expect typically 3 photos per page, but let's be flexible
        photo_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: must be at least 5% but less than 50% of the scanner bed
            if (full_area * 0.05) < area < (full_area * 0.5):
                photo_candidates.append(cnt)
        
        # Sort top-to-bottom based on the center of the bounding box
        def get_center_y(cnt):
            M = cv2.moments(cnt)
            if M["m00"] == 0: return 0
            return int(M["m01"] / M["m00"])
            
        photo_candidates.sort(key=get_center_y)
        
        cropped_images_paths = []
        img_base_name = os.path.basename(image_path).split('.')[0]
        
        for idx, cnt in enumerate(photo_candidates):
            # Use convex hull to smooth out detections and get a more stable box
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            
            # Slightly shrink the box to ensure we don't pick up white margins
            # Shrink by 1% of dimensions or 5 pixels
            (center, size, angle) = rect
            size = (size[0] * 0.98, size[1] * 0.98) # Shrink 2%
            rect = (center, size, angle)
            
            try:
                # Extract and straighten
                cropped = self._get_warped_crop_from_rect(image, rect)
                
                output_name = f"{img_base_name}_photo_{idx}.jpg"
                output_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(output_path, cropped)
                cropped_images_paths.append(output_path)
            except Exception as e:
                print(f"Failed to process photo {idx}: {e}")
            
        return cropped_images_paths

    def manual_crop(self, image_path: str, points: List[List[int]], photo_index: int) -> str:
        """
        Manually crops a photo from the scan using 4 user-provided points.
        points: List of [x, y] coordinates order: TL, TR, BR, BL
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        pts = np.array(points, dtype="float32")
        
        try:
            cropped = self._get_warped_crop_from_rect(image, pts)
            
            img_base_name = os.path.basename(image_path).split('.')[0]
            output_name = f"{img_base_name}_photo_{photo_index}.jpg"
            output_path = os.path.join(self.output_dir, output_name)
            
            cv2.imwrite(output_path, cropped)
            return output_path
        except Exception as e:
            print(f"Manual crop failed: {e}")
            raise e

    def _get_warped_crop_from_rect(self, image: np.ndarray, rect_or_pts) -> np.ndarray:
        """
        Extracts a straightened crop using perspective warping.
        Accepts either a rotated rect tuple or a numpy array of 4 points.
        """
        if isinstance(rect_or_pts, tuple):
            box = cv2.boxPoints(rect_or_pts)
            pts = np.array(box, dtype="float32")
    def rotate_photo(self, image_url: str, angle: float) -> str:
        """
        Rotates an existing photo by the specified angle (90 or -90 typically).
        image_url is the relative path (e.g. /output/filename.jpg).
        """
        # Convert relative URL to absolute file path
        filename = os.path.basename(image_url)
        filepath = os.path.join(self.output_dir, filename)
        
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Could not read image at {filepath}")
            
        if angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Arbitrary rotation not implemented for simple buttons yet, use existing
            raise ValueError("Only 90 and -90 supported efficiently")
            
        cv2.imwrite(filepath, rotated)
        return filepath

        # Order points: tl, tr, br, bl
        # Sum: min = tl, max = br
        # Diff: min = tr, max = bl
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

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

        src = np.array([tl, tr, br, bl], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

if __name__ == "__main__":
    # Internal test logic
    processor = ProcessorService("output")
    # processor.detect_and_crop("scans/test_page.jpg")
    print("ProcessorService initialized.")
