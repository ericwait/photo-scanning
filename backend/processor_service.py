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

        # 1. Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]
        
        # 2. Create Masks
        # Mask 1: Saturation (Color). Lowered to 30 to catch duller colors.
        _, s_thresh = cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)
        
        # Mask 2: Value (Brightness). Catch dark items on white background.
        # Scanner background is usually near 255. Photos are darker.
        _, v_thresh = cv2.threshold(v_channel, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Combine: It's a photo if it has color OR is dark
        combined_mask = cv2.bitwise_or(s_thresh, v_thresh)
        
        # 3. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
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
        
        # Process found candidates
        for idx, cnt in enumerate(photo_candidates):
            # Limit to top 3 largest/most relevant if we have too many? 
            # Logic: If we found > 3, we might need better logic, but usually it's noise.
            # For now, let's take up to 3.
            if len(cropped_images_paths) >= 3:
                break

            # Use convex hull to smooth out detections and get a more stable box
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            
            # Slightly shrink the box to ensure we don't pick up white margins
            (center, size, angle) = rect
            size = (size[0] * 0.98, size[1] * 0.98) # Shrink 2%
            rect = (center, size, angle)
            
            try:
                cropped = self._get_warped_crop_from_rect(image, rect)
                output_name = f"{img_base_name}_photo_{len(cropped_images_paths)}.jpg"
                output_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(output_path, cropped)
                cropped_images_paths.append(output_path)
            except Exception as e:
                print(f"Failed to process photo candidate {idx}: {e}")
                
        # Fallback: Ensure we always return at least 3 items (or slots)
        # If we missed one, create a placeholder so the user can manually refine it
        while len(cropped_images_paths) < 3:
            idx = len(cropped_images_paths)
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            placeholder[:] = (30, 30, 30) # Dark gray background
            
            # Add text
            text = "Photo Not Detected"
            cv2.putText(placeholder, text, (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            text2 = "Click 'Refine Crop' to set manually"
            cv2.putText(placeholder, text2, (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            output_name = f"{img_base_name}_photo_{idx}.jpg"
            output_path = os.path.join(self.output_dir, output_name)
            cv2.imwrite(output_path, placeholder)
            cropped_images_paths.append(output_path)
            
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
            
            img_base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_name = f"{img_base_name}_photo_{photo_index}.png"
            output_path = os.path.join(self.output_dir, output_name)
            
            cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
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
        else:
            pts = rect_or_pts

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

    def rotate_photo(self, image_url: str, angle: float) -> str:
        """
        Rotates an existing photo by the specified angle (90 or -90 typically).
        image_url is the relative path (e.g. /output/filename.png).
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
            
        cv2.imwrite(filepath, rotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return filepath

if __name__ == "__main__":
    # Internal test logic
    processor = ProcessorService("output")
    # processor.detect_and_crop("scans/test_page.jpg")
    print("ProcessorService initialized.")
