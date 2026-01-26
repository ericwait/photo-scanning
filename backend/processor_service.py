import cv2
import numpy as np
import os
from typing import List, Tuple

class ProcessorService:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)



    def detect_and_crop(self, image_path: str, output_subfolder: str = None, sensitivity: int = 210, crop_margin: int = 10, contrast: float = 1.0, auto_contrast: bool = False, auto_wb: bool = False, grid_rows: int = 3, grid_cols: int = 1, ignore_black_background: bool = False) -> List[dict]:
        """
        Detects multiple photos in a single scanned page and crops them.
        Returns: List of dicts { "path": str, "points": List[List[int]] }
        """
        # Load with ANYDEPTH to support 16-bit (uint16)
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Check depth
        is_16bit = image.dtype == np.uint16
        
        # Create an 8-bit version for detection logic (HSV, Thresholds work best in 8-bit standard range)
        if is_16bit:
            # Scale down to 8-bit
            scan_8bit = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
            # Normalize to full 0-255 range to align with expected thresholds
            scan_8bit = cv2.normalize(scan_8bit, None, 0, 255, cv2.NORM_MINMAX)
        else:
            scan_8bit = image.copy()

        # Determine output directory
        current_output_dir = self.output_dir
        if output_subfolder:
            if os.path.isabs(output_subfolder):
                current_output_dir = output_subfolder
            else:
                current_output_dir = os.path.join(self.output_dir, output_subfolder)
            
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

        # 1. Convert to HSV (Use 8-bit version)
        hsv = cv2.cvtColor(scan_8bit, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]
        
        # 2. Create Masks
        # Mask 1: Saturation (Color). Lowered to 30 to catch duller colors.
        _, s_thresh = cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)
        
        # Mask 2: Value (Brightness). Catch dark items on white background.
        v_thresh_val = sensitivity 
        _, v_thresh = cv2.threshold(v_channel, v_thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Combine: It's a photo if it has color OR is dark
        combined_mask = cv2.bitwise_or(s_thresh, v_thresh)
        
        # Optional: Black Background filtering (Bandpass)
        # Apply to the COMBINED mask. 
        # This ensures that even if saturation triggered detection (noise), 
        # simple darkness (V < 40) will veto it.
        if ignore_black_background:
            _, bg_thresh = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_and(combined_mask, bg_thresh)
        
        # 3. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        morphed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Erode to peel off jagged edges or thin connections (like hole punch tabs)
        # 2 iterations with 7x7 kernel = ~7-14 pixels shaving.
        eroded = cv2.erode(morphed, kernel, iterations=2)
        
        # REMOVED dilation to prevent expanding the border
        # dilated = cv2.dilate(morphed, kernel, iterations=1)

        # 4. Find all contours - Use RETR_LIST to ensure we get internal contours if the background is detected
        # Use 'eroded' to get tighter bounds and ignore artifacts
        contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filter for photo-sized objects
        img_h, img_w = scan_8bit.shape[:2]
        full_area = img_h * img_w
        
        photo_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter: must be at least 5% but less than 90% of the scanner bed
            # We increased the upper bound slightly just in case, but usually photos are < 50%
            if (full_area * 0.05) < area < (full_area * 0.90):
                photo_candidates.append(cnt)
        
        # Sort top-to-bottom, left-to-right (Row-major)
        # This handles grids (e.g. 2x2) correctly
        def get_center(cnt):
            M = cv2.moments(cnt)
            if M["m00"] == 0: return (0, 0)
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # x, y

        centers = [get_center(cnt) for cnt in photo_candidates]
        candidates_with_centers = list(zip(photo_candidates, centers))
        
        # Sort primarily by Y to get rough top-down order
        candidates_with_centers.sort(key=lambda x: x[1][1])
        
        sorted_candidates = []
        current_row = []
        # Threshold to consider items as being on the same "row"
        # 150px is generous but ensures slightly skewed placements are grouped
        ROW_Y_THRESHOLD = 150 
        
        for item in candidates_with_centers:
            if not current_row:
                current_row.append(item)
            else:
                # Compare with the Y of the first item in the current row
                row_y = current_row[0][1][1]
                cy = item[1][1]
                
                if abs(cy - row_y) < ROW_Y_THRESHOLD:
                    current_row.append(item)
                else:
                    # Finish this row: Sort by X (Left->Right)
                    current_row.sort(key=lambda x: x[1][0])
                    sorted_candidates.extend([x[0] for x in current_row])
                    # Start new row
                    current_row = [item]
                    
        # Flush last row
        if current_row:
             current_row.sort(key=lambda x: x[1][0])
             sorted_candidates.extend([x[0] for x in current_row])
             
        photo_candidates = sorted_candidates
        
        results = []
        img_base_name = os.path.basename(image_path).split('.')[0]
        
        # Calculate expected total
        expected_count = grid_rows * grid_cols
        if expected_count < 1: expected_count = 1
        
        # Process found candidates
        for idx, cnt in enumerate(photo_candidates):
            # Limit to expected count
            if len(results) >= expected_count:
                break

            # Use convex hull to smooth out detections and get a more stable box
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            
            # Slightly shrink the box to ensure we don't pick up white margins
            (center, size, angle) = rect
            size = (size[0] * 0.98, size[1] * 0.98) # Shrink 2%
            rect = (center, size, angle)
            
            # Convert rect to points for returning
            box = cv2.boxPoints(rect)
            box_points = np.array(box, dtype="int").tolist()

            try:
                # Extract and straighten from original (possibly 16-bit) image
                # Note: We pass the rect tuple to _get_warped... which converts to points internally
                # But we also have box_points now.
                cropped = self._get_warped_crop_from_rect(image, rect, crop_margin=crop_margin)
                
                # Apply enhancements
                cropped = self.apply_corrections(cropped, contrast=contrast, auto_contrast=auto_contrast, auto_wb=auto_wb)
                
                # Convert to 8-bit for final storage (User request: save results as 8-bit)
                if cropped.dtype == np.uint16:
                    # Simple scaling: divide by 256
                    cropped = (cropped / 256.0).astype(np.uint8)
                
                output_name = f"{img_base_name}_photo_{len(results)}.png"
                output_path = os.path.join(current_output_dir, output_name)
                # PNG compression: 0-9. 3 is a good balance.
                cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                
                results.append({
                    "path": output_path,
                    "points": box_points
                })
            except Exception as e:
                print(f"Failed to process photo candidate {idx}: {e}")
                
        # Fallback: Ensure we always return 'expected_count' items (or slots)
        # If we missed one, create a placeholder so the user can manually refine it
        while len(results) < expected_count:
            idx = len(results)
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            placeholder[:] = (30, 30, 30) # Dark gray background
            
            # Add text
            text = "Photo Not Detected"
            cv2.putText(placeholder, text, (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            text2 = "Click 'Refine Crop' to set manually"
            cv2.putText(placeholder, text2, (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            output_name = f"{img_base_name}_photo_{idx}.png"
            output_path = os.path.join(current_output_dir, output_name)
            cv2.imwrite(output_path, placeholder)
            
            # Placeholder has no valid points really, but we can provide dummy or empty
            results.append({
                "path": output_path,
                "points": [] 
            })
            
        return results

    def manual_crop(self, image_path: str, points: List[List[int]], photo_index: int, output_subfolder: str = None, contrast: float = 1.0, auto_contrast: bool = False, auto_wb: bool = False) -> str:
        """
        Manually crops a photo from the scan using 4 user-provided points.
        points: List of [x, y] coordinates order: TL, TR, BR, BL
        """
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Determine output directory
        current_output_dir = self.output_dir
        if output_subfolder:
            if os.path.isabs(output_subfolder):
                current_output_dir = output_subfolder
            else:
                current_output_dir = os.path.join(self.output_dir, output_subfolder)
            
            if not os.path.exists(current_output_dir):
                os.makedirs(current_output_dir)

        pts = np.array(points, dtype="float32")
        
        try:
            # Manual crop generally implies no safety margin needed, or minimal, 
            # since user clicked exact points. Let's default to 0 for manual.
            cropped = self._get_warped_crop_from_rect(image, pts, crop_margin=0)
            
            # Apply enhancements
            cropped = self.apply_corrections(cropped, contrast=contrast, auto_contrast=auto_contrast, auto_wb=auto_wb)
            
            # Convert to 8-bit for final storage
            if cropped.dtype == np.uint16:
                cropped = (cropped / 256.0).astype(np.uint8)
            
            img_base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_name = f"{img_base_name}_photo_{photo_index}.png"
            output_path = os.path.join(current_output_dir, output_name)
            
            cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return output_path
        except Exception as e:
            print(f"Manual crop failed: {e}")
            raise e

    def _get_warped_crop_from_rect(self, image: np.ndarray, rect_or_pts, crop_margin: int = 10) -> np.ndarray:
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
        M = cv2.getPerspectiveTransform(src, dst)        # Warp
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        
        # Apply Safety Margin (Inactive Crop) to remove jagged edges/white borders
        # Crop 'crop_margin' pixels from each side
        h_final, w_final = warped.shape[:2]
        if crop_margin > 0 and h_final > (crop_margin * 2) and w_final > (crop_margin * 2):
            warped = warped[crop_margin:-crop_margin, crop_margin:-crop_margin]

        return warped

    def apply_corrections(self, image: np.ndarray, contrast: float = 1.0, auto_contrast: bool = False, brightness: int = 0, auto_wb: bool = False) -> np.ndarray:
        """
        Applies image corrections (Contrast/Brightness, White Balance).
        contrast: 1.0 is neutral, >1.0 increases contrast (applied after auto-contrast).
        auto_contrast: If True, applies histogram clipping (2% low, 1% high) to normalize dynamic range.
        brightness: 0 is neutral, +/- adds to pixel values.
        auto_wb: If True, applies "aggressive" White Balance (RGB channel scaling).
        """
        out = image.copy()
        is_16bit = image.dtype == np.uint16
        max_val = 65535.0 if is_16bit else 255.0
        
        # 1. Aggressive RGB White Balance (Scale to have equal means)
        if auto_wb:
            # We scale R and B to match G's mean luminance
            # This is more "aggressive" and noticeable than the subtle Gray World
            b, g, r = cv2.split(out)
            r_mean = np.mean(r)
            g_mean = np.mean(g)
            b_mean = np.mean(b)
            
            # Avoid division by zero
            if r_mean == 0: r_mean = 1
            if b_mean == 0: b_mean = 1
            
            # Scale R and B
            k_r = g_mean / r_mean
            k_b = g_mean / b_mean
            
            # Use floating point for scaling, convert back safely
            if is_16bit:
                r = cv2.multiply(r.astype(np.float32), k_r).clip(0, max_val).astype(np.uint16)
                b = cv2.multiply(b.astype(np.float32), k_b).clip(0, max_val).astype(np.uint16)
            else:
                r = cv2.convertScaleAbs(r, alpha=k_r)
                b = cv2.convertScaleAbs(b, alpha=k_b)
            
            out = cv2.merge([b, g, r])

        # 2. Auto Contrast (Min-Max Stretching)
        # 2. Auto Contrast (Min-Max Stretching)
        if auto_contrast:
            # OpenCV BGR2LAB does not typically support 16-bit (CV_16U). 
            # We must convert to float32 first if 16-bit.
            original_dtype = out.dtype
            if is_16bit:
                # Convert to float32 (0.0 to 65535.0 or 0..1? OpenCV usually expects 0..1 for float images if displaying, but for conversion it handles structure)
                # Actually for BGR2LAB with float, it expects 0..1 usually.
                # Let's normalize to 0..1
                working_img = out.astype(np.float32) / 65535.0
                
                lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # In float LAB, L is 0..100? or 0..1? 
                # OpenCV docs: BGR (float) -> Lab. L is 0..100, a,b are roughly -127..127
                # Let's check min/max
                # Calculate robust bounds using histogram statistics (CDF)
                # Use 0.05% and 99.95% to avoid clipping details while removing salt/pepper noise
                p_low, p_high = self._get_histogram_bounds(l, 0.05, 99.95)
                
                if p_high > p_low + 1.0: # Ensure reasonable range
                    # Stretch L channel
                    # Target is 0 to 100
                    scale = 100.0 / (p_high - p_low)
                    l = cv2.subtract(l, p_low)
                    l = cv2.multiply(l, scale)
                    l = np.clip(l, 0, 100)
                
                lab = cv2.merge([l, a, b])
                working_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Convert back to 16-bit
                out = (working_img * 65535.0).clip(0, 65535).astype(np.uint16)
                
            else:
                # 8-bit path
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Calculate bounds on 0-255 scale
                p_low, p_high = self._get_histogram_bounds(l, 0.05, 99.95, range_max=255)

                if p_high > p_low + 1.0:
                    # Use float32 for precise stretching
                    l_float = l.astype(np.float32)
                    scale = 255.0 / (p_high - p_low)
                    l_float = (l_float - p_low) * scale
                    l = np.clip(l_float, 0, 255).astype(np.uint8)
                     
                lab = cv2.merge([l, a, b])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                lab = cv2.merge([l, a, b])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. Manual Contrast & Brightness (Applied on top)
        if contrast != 1.0 or brightness != 0:
            if is_16bit:
                # convertScaleAbs is 8-bit output only. We must do math manually or use addWeighted.
                # out = alpha * out + beta
                # We must use cv2.addWeighted or simple multiplication
                
                # out = out * contrast + brightness
                # brightness in 16-bit scale? 
                # Presume 'brightness' input is -100 to 100 relative to 8-bit. Scale it up?
                brightness_16 = brightness * 256
                
                out_f = out.astype(np.float32) * contrast + brightness_16
                out = out_f.clip(0, 65535).astype(np.uint16)
            else:
                out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)
            
        return out

    def _get_histogram_bounds(self, data: np.ndarray, min_percent: float, max_percent: float, range_max: float = 100.0) -> Tuple[float, float]:
        """
        Calculates robust min/max values from histogram to preserve detail.
        min_percent: Percentile for black point (e.g. 0.05)
        max_percent: Percentile for white point (e.g. 99.95)
        range_max: The maximum value of the data domain (100 for Float L, 255 for 8-bit)
        """
        # Ensure flat array
        flat = data.ravel()
        
        # Determine strict float bounds to ensure we handle the type correctly
        # We use a localized histogram for speed and robustness vs sorting
        nbins = 1000 # Enough precision (0.1% of range)
        hist, bin_edges = np.histogram(flat, bins=nbins, range=(0, range_max))
        
        cdf = hist.cumsum()
        total_pixels = cdf[-1]
        
        # Find indices
        low_thresh = total_pixels * (min_percent / 100.0)
        high_thresh = total_pixels * (max_percent / 100.0)
        
        # searchsorted returns the index where the value would be inserted
        idx_low = np.searchsorted(cdf, low_thresh)
        idx_high = np.searchsorted(cdf, high_thresh)
        
        # Clamp indices
        idx_low = min(max(idx_low, 0), nbins - 1)
        idx_high = min(max(idx_high, 0), nbins - 1)
        
        return float(bin_edges[idx_low]), float(bin_edges[idx_high])



    def rotate_photo(self, image_path: str, angle: float) -> str:
        """
        Rotates an existing photo by the specified angle (90 or -90 typically).
        image_path: Absolute path to the photo file.
        """
        filepath = image_path
        
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
