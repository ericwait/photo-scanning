import cv2
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

class StandardSize(Enum):
    WALLET = (2.5, 3.5)
    PHOTO_3X5 = (3.5, 5.0)
    PHOTO_4X6 = (4.0, 6.0)
    PHOTO_5X7 = (5.0, 7.0)
    PHOTO_8X10 = (8.0, 10.0)

@dataclass
class DetectionResult:
    box: List[List[int]]  # 4 points [x, y]
    confidence: float
    size_label: str

class SmartDetector:
    def __init__(self):
        pass

    def detect(self, image: np.ndarray, dpi: int, allowed_sizes: List[str] = None) -> List[DetectionResult]:
        """
        Detects photos in the scanned image using entropy/variance mapping 
        and standard size fitting.
        
        Args:
            image: 8-bit BGR image (numpy array)
            dpi: Scan DPI
            allowed_sizes: Optional list of size labels to allow (e.g. ["PHOTO_4X6", "PHOTO_3X5"])
        """
        if image is None:
            raise ValueError("Image is None")

        # 1. Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Entropy/Texture Map
        # We use local standard deviation as a proxy for entropy/texture
        # It's much faster to compute using integral images or box filters
        entropy_map = self._calculate_texture_map(gray)
        
        # 3. Find Candidates
        candidates = self._find_candidates(entropy_map, dpi)
        
        # 3b. Refine Candidates (Find Paper Edges)
        # Refinement temporarily disabled as it causes performance issues with Canny on high-res scans
        # refined_candidates = []
        # for cand in candidates:
        #    refined_rect = self._refine_candidate_box(gray, cand["contour"])
        #    if refined_rect:
        #         cand["rect"] = refined_rect
        
        # 4. Fit Standard Sizes
        results = self._optimize_fits(candidates, dpi, image.shape, allowed_sizes)
        
        return results

    def _refine_candidate_box(self, gray: np.ndarray, content_contour: np.ndarray) -> Optional[Tuple]:
        """
        Refines the candidate box by looking for the paper edge around the high-entropy content.
        Uses Adaptive Thresholding in a region of interest.
        """
        # ... logic present but disabled in caller ...
        return None 

    def _calculate_texture_map(self, gray: np.ndarray, kernel_size: int = 21) -> np.ndarray:
        # ... existing ...
        return super()._calculate_texture_map(gray, kernel_size) if hasattr(super(), '_calculate_texture_map') else self._calculate_texture_map_impl(gray, kernel_size)
    
    # Preventing replace error, reverting to simple replacement of the method block above
    # Actually I used text replace. I should just target the loop.

    def _optimize_fits(self, candidates: List[Dict], dpi: int, image_shape: Tuple[int, ...], allowed_sizes: List[str] = None) -> List[DetectionResult]:
        """
        Fits standard sizes to the candidates.
        """
        results = []
        
        for cand in candidates:
            # Get the candidate approximate size in pixels
            rect = cand["rect"]
            (cx, cy), (w, h), angle = rect
            
            best_fit_score = float('inf')
            best_label = "Unknown"
            best_size_px = (0, 0)
            
            # Filter StandardSize enum based on allowed_sizes if present
            sizes_to_check = []
            if allowed_sizes:
                for size in StandardSize:
                    if size.name in allowed_sizes:
                        sizes_to_check.append(size)
            else:
                sizes_to_check = list(StandardSize)
            
            # Check against standard sizes
            for size in sizes_to_check:
                sw_in, sl_in = size.value
                
                # Convert to pixels
                sw_px = sw_in * dpi
                sl_px = sl_in * dpi
                
                # Check orientation 1: (w ~ sw, h ~ sl)
                score1 = abs(w - sw_px) + abs(h - sl_px)
                
                # Check orientation 2: (w ~ sl, h ~ sw)
                score2 = abs(w - sl_px) + abs(h - sw_px)
                
                if score1 < best_fit_score:
                    best_fit_score = score1
                    best_label = size.name
                    best_size_px = (sw_px, sl_px) # Matches w, h orientation
                    
                if score2 < best_fit_score:
                    best_fit_score = score2
                    best_label = size.name
                    best_size_px = (sl_px, sw_px) # Matches w, h orientation
            
            # tolerance
            tolerance = dpi * 1.5 # Relaxed tolerance (1.5 inches) because we might detect content inside margin
            
            # IMPROVEMENT: If we didn't match a standard size within strict tolerance, 
            # check if we are "close enough" (e.g. +/- 20%) to one, and snap to it anyway.
            # This handles cases where crop is slightly off or paper has borders.
            
            final_rect = None
            final_label = "Custom"
            final_confidence = 0.5
            
            if best_fit_score < tolerance:
                # Good match
                final_size = best_size_px
                final_rect = ((cx, cy), final_size, angle)
                final_label = best_label
                final_confidence = 1.0 - (best_fit_score / tolerance)
                
            elif allowed_sizes and "Custom" not in allowed_sizes:
                 # Strict mode: If allowed_sizes is set, we try relative error fitting for allowed sizes only.
                 best_rel_error = float('inf')
                 best_rel_size = None
                 best_rel_label = None
                 
                 for size in sizes_to_check:
                    sw_in, sl_in = size.value
                    sw_px = sw_in * dpi
                    sl_px = sl_in * dpi
                    
                    # Check dim1/dim2 vs sw/sl
                    # dim1 is smaller, dim2 is larger (from sorted((w,h)))
                    std_min, std_max = sorted((sw_px, sl_px))
                    detected_min, detected_max = sorted((w, h))
                    
                    err_min = abs(detected_min - std_min) / std_min
                    err_max = abs(detected_max - std_max) / std_max
                    
                    avg_err = (err_min + err_max) / 2.0
                    
                    if avg_err < 0.20: # 20% tolerance
                        if avg_err < best_rel_error:
                            best_rel_error = avg_err
                            # Which orientation?
                            if w < h:
                                best_rel_size = (std_min, std_max)
                            else:
                                best_rel_size = (std_max, std_min)
                            best_rel_label = size.name
                 
                 if best_rel_size:
                     final_size = best_rel_size
                     final_rect = ((cx, cy), final_size, angle)
                     final_label = best_rel_label
                     final_confidence = 0.8 - best_rel_error # Lower confidence than direct fit
                 else:
                     # Strict mode and no match found
                     final_rect = None
                     
            else:
                 # Original logic for fallback if allowed_sizes is NOT set (or includes Custom implicitly?)
                 # For now, if allowed_sizes is None, we use existing fallback logic
                 
                 # ... existing relative error logic ...
                 best_rel_error = float('inf')
                 best_rel_size = None
                 best_rel_label = None
                 
                 for size in sizes_to_check: # Logic duplicated but filtered by check list
                    sw_in, sl_in = size.value
                    sw_px = sw_in * dpi
                    sl_px = sl_in * dpi
                    
                    std_min, std_max = sorted((sw_px, sl_px))
                    detected_min, detected_max = sorted((w, h))
                    
                    err_min = abs(detected_min - std_min) / std_min
                    err_max = abs(detected_max - std_max) / std_max
                    avg_err = (err_min + err_max) / 2.0
                    
                    if avg_err < 0.20: 
                        if avg_err < best_rel_error:
                            best_rel_error = avg_err
                            if w < h: best_rel_size = (std_min, std_max)
                            else: best_rel_size = (std_max, std_min)
                            best_rel_label = size.name
                 
                 if best_rel_size:
                     final_size = best_rel_size
                     final_rect = ((cx, cy), final_size, angle)
                     final_label = best_rel_label
                     final_confidence = 0.8 - best_rel_error 
                 else:
                     # True Custom
                     # Only accept if reasonably photosized (e.g. > 2x2 inches)
                     # 1.5 inch min dimension
                     dim1, dim2 = sorted((w, h))
                     if dim1 > dpi * 1.5 and dim2 > dpi * 1.5: 
                        final_rect = rect
                        final_label = "Custom"
                        final_confidence = 0.5
            
            if final_rect:
                # Convert rotated rect to 4 points
                box = cv2.boxPoints(final_rect)
                box = np.int64(box)
                
                results.append(DetectionResult(
                    box=box.tolist(),
                    confidence=final_confidence, 
                    size_label=final_label
                ))
                    
        return results
        """
        Refines the candidate box by looking for the paper edge around the high-entropy content.
        Uses Adaptive Thresholding in a region of interest.
        """
        x, y, w, h = cv2.boundingRect(content_contour)
        
        # Define expansion margin (e.g. 1 inch = ~300-400 px, or just a fixed large buffer)
        margin = 100 
        
        h_img, w_img = gray.shape[:2]
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Use Otsu's thresholding in the ROI to find the paper object vs background
        # Assume paper is darker than the white lid background? NO.
        # Paper is white, Lid is white. Edge is shadow.
        # Adaptive was better for edges, but maybe hanging?
        # Let's try simple gradient/Canny?
        # Or just use Otsu on the INVERTED image (if finding dark edge).
        
        # Let's try Otsu on blurred ROI
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        # Try finding the object directly. 
        # Usually Photo Content is darker than White Page Border is darker/lighter than White Lid?
        # Actually photos are usually darker than the lid.
        # But we want the Paper Edge.
        
        # Let's stick to Canny for Edge Detection, it's robust and fast.
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to close the loop
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 500:
            # Too noisy, abort refinement
            return cv2.minAreaRect(content_contour)
        
        # Find largest contour that is "larger" than content_contour (or simply largest in ROI)
        # The content contour is in global coords. We need to check relative.
        
        # Heuristic: Largest contour in ROI is likely the paper edge (plus maybe some noise attached)
        # We can also check if it contains the center of the ROI.
        if not contours:
            return cv2.minAreaRect(content_contour)
            
        largest = max(contours, key=cv2.contourArea)
        
        # If largest contour is significantly smaller than content, something went wrong.
        # Otherwise, use it.
        # But we need to map it back to global coordinates.
        
        # minAreaRect in ROI
        (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(largest)
        
        # Shift center to global
        gcx = rcx + x1
        gcy = rcy + y1
        
        # If the refined size is smaller than original content, reject it (it's noise)
        orig_rect = cv2.minAreaRect(content_contour)
        orig_area = orig_rect[1][0] * orig_rect[1][1]
        new_area = rw * rh
        
        if new_area < orig_area * 0.8:
            return orig_rect
            
        return ((gcx, gcy), (rw, rh), rang)

    def _calculate_texture_map(self, gray: np.ndarray, kernel_size: int = 21) -> np.ndarray:
        """
        Calculates a texture map (std deviation) of the image.
        High values indicate photo content, low values indicate background.
        """
        # Convert to float for precision
        gray_f = gray.astype(np.float32)
        
        # Blur to reduce noise before texture calc? 
        # Or just compute local std dev: sqrt(E[x^2] - (E[x])^2)
        
        # E[x]
        mu = cv2.blur(gray_f, (kernel_size, kernel_size))
        
        # E[x^2]
        mu2 = cv2.blur(gray_f * gray_f, (kernel_size, kernel_size))
        
        # Variance = E[x^2] - (E[x])^2
        variance = mu2 - mu * mu
        
        # Std Dev
        # Clip negative values due to floating point errors
        sigma = np.sqrt(np.maximum(variance, 0))
        
        # Normalize to 0-255
        sigma = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return sigma

    def _find_candidates(self, texture_map: np.ndarray, dpi: int) -> List[Dict]:
        """
        Finds rough regions of interest (blobs) in the texture map.
        """
        # Threshold the texture map
        # Assume background is smooth (low texture), photos are detailed (high texture)
        # Otsu failed (merged background), so we use a higher fixed threshold to cut out background noise.
        _, thresh = cv2.threshold(texture_map, 50, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        # Use RETR_TREE to handle nested contours (Frame -> Photos)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        h, w = texture_map.shape[:2]
        total_area = h * w
        min_area = (dpi * 1.0) ** 2  # Min 1 square inch
        
        if contours and hierarchy is not None:
             hierarchy = hierarchy[0] # Flatten
             
             for i, c in enumerate(contours):
                rect = cv2.minAreaRect(c)
                box_area = rect[1][0] * rect[1][1]
                
                # Filter out tiny noise
                if box_area < min_area:
                    continue
                
                # Check for Container/Frame
                # If it's a Parent (has child) and is Large (> 25% area)
                # hierarchy: [Next, Prev, First_Child, Parent]
                # child index != -1 means it has a child
                has_child = hierarchy[i][2] != -1
                
                if has_child and box_area > (total_area * 0.25):
                     # This is likely the scanner frame containing the photos
                     continue
                
                # Filter out giant contours that are just too big regardless of hierarchy (e.g. 95%)
                if box_area > (total_area * 0.95):
                    continue

                candidates.append({
                    "contour": c,
                    "rect": rect  # (center(x, y), (width, height), angle)
                })
            
        return candidates

    def _optimize_fits(self, candidates: List[Dict], dpi: int, image_shape: Tuple[int, ...]) -> List[DetectionResult]:
        """
        Fits standard sizes to the candidates.
        """
        results = []
        
        for cand in candidates:
            # Get the candidate approximate size in pixels
            rect = cand["rect"]
            (cx, cy), (w, h), angle = rect
            
            # Normalize w, h so w is always the smaller dimension for comparison
            dim1, dim2 = sorted((w, h))
            
            best_fit_score = float('inf')
            best_label = "Unknown"
            best_points = None
            
            # Check against standard sizes
            for size in StandardSize:
                sw_in, sl_in = size.value
                
                # Convert to pixels
                sw_px = sw_in * dpi
                sl_px = sl_in * dpi
                
                # Check both orientations
                # 1. Match dim1 to sw_px, dim2 to sl_px
                # Score = abs(diff_w) + abs(diff_h)
                score1 = abs(dim1 - sw_px) + abs(dim2 - sl_px)
                
                # We could also check rotated? But dim1/dim2 are sorted
                
                if score1 < best_fit_score:
                    best_fit_score = score1
                    best_label = size.name
                    # Create a box of this size centered at cx, cy with angle
                    # for now just taking the minAreaRect box, but ideally we force the aspect ratio
            
            # If the best score is reasonable (e.g. within 1 inch error?)
            # 1 inch = dpi pixels.
            tolerance = dpi * 1.0 
            
            if best_fit_score < tolerance:
                
                # Force the box to be exactly the standard size
                # Determine orientation
                # If the detection says it's portrait (h > w), we use the portrait version of standard size
                # StandardSize values are typically (min, max) or just defined. 
                # Let's ensure sw_px is the smaller, sl_px is the larger
                std_min, std_max = sorted((sw_px, sl_px))
                
                # Check detection orientation
                if w < h:
                    new_size = (std_min, std_max)
                else:
                    new_size = (std_max, std_min)
                
                # Create a new rect with the standard size
                new_rect = ((cx, cy), new_size, angle)
                
                # Convert rotated rect to 4 points
                box = cv2.boxPoints(new_rect)
                box = np.int64(box)
                
                results.append(DetectionResult(
                    box=box.tolist(),
                    confidence=1.0 - (best_fit_score / tolerance), # rough confidence
                    size_label=best_label
                ))
            else:
                 # If it doesn't match a standard size well, maybe keep it anyway if it's large enough?
                 # For now, simplistic acceptance if large enough
                 if dim1 > dpi * 2 and dim2 > dpi * 2: # Min 2x2 inches
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    results.append(DetectionResult(
                        box=box.tolist(),
                        confidence=0.5,
                        size_label="Custom"
                    ))
                    
        return results
