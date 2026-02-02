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

    def detect(self, image: np.ndarray, dpi: int) -> List[DetectionResult]:
        """
        Detects photos in the scanned image using entropy/variance mapping 
        and standard size fitting.
        
        Args:
            image: 8-bit BGR image (numpy array)
            dpi: Scan DPI
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
        candidates = self._find_candidates(entropy_map)
        
        # 4. Fit Standard Sizes
        results = self._optimize_fits(candidates, dpi, image.shape)
        
        return results

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

    def _find_candidates(self, texture_map: np.ndarray) -> List[Dict]:
        """
        Finds rough regions of interest (blobs) in the texture map.
        """
        # Threshold the texture map
        # Assume background is smooth (low texture), photos are detailed (high texture)
        _, thresh = cv2.threshold(texture_map, 20, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for c in contours:
            rect = cv2.minAreaRect(c)
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
