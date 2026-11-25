"""
Photometric Conditioning and Adaptive Image Enhancement.
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE),
bilateral filtering for asphalt texture preservation, and wet road glare suppression.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

class PhotometricEnhancer:
    """Enhances raw camera frames under difficult lighting (shadows, wet road glare)."""
    
    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply luminance normalization in LAB color space."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")
            
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Equalize only L (Lightness) channel to avoid color shifting
        l_eq = self.clahe.apply(l_channel)
        
        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((l_eq, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def suppress_glare(self, frame: np.ndarray, threshold: int = 240) -> np.ndarray:
        """Attenuate specular reflections from water puddles and direct sunlight."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glare_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Inpaint glare regions using Navier-Stokes based fluid dynamics
        inpainted = cv2.inpaint(frame, glare_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        return inpainted

    def bilateral_smooth(self, frame: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
        """Smooth asphalt noise while preserving sharp pothole rim edges."""
        return cv2.bilateralFilter(frame, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
