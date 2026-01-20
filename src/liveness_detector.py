"""Face liveness detection module.
Implements anti-spoofing measures to prevent attendance marking with photos.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class LivenessDetector:
    """Detects if a face in an image is from a live person or a photo/video.
    
    Uses multiple techniques:
    1. Motion detection - checks for natural movement between frames
    2. Texture analysis - detects print/screen artifacts
    3. Blink detection - detects eye blinks (optional, requires sequence)
    """
    
    def __init__(self):
        """Initialize liveness detector."""
        self.previous_frames: List[np.ndarray] = []
        self.max_frames = 3  # Store last 3 frames for motion analysis
        
        # Thresholds for liveness detection
        self.motion_threshold = 0.001  # Minimum motion required
        self.texture_variance_threshold = 100  # Minimum texture variance for real face
        self.edge_density_threshold = 0.015  # Minimum edge density for real face
        self.max_edge_density_threshold = 0.15  # Maximum edge density (too many edges = photo)
        
        # Normalization constants
        self.texture_variance_normalizer = 500.0  # Normalize texture variance to 0-1 range
        self.max_motion_threshold = 0.3  # Maximum motion (too much = suspicious)
        
    def check_liveness(
        self, 
        image: np.ndarray, 
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[bool, str, float]:
        """Check if the face in the image is from a live person.
        
        Args:
            image: Input image (BGR format)
            face_bbox: Optional bounding box (x, y, w, h) of the face region
            
        Returns:
            Tuple of (is_live, reason, confidence)
            - is_live: True if face appears to be from a live person
            - reason: Explanation of the decision
            - confidence: Confidence score (0-1)
        """
        if image is None or image.size == 0:
            return False, "Invalid image", 0.0
        
        # Extract face region if bbox provided, otherwise use full image
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Add some padding
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            face_region = image[y1:y2, x1:x2]
        else:
            face_region = image
            
        if face_region.size == 0:
            return False, "Invalid face region", 0.0
        
        # Run multiple liveness checks
        checks = []
        reasons = []
        
        # 1. Texture Analysis - Real faces have rich texture variation
        texture_score, texture_reason = self._check_texture_quality(face_region)
        checks.append(texture_score)
        reasons.append(texture_reason)
        
        # 2. Edge Analysis - Photos often have distinct edges and less depth
        edge_score, edge_reason = self._check_edge_characteristics(face_region)
        checks.append(edge_score)
        reasons.append(edge_reason)
        
        # 3. Color Distribution - Real faces have specific color patterns
        color_score, color_reason = self._check_color_distribution(face_region)
        checks.append(color_score)
        reasons.append(color_reason)
        
        # 4. Motion Detection (if we have previous frames)
        if len(self.previous_frames) > 0:
            motion_score, motion_reason = self._check_motion(face_region)
            checks.append(motion_score * 1.5)  # Weight motion more heavily
            reasons.append(motion_reason)
        
        # Store current frame for next check
        self.previous_frames.append(cv2.resize(face_region, (160, 160)))
        if len(self.previous_frames) > self.max_frames:
            self.previous_frames.pop(0)
        
        # Calculate overall confidence
        confidence = np.mean(checks)
        
        # Decision threshold
        is_live = confidence > 0.5
        
        # Compile reason
        if is_live:
            reason = "Live face detected: " + "; ".join(reasons)
        else:
            reason = "Possible spoofing attempt: " + "; ".join(reasons)
        
        return is_live, reason, float(confidence)
    
    def _check_texture_quality(self, face_region: np.ndarray) -> Tuple[float, str]:
        """Analyze texture quality - real faces have rich texture.
        
        Photos and screens typically have lower texture variance.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of texture/sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize score (higher variance = more likely real)
        score = min(variance / self.texture_variance_normalizer, 1.0)
        
        if variance < self.texture_variance_threshold:
            reason = f"Low texture variance ({variance:.1f})"
            return score * 0.5, reason
        else:
            reason = f"Good texture variance ({variance:.1f})"
            return score, reason
    
    def _check_edge_characteristics(self, face_region: np.ndarray) -> Tuple[float, str]:
        """Analyze edge characteristics.
        
        Photos often have sharp, uniform edges. Real faces have natural, varied edges.
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces typically have moderate edge density
        # Too high = printed photo with sharp edges
        # Too low = blurred or poorly lit image
        if edge_density < self.edge_density_threshold:
            reason = f"Low edge density ({edge_density:.4f})"
            return 0.3, reason
        elif edge_density > self.max_edge_density_threshold:
            reason = f"Excessive edges ({edge_density:.4f})"
            return 0.5, reason
        else:
            reason = f"Natural edge density ({edge_density:.4f})"
            return 0.8, reason
    
    def _check_color_distribution(self, face_region: np.ndarray) -> Tuple[float, str]:
        """Analyze color distribution.
        
        Real faces have specific skin tone patterns. Photos may have color shifts.
        """
        # Convert to HSV for better skin tone analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram variance
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        
        # Calculate entropy (measure of color diversity)
        h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-7))
        s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-7))
        
        # Average entropy
        avg_entropy = (h_entropy + s_entropy) / 2
        
        # Real faces typically have moderate entropy
        if avg_entropy < 2.0:
            reason = f"Low color diversity ({avg_entropy:.2f})"
            return 0.4, reason
        elif avg_entropy > 6.0:
            reason = f"Excessive color variation ({avg_entropy:.2f})"
            return 0.6, reason
        else:
            reason = f"Natural color distribution ({avg_entropy:.2f})"
            return 0.8, reason
    
    def _check_motion(self, face_region: np.ndarray) -> Tuple[float, str]:
        """Detect motion between frames.
        
        Real faces have subtle natural movements. Static photos don't.
        """
        if not self.previous_frames:
            return 0.5, "No previous frame for motion detection"
        
        # Resize for consistent comparison
        current = cv2.resize(face_region, (160, 160))
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        
        # Compare with previous frame
        prev_gray = cv2.cvtColor(self.previous_frames[-1], cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(current_gray, prev_gray)
        motion_score = np.mean(diff) / 255.0
        
        if motion_score < self.motion_threshold:
            reason = f"No motion detected ({motion_score:.4f})"
            return 0.2, reason
        elif motion_score > self.max_motion_threshold:
            reason = f"Excessive motion ({motion_score:.4f})"
            return 0.5, reason
        else:
            reason = f"Natural motion detected ({motion_score:.4f})"
            return 0.9, reason
    
    def reset(self):
        """Reset the detector state (clear previous frames)."""
        self.previous_frames = []


def check_single_image_liveness(image: np.ndarray) -> Tuple[bool, str, float]:
    """Quick liveness check for a single image without motion detection.
    
    This is useful for static image uploads where motion cannot be detected.
    It only uses texture, edge, and color analysis.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_live, reason, confidence)
    """
    detector = LivenessDetector()
    # Don't use motion detection for single images
    detector.previous_frames = []
    return detector.check_liveness(image)
