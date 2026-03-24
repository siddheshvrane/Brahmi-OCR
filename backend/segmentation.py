import cv2
import numpy as np

def clean_image_noise(image_bgr, min_dot_area=50):
    """
    Removes small dots (stone noise) from the image using connected components.
    Binarizes the image, keeps only large connected components, and returns a clean image.
    Outputs a clean image with black characters on a white background.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Median Blur to kill 1px salt-and-pepper noise
    gray_blurred = cv2.medianBlur(gray, 3)
    
    # 2. Global Otsu Thresholding
    # Text becomes white (255) on black (0) background
    _, thresh = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morphological closing to join slightly disjoint strokes (prevents removing chunks of characters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Prepare a clean white canvas
    clean_bgr = np.full_like(image_bgr, 255)
    
    # 5. Draw valid components in black on the white canvas
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep components larger than `min_dot_area` threshold
        if area >= min_dot_area:
            clean_bgr[labels == i] = [0, 0, 0]
            
    return clean_bgr

def remove_background_noise(image_bgr, min_dot_area=60):
    """
    Cleans floating stone texture and pepper noise from the image.
    Uses a structural proximity algorithm:
    - Long strokes or heavy blobs are marked as core character structures.
    - Small blobs (stone texture vs valid punctuation dots) are judged by proximity.
    - If a small blob is near a core structure, it's preserved as punctuation (Visarga/fragment).
    - If a small blob is floating in the background, it's nuked.
    """
    # 1. Median Blur to kill 1-2px intense salt-and-pepper noise safely instantly
    blurred = cv2.medianBlur(image_bgr, 3)
    
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Morphological close to join slightly disjoint strokes physically 
    # (only for structural analysis, not drawn to final output)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    cleaned_bgr = blurred.copy()
    
    # 3. Distinguish "core structures" vs "small dots"
    large_mask = np.zeros_like(thresh)
    small_labels = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # A core stroke is either physically long/wide (w or h >= 20) or massive (area >= 150)
        # Anything else is a tiny isolated dot or artifact
        if area >= 150 or max(w, h) >= 20:
            large_mask[labels == i] = 255
        else:
            small_labels.append((i, area))
            
    # 4. Dilate the core structures to create a "Safe Zone" 
    # Distance = 25px kernel (roughly 12px radius) around a core stroke
    safe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    safe_zone = cv2.dilate(large_mask, safe_kernel)
    
    # 5. Evaluate small dots based on location
    for i, area in small_labels:
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # Clamp centroid inside image bounds (safety)
        cy = max(0, min(cy, safe_zone.shape[0]-1))
        cx = max(0, min(cx, safe_zone.shape[1]-1))
        
        is_safe = False
        # If it's incredibly tiny pepper dust (< 10 area), always nuke it
        if area < 10:
            is_safe = False
        # If it's lying inside the safe zone, it's a valid punctuation / broken character piece
        elif safe_zone[cy, cx] == 255:
            is_safe = True
            
        if not is_safe:
            cleaned_bgr[labels == i] = [255, 255, 255]
            
    return cleaned_bgr


def merge_nested_boxes(boxes):
    """
    If a box is completely 100% inside another box, discard the inner one.
    Args:
        boxes: List of (x, y, w, h)
    Returns:
        Filtered list of boxes.
    """
    if not boxes:
        return []
        
    filtered = []
    for i in range(len(boxes)):
        x1, y1, w1, h1 = boxes[i]
        is_nested = False
        for j in range(len(boxes)):
            if i == j:
                continue
            x2, y2, w2, h2 = boxes[j]
            
            # Check if boxes[i] is inside boxes[j]
            # Tie-breaker handles identical boxes (keeps one)
            if (x1 >= x2 and y1 >= y2 and 
                (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)):
                if (x1 == x2 and y1 == y2 and w1 == w2 and h1 == h2):
                    if i > j: # Drop later duplicate
                        is_nested = True
                        break
                else:
                    is_nested = True
                    break
        
        if not is_nested:
            filtered.append(boxes[i])
            
    return filtered

# Target aspect ratio derived from dataset analysis: avg width / height = 0.7145
TARGET_ASPECT_RATIO = 0.7145  # width / height

def normalize_box_aspect_ratio(x, y, w, h, img_width, img_height, target_ar=TARGET_ASPECT_RATIO):
    """
    Adjusts a bounding box so its width/height ratio matches `target_ar`.
    Expands the shorter dimension (keeping the box centred) rather than shrinking,
    so we never clip character pixels.  Result is clamped to image boundaries.
    """
    current_ar = w / h if h > 0 else target_ar

    if current_ar < target_ar:
        # Box is too tall — needs to be wider
        new_w = int(round(h * target_ar))
        delta = new_w - w
        x = x - delta // 2
        w = new_w
    else:
        # Box is too wide — needs to be taller
        new_h = int(round(w / target_ar))
        delta = new_h - h
        y = y - delta // 2
        h = new_h

    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(img_width - x, w)
    h = min(img_height - y, h)

    return int(x), int(y), int(w), int(h)


def detect_characters(image_input, min_area=100, padding_ratio=0.15):
    """
    Detects characters with improved "Lens-style" closing.
    Merging logic has been added for 100% nested boxes.
    """
    # Read image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image at {image_input}")
    else:
        img = image_input

    if img is None:
        raise ValueError("Invalid image input")

    height, width = img.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find Contours using RETR_LIST so we catch characters inside drawn border frames
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    initial_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0.9 * width or h > 0.9 * height:
                continue
            initial_boxes.append((x, y, w, h))
            
    # Filter Nested Boxes: 100% inner boxes are removed
    initial_boxes = merge_nested_boxes(initial_boxes)

    # Apply padding and ensure box is within image bounds
    final_boxes = []
    # Calculate padding dynamically based on character size (e.g., 15% of the max dimension)
    for (x, y, w, h) in initial_boxes:
        # Dynamic padding: proportionate breathing room
        pad = int(max(w, h) * padding_ratio)
        
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(width - x_pad, w + 2 * pad)
        h_pad = min(height - y_pad, h + 2 * pad)

        # Normalize to consistent aspect ratio (dataset average: width/height = 0.7145)
        x_pad, y_pad, w_pad, h_pad = normalize_box_aspect_ratio(
            x_pad, y_pad, w_pad, h_pad, width, height
        )
        final_boxes.append((x_pad, y_pad, w_pad, h_pad))
            
    return final_boxes, img

def sort_boxes(boxes, y_threshold=None):
    """
    Sorts bounding boxes from top to bottom, left to right.
    Uses dynamic line grouping based on median character height to properly cluster 
    slightly misaligned characters on the exact same visible line.
    """
    if not boxes:
        return []

    # Calculate dynamic y_threshold (e.g. 50% of the median character height)
    # This prevents fixed thresholds (like 20px) from failing on high-res images
    if y_threshold is None:
        import numpy as np
        median_h = np.median([b[3] for b in boxes])
        y_threshold = max(20, median_h * 0.5)

    # Sort strictly by Top Y initially
    boxes = sorted(boxes, key=lambda b: b[1])
    
    lines = []
    current_line = [boxes[0]]
    # Running average center Y for the current line
    current_line_center_y = boxes[0][1] + boxes[0][3] // 2
    
    for i in range(1, len(boxes)):
        x, y, w, h = boxes[i]
        center_y = y + h // 2
        
        # Check against the entire line's average (prevents drifting separation)
        if abs(center_y - current_line_center_y) <= y_threshold:
            current_line.append(boxes[i])
            # Update running average
            current_line_center_y = sum(b[1] + b[3] // 2 for b in current_line) / len(current_line)
        else:
            # Shift to Next Line
            lines.append(current_line)
            current_line = [boxes[i]]
            current_line_center_y = y + h // 2
            
    lines.append(current_line)
    
    # Finally sort each constructed line strictly by X coordinate (Left to Right)
    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        sorted_boxes.extend(line)
        
    return sorted_boxes
