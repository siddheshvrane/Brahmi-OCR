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
    
    # Find Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def sort_boxes(boxes, y_threshold=20):
    """
    Sorts bounding boxes from top to bottom, left to right.
    
    Args:
        boxes: List of (x, y, w, h) tuples.
        y_threshold: Vertical distance threshold to consider boxes in the same line.
        
    Returns:
        Sorted list of boxes.
    """
    if not boxes:
        return []

    # Sort by Y coordinate first
    # This helps but isn't enough for clean line detection if y varies slightly
    boxes = sorted(boxes, key=lambda b: b[1])
    
    lines = []
    current_line = [boxes[0]]
    
    for i in range(1, len(boxes)):
        x, y, w, h = boxes[i]
        prev_x, prev_y, prev_w, prev_h = current_line[-1]
        
        # Check if the current box is roughly on the same line as the previous one
        # using the center y or just y diff
        center_y = y + h // 2
        prev_center_y = prev_y + prev_h // 2
        
        if abs(center_y - prev_center_y) <= y_threshold:
            current_line.append(boxes[i])
        else:
            # New line
            lines.append(current_line)
            current_line = [boxes[i]]
            
    lines.append(current_line)
    
    # Sort each line by X coordinate
    sorted_boxes = []
    for line in lines:
        # Sort by x
        line.sort(key=lambda b: b[0])
        sorted_boxes.extend(line)
        
    return sorted_boxes
