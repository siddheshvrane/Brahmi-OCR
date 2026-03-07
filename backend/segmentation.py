import cv2
import numpy as np

def detect_characters(image_input, min_area=100, padding=40):
    """
    Detects characters with improved "Lens-style" closing.
    Merging logic has been removed to keep overlapping characters separate.
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
    
    # Morphological Closing: Connect broken strokes and keep components together
    # Slightly larger kernel for ancient scripts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    initial_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Ignore massive contours that are likely page borders or artifacts
            if w > 0.9 * width or h > 0.9 * height:
                continue
                
            initial_boxes.append((x, y, w, h))
            
    # Apply padding and ensure box is within image bounds
    final_boxes = []
    # Using the padding value passed as an argument to give characters breathing room
    for (x, y, w, h) in initial_boxes:
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(width - x_pad, w + 2 * padding)
        h_pad = min(height - y_pad, h + 2 * padding)
        final_boxes.append((int(x_pad), int(y_pad), int(w_pad), int(h_pad)))
            
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
