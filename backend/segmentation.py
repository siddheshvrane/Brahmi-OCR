import cv2
import numpy as np

def merge_boxes(boxes, x_overlap_thresh=10, y_overlap_thresh=20):
    """
    Intelligently merges bounding boxes that are close to each other or overlap.
    Crucial for scripts like Brahmi where vowel marks are detached.
    """
    if not boxes:
        return []

    # Convert to list for manip
    merged = []
    
    # Sort boxes by X to start merging candidates
    boxes = sorted(boxes, key=lambda b: b[0])
    
    while boxes:
        curr = list(boxes.pop(0)) # [x, y, w, h]
        cx1, cy1, cx2, cy2 = curr[0], curr[1], curr[0] + curr[2], curr[1] + curr[3]
        
        i = 0
        while i < len(boxes):
            next_box = boxes[i]
            nx1, ny1, nx2, ny2 = next_box[0], next_box[1], next_box[0] + next_box[2], next_box[1] + next_box[3]
            
            # Check for vertical overlap (same X range) or horizontal proximity
            # Horizontal overlap check
            has_x_overlap = not (cx2 < nx1 - x_overlap_thresh or nx2 < cx1 - x_overlap_thresh)
            # Vertical proximity check (important for vowel marks above/below)
            # We check if they are within y_overlap_thresh even if they don't overlap vertically
            y_dist = 0
            if cy2 < ny1: 
                y_dist = ny1 - cy2
            elif ny2 < cy1:
                y_dist = cy1 - ny2
            
            has_y_proximity = y_dist < y_overlap_thresh

            if has_x_overlap and has_y_proximity:
                # Merge: take the envelope of both boxes
                mx1 = min(cx1, nx1)
                my1 = min(cy1, ny1)
                mx2 = max(cx2, nx2)
                my2 = max(cy2, ny2)
                
                # Update current merged box
                cx1, cy1, cx2, cy2 = mx1, my1, mx2, my2
                curr = [cx1, cy1, cx2 - cx1, cy2 - cy1]
                
                # Remove the merged box and reset loop to check again with updated curr
                boxes.pop(i)
                i = 0 
            else:
                i += 1
        
        merged.append(tuple(curr))
        
    return merged

def detect_characters(image_input, min_area=100, padding=20):
    """
    Detects characters with improved "Lens-style" merging and closing.
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    initial_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            initial_boxes.append((x, y, w, h))
            
    # Intelligent Merging
    merged_boxes = merge_boxes(initial_boxes)
    
    # Apply padding and ensure box is within image bounds
    final_boxes = []
    for (x, y, w, h) in merged_boxes:
        # We use a more "loose" padding strategy
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
