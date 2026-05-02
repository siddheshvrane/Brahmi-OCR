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

    # 3. Morphological closing to join slightly disjoint strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Prepare a clean white canvas
    clean_bgr = np.full_like(image_bgr, 255)

    # 5. Draw valid components in black on the white canvas
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
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
    # 1. Median Blur to kill 1-2px intense salt-and-pepper noise
    blurred = cv2.medianBlur(image_bgr, 3)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Morphological close to join slightly disjoint strokes physically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    cleaned_bgr = blurred.copy()

    # 3. Distinguish "core structures" vs "small dots"
    large_mask  = np.zeros_like(thresh)
    small_labels = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]

        if area >= 150 or max(w, h) >= 20:
            large_mask[labels == i] = 255
        else:
            small_labels.append((i, area))

    # 4. Dilate the core structures to create a "Safe Zone"
    safe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    safe_zone   = cv2.dilate(large_mask, safe_kernel)

    # 5. Evaluate small dots based on location
    for i, area in small_labels:
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cy = max(0, min(cy, safe_zone.shape[0] - 1))
        cx = max(0, min(cx, safe_zone.shape[1] - 1))

        is_safe = False
        if area < 10:
            is_safe = False
        elif safe_zone[cy, cx] == 255:
            is_safe = True

        if not is_safe:
            cleaned_bgr[labels == i] = [255, 255, 255]

    return cleaned_bgr


def merge_nested_boxes(boxes):
    """
    If a box is completely 100% inside another box, discard the inner one.
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
            if (x1 >= x2 and y1 >= y2 and
                (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)):
                if (x1 == x2 and y1 == y2 and w1 == w2 and h1 == h2):
                    if i > j:
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


def normalize_box_aspect_ratio(x, y, w, h, img_width, img_height,
                                target_ar=TARGET_ASPECT_RATIO,
                                max_expand_ratio=1.5):
    """
    Adjusts a bounding box so its width/height ratio approaches `target_ar`.

    Expands the shorter dimension (keeping the box centred) rather than shrinking,
    so we never clip character pixels. Result is clamped to image boundaries.

    max_expand_ratio caps how much the HEIGHT can grow via AR normalization.
    Without this cap, a wide character's box (AR >> target_ar) gets its height
    doubled or tripled, causing it to bleed into the next line and confusing
    sort_boxes. Default cap = 1.5× original height.
    """
    current_ar = w / h if h > 0 else target_ar

    if current_ar < target_ar:
        # Box is too tall — widen it
        new_w  = int(round(h * target_ar))
        delta  = new_w - w
        x      = x - delta // 2
        w      = new_w
    else:
        # Box is too wide — would need to grow taller.
        # Cap the height growth to avoid bleeding into adjacent lines.
        new_h   = int(round(w / target_ar))
        max_h   = int(h * max_expand_ratio)
        new_h   = min(new_h, max_h)          # ← KEY FIX: cap height expansion
        delta   = new_h - h
        y       = y - delta // 2
        h       = new_h

    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(img_width  - x, w)
    h = min(img_height - y, h)

    return int(x), int(y), int(w), int(h)


def detect_characters(image_input, min_area=100, padding_ratio=0.15):
    """
    Detects characters with improved "Lens-style" closing.
    Merging logic has been added for 100% nested boxes.
    """
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
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

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

    # Filter Nested Boxes
    initial_boxes = merge_nested_boxes(initial_boxes)

    # Apply padding and AR normalization
    final_boxes = []
    for (x, y, w, h) in initial_boxes:
        pad   = int(max(w, h) * padding_ratio)
        x_pad = max(0, x - pad)
        y_pad = max(0, y - pad)
        w_pad = min(width  - x_pad, w + 2 * pad)
        h_pad = min(height - y_pad, h + 2 * pad)

        x_pad, y_pad, w_pad, h_pad = normalize_box_aspect_ratio(
            x_pad, y_pad, w_pad, h_pad, width, height
        )
        final_boxes.append((x_pad, y_pad, w_pad, h_pad))

    return final_boxes, img


def sort_boxes(boxes):
    """
    Sorts bounding boxes from top-to-bottom, left-to-right using
    robust center-Y clustering for line grouping.

    WHY THE OLD APPROACH FAILED
    ───────────────────────────
    The old algorithm relied on top-Y sorting and a sequential greedy overlap.
    If a character from a lower line had a tall ascender (small top-Y), it was 
    sorted before the rest of its line, and its large bounding box would heavily 
    overlap with the line above it. This caused characters from below lines to 
    get "jumbled upon" and grouped into the above lines.

    NEW APPROACH: CENTER-Y CLUSTERING
    ─────────────────────────────────
    1. Sort all boxes by their center Y coordinate. Center Y is much more robust
       than top Y against tall ascenders or long descenders.
    2. Dynamically calculate the global median height to use as a threshold basis.
    3. Group boxes into lines: if a box's center Y is within `0.7 * median_h` of 
       the current line's median center Y, it belongs to that line.
    4. Sort left-to-right within each clustered line.
    """
    if not boxes:
        return []

    # 1. Calculate center Y for all boxes
    boxes_with_cy = [(b, b[1] + b[3] / 2.0) for b in boxes]
    
    # 2. Sort top-to-bottom by center Y
    boxes_with_cy.sort(key=lambda item: item[1])
    
    # 3. Calculate global median height to use as a dynamic threshold
    # This ensures the grouping adapts to the text size in the image
    median_h = np.median([b[3] for b in boxes])
    
    # Threshold for grouping: if center Y is within this distance of the line's median center Y
    # 0.7 * median_h provides a robust margin for grouping characters on the same line
    # while strictly preventing characters from different lines from mixing.
    threshold = max(10, median_h * 0.7)
    
    lines = []
    current_line = [boxes_with_cy[0][0]]
    current_cys = [boxes_with_cy[0][1]]
    
    for b, cy in boxes_with_cy[1:]:
        # Current line's representative center Y (median of its members)
        line_cy = np.median(current_cys)
        
        # If the box's center is close to the line's center, it belongs to this line
        if abs(cy - line_cy) <= threshold:
            current_line.append(b)
            current_cys.append(cy)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [b]
            current_cys = [cy]
            
    if current_line:
        lines.append(current_line)
        
    # 4. Within each line: sort left-to-right by x
    result = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        result.extend(line)
        
    return result