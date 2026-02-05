import cv2
import sys
import os
import numpy as np

# Ensure backend is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmentation import detect_characters, sort_boxes

def visualize_segmentation(image_path):
    print(f"Testing segmentation on: {image_path}")
    
    try:
        boxes, img = detect_characters(image_path)
        print(f"Detected {len(boxes)} characters.")
        
        sorted_boxes = sort_boxes(boxes)
        
        # Draw all boxes
        for i, (x, y, w, h) in enumerate(sorted_boxes):
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Put order number
            cv2.putText(img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        output_path = 'debug_segmentation.png'
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_segmentation.py <image_path>")
        sys.exit(1)
        
    visualize_segmentation(sys.argv[1])
