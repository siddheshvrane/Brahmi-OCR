Brahmi ResNet50 Deployment Package
=================================

This folder contains all the essential files needed to run inference with the trained Brahmi character recognition model.

FILES:
- resnet50_brahmi.pth : The best trained weights from the ResNet50 training.
- model.py            : The PyTorch model architecture definition.
- class_names.json     : The mapping of class indices to character names (214 classes).
- inference.py        : A clean script to perform inference on single images.
- requirements.txt    : Minimal Python dependencies for inference.

USAGE:
1. Install dependencies:
   pip install -r requirements.txt

2. Run inference:
   python inference.py path/to/your/image.png

INTEGRATION:
To use this in your main project, import the `BrahmiClassifier` class from `inference.py`.

Example:
```python
from inference import BrahmiClassifier

classifier = BrahmiClassifier(
    model_path="resnet50_brahmi.pth",
    class_names_path="class_names.json"
)
results = classifier.predict("image.png")
print("Top prediction:", results[0]['character'])
```
