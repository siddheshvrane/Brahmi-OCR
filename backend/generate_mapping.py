import json
import os

def roman_to_devanagari_logic(label):
    """Dynamically converts Romanized Brahmi labels to Devanagari."""
    if not label or label == "Unknown" or label == "<UNK>":
        return label
        
    # Independent vowels
    vowel_independent = {
        "a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ", "e": "ए", "o": "ओ", 
        "aa": "आ", "ii": "ई", "uu": "ऊ", "baa": "बा"
    }
    
    # Consonant bases
    consonant_bases = {
        "bh": "भ", "b": "ब", "ch": "छ", "c": "च", "dh": "ध", "d": "द", "gh": "घ", "g": "ग",
        "jh": "झ", "j": "ज", "kh": "ख", "k": "क", "ph": "फ", "p": "प", "th": "थ", "t": "त",
        "ḍh": "ढ", "ḍ": "ड", "ṇa": "ण", "ṇ": "ण", "ṣ": "ष", "ṭh": "ठ", "ṭ": "ट", "h": "ह", "l": "ल",
        "m": "म", "n": "न", "r": "र", "s": "स", "v": "व", "y": "य", "ñ": "ञ", "ś": "श"
    }

    # Vowel suffixes (matras)
    vowel_suffixes = {
        "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू", "e": "े", "o": "ो", "a": "",
        "aa": "ा", "ii": "ि", "uu": "ु", # Fallbacks for common alternative spellings
        "aii": "ी", "auu": "ू"
    }

    if label in vowel_independent:
        return vowel_independent[label]
    
    # Sort bases by length desc to match 'bh' before 'b'
    for base, dev_base in sorted(consonant_bases.items(), key=lambda x: len(x[0]), reverse=True):
        if label.startswith(base):
            suffix = label[len(base):]
            if not suffix:
                return dev_base
            elif suffix in vowel_suffixes:
                return dev_base + vowel_suffixes[suffix]
            # Handle cases like 'bhaa' or 'bhiii'
            elif suffix == 'aa' or suffix == 'ā':
                return dev_base + "ा"
            elif suffix == 'ii' or suffix == 'ī':
                return dev_base + "ी"
            elif suffix == 'uu' or suffix == 'ū':
                return dev_base + "ू"
    
    return label

def generate_full_mapping():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATHS = {
        'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50_new', 'class_names.json'),
        'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0_new', 'model_config_efficientnetb0_new.json'),
        'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v2', 'brahmi_ocr_best_config.json')
    }

    all_labels = set()
    
    for model_name, config_path in CONFIG_PATHS.items():
        if not os.path.exists(config_path):
            print(f"Warning: Config not found for {model_name} at {config_path}")
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            
            # Extract from class_names
            if 'class_names' in cfg:
                all_labels.update(cfg['class_names'])
            
            # Extract from idx2label
            if 'idx2label' in cfg:
                all_labels.update(cfg['idx2label'].values())
            
            # Extract from idx2char, id2char
            for key in ['idx2char', 'id2char']:
                if key in cfg:
                    all_labels.update(cfg[key].values())

    # Build mapping
    translit_map = {}
    for label in sorted(list(all_labels)):
        translit_map[label] = roman_to_devanagari_logic(label)
    
    mapping_path = os.path.join(BASE_DIR, 'transliteration_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(translit_map, f, ensure_ascii=False, indent=2)
    
    print(f"Generated mapping for {len(translit_map)} unique labels at {mapping_path}")

if __name__ == "__main__":
    generate_full_mapping()
