import numpy as np
import easyocr
from langdetect import detect
from typing import List, Tuple

def detect_language(text: str) -> Tuple[str, float]:
    try:
        language = detect(text)
        return language, 1.0
    except Exception as e:
        return "unknown", 0

def multi_language_ocr(image: np.ndarray, regions: List[Tuple[int, int, int, int]], reader, min_confidence: float = 0.5):
    results = []
    for x, y, w, h in regions:
        region = image[y:y+h, x:x+w]
        detected_text = reader.readtext(region, detail=1)
        for bbox, text, confidence in detected_text:
            if confidence > min_confidence:
                language, lang_confidence = detect_language(text)
                results.append((text, language, lang_confidence, confidence, bbox))
    return results
