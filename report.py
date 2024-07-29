import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def save_annotated_image(original_image, ocr_results, output_path):
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    for text, language, lang_confidence, text_confidence, bbox in ocr_results:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = bbox
        rect = patches.Polygon([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0 - 10, f"{text} ({language}, {text_confidence*100:.1f}%, {lang_confidence*100:.1f}%)", color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
