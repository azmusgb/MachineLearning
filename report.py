import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def save_annotated_image(original_image, ocr_results, output_path):
    """
    Save an annotated image with OCR results.

    Parameters:
    original_image (numpy array): The original image.
    ocr_results (list of tuples): List of OCR results, where each tuple contains:
        - text (str): The recognized text.
        - language (str): The language of the text.
        - lang_confidence (float): The confidence of the language detection.
        - text_confidence (float): The confidence of the text recognition.
        - bbox (list of tuples): The bounding box coordinates (x0, y0), (x1, y1), (x2, y2), (x3, y3).
    output_path (str): The path to save the annotated image.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    for result in ocr_results:
        text, language, lang_confidence, text_confidence, bbox = result
        polygon = patches.Polygon(bbox, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(polygon)

        annotation = f"{text} ({language}, {text_confidence*100:.1f}%, {lang_confidence*100:.1f}%)"
        ax.text(
            bbox[0][0],
            bbox[0][1] - 10,
            annotation,
            color="red",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
