from pathlib import Path
from config import load_config
from logging_setup import setup_logging
from image_processing import load_image, preprocess_image, segment_text_regions, augment_image
from dataset import AugmentedDataset
from model import EnhancedModelClass
from ocr import multi_language_ocr
from report import save_annotated_image
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import easyocr
from reporting_utils import capture_stage, generate_html_report

def main():
    # Load configuration
    config = load_config("config.json")
    logger = setup_logging("process.log")
    
    image_path = Path("input/image.jpg")
    output_dir = Path("output")
    
    # Load and save the original image
    original_image = load_image(image_path)
    if original_image is None:
        logger.error("Failed to load image.")
        return
    original_image_path = capture_stage(original_image, "original", output_dir)
    
    # Preprocess the image
    processed_image = preprocess_image(original_image, config["blur_ksize"])
    processed_image_path = capture_stage(processed_image, "processed", output_dir)
    
    # Segment text regions
    regions = segment_text_regions(processed_image)
    if not regions:
        logger.error("No text regions found.")
        return
    
    # Augment the image and save examples
    augmented_images = augment_image(processed_image)
    augmented_image_paths = []
    for i, aug_image in enumerate(augmented_images):
        aug_path = capture_stage(aug_image, f"augmented_{i}", output_dir)
        augmented_image_paths.append(aug_path)
    
    # OCR and save the annotated result
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    ocr_results = multi_language_ocr(processed_image, regions, reader)
    final_image_path = output_dir / "final_output.jpg"
    save_annotated_image(original_image, ocr_results, final_image_path)
    
    # Placeholder data for training progress and performance metrics
    training_progress = [0.1, 0.08, 0.07, 0.05]  # Replace with actual training loss data
    validation_accuracy = [0.8, 0.85, 0.88, 0.9]  # Replace with actual validation accuracy data
    performance_metrics = {
        'Class A': (0.95, 0.92, 0.93),  # Replace with actual precision, recall, F1 data
        'Class B': (0.88, 0.85, 0.86)
    }
    
    # Generate HTML report
    generate_html_report(
        original_image_path,
        processed_image_path,
        augmented_image_paths,
        final_image_path,
        ocr_results,
        output_dir,
        training_progress,
        validation_accuracy,
        performance_metrics
    )

if __name__ == "__main__":
    main()
