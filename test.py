import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import easyocr
import logging
from langdetect import detect
from imgaug import augmenters as iaa
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Dict, Any, Optional
from torch.amp import GradScaler, autocast

# Set up logging configuration
log_filename = "process.log"
if os.path.exists(log_filename):
    os.remove(log_filename)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define project directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "ocr_report.html")

# Create output and model directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Detect if running in Replit
is_replit = os.getenv('REPL_ID') is not None or os.getenv('REPL_SLUG') is not None
logger.info(f"Running in Replit: {is_replit}")

# Initialize EasyOCR reader with GPU support if not in Replit
use_gpu = not is_replit and torch.cuda.is_available()
logger.info(f"Initializing EasyOCR reader with GPU support: {use_gpu}.")
reader = easyocr.Reader(["en"], gpu=use_gpu)

class EnhancedModelClass(nn.Module):
    def __init__(self, input_size=(64, 64)):
        super(EnhancedModelClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop = nn.Dropout(0.5)

        self._to_linear = None
        self._initialize_to_linear(input_size)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # Assuming 10 classes

        logger.info("EnhancedModelClass initialized with input size: %s", input_size)

    def _initialize_to_linear(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            output = self.convs(dummy_input)
            self._to_linear = output.view(1, -1).size(1)
            logger.info(f"Calculated _to_linear: {self._to_linear}")

    def convs(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        logger.debug(f"Shape after conv1: {x.shape}")
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        logger.debug(f"Shape after conv2: {x.shape}")
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        logger.debug(f"Shape after conv3: {x.shape}")
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        logger.debug(f"Shape after conv4: {x.shape}")
        return x

    def forward(self, x):
        x = self.convs(x)
        logger.debug(f"Shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)
        logger.debug(f"Shape after flattening: {x.shape}")
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class AugmentedDataset(Dataset):
    def __init__(self, original_image: np.ndarray, transform: Optional[iaa.Sequential] = None, num_augments: int = 100):
        self.original_image = original_image
        self.transform = transform
        self.num_augments = num_augments
        logger.info("AugmentedDataset initialized with %d augmentations", num_augments)

    def __len__(self) -> int:
        return self.num_augments

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.original_image.copy()
        if self.transform:
            image = self.transform(images=[image])[0]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        label = idx % 10
        logger.debug(f"Generated augmented image with label: {label}")
        return torch.tensor(image), torch.tensor(label)

augmentation_seq = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)),
    iaa.ScaleX((0.8, 1.2)),
    iaa.ScaleY((0.8, 1.2)),
    iaa.AdditiveGaussianNoise(scale=(10, 30)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.ShearX((-10, 10)),
    iaa.ShearY((-10, 10)),
    iaa.contrast.LinearContrast((0.75, 1.5)),
])
logger.info("Augmentation sequence created.")

def load_image(image_path: str) -> Optional[np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Unable to load image at {image_path}")
        return None
    logger.info(f"Loaded image: {image_path}")
    return image

def augment_image(image: np.ndarray) -> List[np.ndarray]:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    images_aug = augmentation_seq(images=[image])
    logger.info("Generated augmented images.")
    return images_aug

def generate_synthetic_image(text: str, font_path: str, image_size: Tuple[int, int]) -> np.ndarray:
    image = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))
    synthetic_image = np.array(image)
    save_image(synthetic_image, os.path.join(OUTPUT_DIR, "synthetic_image.jpg"))
    logger.info(f"Generated synthetic image with text: {text}")
    return synthetic_image

def save_image(img: np.ndarray, filename: str) -> None:
    cv2.imwrite(filename, img)
    logger.info(f"Saved image: {filename}")

def preprocess_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int, int, int]]]]:
    try:
        logger.info(f"Starting preprocessing for image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image '{image_path}'.")

        logger.info(f"Loaded image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        segmented_regions = segment_text_regions(img)
        for x, y, w, h in segmented_regions:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        save_image(img, os.path.join(OUTPUT_DIR, "segmented_regions.jpg"))

        return img, segmented_regions

    except Exception as e:
        logger.error(f"Error preprocessing image '{image_path}': {e}")
        return None, None

def segment_text_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    logger.info(f"Segmented {len(regions)} text regions")
    return regions

def detect_language(text: str) -> Tuple[str, float]:
    try:
        language = detect(text)
        logger.info(f"Detected language: {language}")
        return language, 1.0
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "unknown", 0

def multi_language_ocr(image: np.ndarray, regions: List[Tuple[int, int, int, int]], min_confidence: float = 0.5) -> List[Tuple[str, str, float, float, List[Tuple[int, int]]]]:
    results = []
    min_region_size = 10
    model_path = os.path.join(MODEL_DIR, "model.pth")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model = EnhancedModelClass()
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully.")
    except RuntimeError as e:
        logger.error(f"Error loading model: {e}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return results

    model.eval()

    for x, y, w, h in regions:
        if w < min_region_size or h < min_region_size:
            continue
        region = image[y : y + h, x : x + w]
        logger.info(f"Processing region: {(x, y, w, h)}")
        detected_text = reader.readtext(region, detail=1)
        for text_info in detected_text:
            bbox, text, confidence = text_info
            if confidence > min_confidence:
                language, lang_confidence = detect_language(text)
                results.append((text, language, lang_confidence, confidence, bbox))
                logger.info(
                    f"Detected text: '{text}' in language: '{language}' with confidence: {confidence}"
                )
    return results

def save_annotated_image(original_image: np.ndarray, ocr_results: List[Tuple[str, str, float, float, List[Tuple[int, int]]]], output_path: str) -> None:
    logger.info(f"Saving annotated image to {output_path}.")
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    for text, language, lang_confidence, text_confidence, bbox in ocr_results:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = bbox
        rect = patches.Polygon(
            [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 10,
            f"{text} ({language}, {text_confidence*100:.1f}%, {lang_confidence*100:.1f}%)",
            color="red",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
        )
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Annotated image saved at {output_path}")

def generate_html_report(image_path: str, ocr_results: List[Tuple[str, str, float, float, List[Tuple[int, int]]]], annotated_image_path: str) -> None:
    logger.info(f"Generating HTML report for {image_path}.")
    report_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 20px;
            }}
            h1, h2 {{
                color: #4CAF50;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 5px;
            }}
            h2 {{
                margin-top: 20px;
            }}
            ul {{
                list-style: none;
                padding-left: 0;
            }}
            li {{
                margin-bottom: 15px;
                background-color: #f9f9f9;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .detected-text {{
                font-weight: bold;
            }}
            .language, .confidence, .bounding-box {{
                font-style: italic;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <h1>OCR Report for {image_path}</h1>
        <h2>Annotated Image</h2>
        <img src="{annotated_image_path}" alt="Annotated Image">
        <h2>Detected Text</h2>
        <ul>
    """

    for text, language, lang_confidence, text_confidence, bbox in ocr_results:
        report_content += f"""
        <li>
            <span class="detected-text">{text}</span><br>
            <span class="language">Language: {language} ({lang_confidence*100:.1f}%)</span><br>
            <span class="confidence">Confidence: {text_confidence*100:.1f}%</span><br>
            <span class="bounding-box">Bounding Box: {bbox}</span>
        </li>
        """

    report_content += """
        </ul>
    </body>
    </html>
    """

    with open(HTML_REPORT_PATH, "w") as f:
        f.write(report_content)
    logger.info(f"HTML report generated at {HTML_REPORT_PATH}")

def generate_report(original_image_path: str) -> None:
    logger.info(f"Generating report for {original_image_path}")
    enhanced, original_image = preprocess_image(original_image_path)
    if enhanced is None or original_image is None:
        logger.error("Failed to process image, aborting report generation.")
        return

    ocr_results = multi_language_ocr(enhanced, enhanced)
    if not ocr_results:
        logger.info("No text found in the image.")
        return

    annotated_image_path = os.path.join(OUTPUT_DIR, "annotated_output.jpg")
    save_annotated_image(original_image, ocr_results, annotated_image_path)

    generate_html_report(original_image_path, ocr_results, annotated_image_path)

scaler = GradScaler('cuda')

def train_model(
    model: nn.Module, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    train_loader: DataLoader, 
    num_epochs: int = 10
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

        scheduler.step(epoch_loss)

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            model_save_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_save_path, _use_new_zipfile_serialization=False)
            logger.info(f"Best model saved to {model_save_path} with accuracy {best_accuracy:.4f}")

    logger.info("Training complete.")

def main() -> None:
    image_path = os.path.join(INPUT_DIR, "image.jpg")

    original_image = load_image(image_path)
    if original_image is None:
        return

    if original_image.ndim == 2:
        original_image = np.expand_dims(original_image, axis=0)
    elif original_image.ndim == 3 and original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        original_image = np.expand_dims(original_image, axis=0)

    dataset = AugmentedDataset(
        original_image, transform=augmentation_seq, num_augments=500
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Reduced batch size

    model = EnhancedModelClass(input_size=(64, 64))  # Adjust input_size as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Clear cache before starting training to free up memory
    torch.cuda.empty_cache()

    train_model(model, criterion, optimizer, train_loader, num_epochs=10)

    logger.info("Processing original image for OCR and report generation.")
    generate_report(image_path)

if __name__ == "__main__":
    main()
