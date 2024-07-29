from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict  # Ensure Dict is imported

def capture_stage(image: np.ndarray, stage_name: str, output_dir: Path) -> Path:
    """Save an image at a particular stage of the processing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stage_name}.jpg"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), image)
    return filepath

def generate_html_report(
    original_image_path: Path,
    processed_image_path: Path,
    augmented_image_paths: List[Path],
    final_image_path: Path,
    ocr_results: List[Tuple[str, str, float, float, List[Tuple[int, int]]]],
    output_dir: Path,
    training_progress: List[float],  # Example: [0.1, 0.2, 0.3, 0.4] for training loss over epochs
    validation_accuracy: List[float],  # Example: [0.8, 0.85, 0.88, 0.9] for accuracy over epochs
    performance_metrics: Dict[str, Tuple[float, float, float]]  # Example: {'Class A': (0.95, 0.92, 0.93)} for precision, recall, F1-score
) -> None:
    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OCR Processing Report</title>
            <link rel="stylesheet" href="styles.css">
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
        </head>
        <body>
            <div class="container">
                <h1>OCR Processing Report</h1>
                <div class="card">
                    <h2>Model Architecture</h2>
                    <p>The model consists of several convolutional layers followed by fully connected layers, using ReLU activations and dropout for regularization.</p>
                    <img src="{original_image_path.name}" alt="Original Image">
                </div>
                <div class="card">
                    <h2>Processed Image</h2>
                    <img src="{processed_image_path.name}" alt="Processed Image">
                </div>
                <div class="card carousel">
                    <h2>Augmented Images</h2>
                    <div class="carousel-container">
                        {"".join([f'<img class="carousel-image" src="{aug_path.name}" alt="Augmented Image {i + 1}">' for i, aug_path in enumerate(augmented_image_paths)])}
                    </div>
                    <div class="carousel-controls">
                        <button id="prev" onclick="prevImage()">&#10094;</button>
                        <button id="next" onclick="nextImage()">&#10095;</button>
                    </div>
                </div>
                <div class="card">
                    <h2>Final Annotated Image</h2>
                    <img src="{final_image_path.name}" alt="Final Annotated Image">
                </div>
                <div class="card">
                    <h2>Training Progress</h2>
                    <canvas id="trainingProgressChart"></canvas>
                </div>
                <div class="card">
                    <h2>Performance Metrics</h2>
                    <canvas id="performanceMetricsChart"></canvas>
                </div>
                <div class="card">
                    <h2>Detected Text</h2>
                    <ul>
        """)
        for text, language, lang_confidence, text_confidence, bbox in ocr_results:
            f.write(f"""
                        <li class="tooltip">
                            <p><strong>Text:</strong> {text}</p>
                            <p><strong>Language:</strong> {language} ({lang_confidence:.1f}%)</p>
                            <p><strong>Confidence:</strong> {text_confidence:.1f}%</p>
                            <p><strong>Bounding Box:</strong> {bbox}</p>
                            <span class="tooltiptext">Bounding Box Coordinates: {bbox}</span>
                        </li>
            """)
        f.write("""
                    </ul>
                </div>
            </div>
            <script>
                var ctxTraining = document.getElementById('trainingProgressChart').getContext('2d');
                var trainingProgressChart = new Chart(ctxTraining, {
                    type: 'line',
                    data: {
                        labels: [...Array(training_progress.length).keys()], // Assuming epochs as labels
                        datasets: [{
                            label: 'Training Loss',
                            data: training_progress,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            fill: false
                        },
                        {
                            label: 'Validation Accuracy',
                            data: validation_accuracy,
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 2,
                            fill: false
                        }]
                    },
                    options: {
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Epochs'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Value'
                                }
                            }
                        }
                    }
                });

                var ctxPerformance = document.getElementById('performanceMetricsChart').getContext('2d');
                var performanceMetricsChart = new Chart(ctxPerformance, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(performance_metrics),
                        datasets: [{
                            label: 'Precision',
                            data: Object.values(performance_metrics).map(m => m[0]),
                            backgroundColor: 'rgba(75, 192, 192, 0.5)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }, {
                            label: 'Recall',
                            data: Object.values(performance_metrics).map(m => m[1]),
                            backgroundColor: 'rgba(153, 102, 255, 0.5)',
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 1
                        }, {
                            label: 'F1 Score',
                            data: Object.values(performance_metrics).map(m => m[2]),
                            backgroundColor: 'rgba(255, 159, 64, 0.5)',
                            borderColor: 'rgba(255, 159, 64, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Classes'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Scores'
                                }
                            }
                        }
                    }
                });

                var currentIndex = 0;
                showImage(currentIndex);

                function showImage(index) {
                    var images = document.querySelectorAll('.carousel-image');
                    images.forEach((img, i) => {
                        img.classList.remove('active');
                        if (i === index) {
                            img.classList.add('active');
                        }
                    });
                }

                function prevImage() {
                    currentIndex = (currentIndex === 0) ? document.querySelectorAll('.carousel-image').length - 1 : currentIndex - 1;
                    showImage(currentIndex);
                }

                function nextImage() {
                    currentIndex = (currentIndex === document.querySelectorAll('.carousel-image').length - 1) ? 0 : currentIndex + 1;
                    showImage(currentIndex);
                }
            </script>
        </body>
        </html>
        """)
    print(f"HTML report generated at {report_path}")
