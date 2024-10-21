# Real-Time Object Detection using YOLOv7

This project uses YOLOv7 to capture real-time video feed and detect the presence of fire, guns/weapons, knives, or classify frames as normal. It is built for safety and security monitoring.

## Features
- **Real-time detection**: High-speed detection for fire, weapons, knives, or normal.
- **High accuracy**: 98% overall accuracy on a custom dataset.
- **Multi-class classification**: Detects multiple threat categories.

## Model Details
- **Model**: YOLOv7
- **Framework**: PyTorch
- **Input Size**: 128x128 pixels
- **Optimizer**: Adam
- **Accuracy**: 98%

[Image]

## Getting Started

### Prerequisites
1. Install Python and dependencies:
    ```bash
    pip install torch torchvision opencv-python
    ```
2. Clone the repo:
    ```bash
    git clone https://github.com/pojesh/Anomaly-Detection.git
    cd Anomaly-Detection
    ```

### Running the Model
1. Download the pre-trained weights and place them in the `weights/` directory.
2. Run detection:
    ```bash
    python detect.py --source 0 --weights weights/best.pt --conf-thres 0.5
    ```

## Example Output
Sample frames from testing:

- Fire detected:
  ![Fire Detection](./results/fire_example.png)

- Weapon detected:
  ![Weapon Detection](./results/gun_example.png)

## Future Work
- Integrate with alarm systems.
- Deploy on edge devices (Raspberry Pi).
