
## Vehicle Counter 🚗🚌🚛

A computer vision project that detects and counts vehicles in video footage using YOLO (You Only Look Once), OpenCV, and CVZone. This system can be applied in traffic monitoring, congestion analysis, and smart city solutions.
## 📌 Features

- Real-time vehicle detection and counting.

- Supports multiple vehicle types (cars, buses, trucks, bikes, etc.).

- Tracks vehicles across frames to prevent duplicate counts.

- Masked area support for region-specific counting.

- Easy to integrate with video feeds (CCTV, recorded clips).
## 🚀 How It Works

1. Load YOLOv8 pre-trained weights for object detection.

2. Process video frames using OpenCV.

3. Detect vehicles in each frame.

4. Use SORT tracker to maintain consistent IDs for vehicles.

5. Count vehicles only once when they cross a defined line/area.

6. Display results with CVZone overlays.
## ⚙️ Installation & Usage

1. Clone the repository:
    ```
    bash
    git clone https://github.com/your-username/Vehicle_Counter.git
    cd Vehicle_Counter
    ```

2. Install dependencies:
    ```
    bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```
    bash
    python vehicle_counter.py
    ```
## 📊 Use Cases

- Traffic monitoring systems.

- Intelligent Transportation Systems (ITS).

- Road planning & congestion management.

- Smart city surveillance.