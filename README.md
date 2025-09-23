# Vehicle Counter

A real-time **vehicle counting system** using **YOLOv8** for object detection and **OpenCV + CVZone** for tracking and counting vehicles across two lanes.

---

## 🚗 Project Overview

This project detects and counts vehicles moving in **two separate lanes** (incoming and outgoing) from a video stream. It leverages:

- **YOLOv8**: State-of-the-art object detection for vehicles.
- **OpenCV**: For video processing and image manipulation.
- **CVZone**: For easy visualization, including drawing bounding boxes and text on frames.
- **SORT Tracker**: To maintain consistent IDs for vehicles across frames.

The system identifies vehicles such as **cars, buses, motorcycles, and trucks**, and updates the count when a vehicle crosses a predefined line on each lane.

---

## 🛠 Features

- Real-time vehicle detection on video input.
- Separate counting for incoming and outgoing lanes.
- Highlighted counting lines (red by default, flashes green when crossed).
- Bounding boxes and vehicle IDs displayed on the video.
- Simple integration with video files.

  
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
