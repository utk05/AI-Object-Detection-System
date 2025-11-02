# AI-Object-Detection-System
# 🖼️ YOLOv8 Real-Time Object Detection with Tkinter UI

This project integrates **YOLOv8 (Ultralytics)** with **OpenCV** and a **Tkinter-based GUI** to perform **real-time object detection** using a webcam.  
It provides a simple desktop interface where users can view live detections with bounding boxes directly inside a Python GUI window.

---

## ✨ Features
- 🎥 **Real-time object detection** using YOLOv8 and OpenCV  
- 🖼️ **Tkinter GUI** to display live webcam feed with bounding boxes  
- ⚡ Lightweight and fast (uses YOLOv8n by default)  
- 🛠️ Easy to extend with buttons, sliders, or custom UI elements  
- 🔧 Configurable confidence threshold and model size  

---

## Install dependencies:
pip install ultralytics opencv-python pillow

## ▶️ Usage
Run the script:
python app.py
-The Tkinter window will open and start the webcam feed.
-YOLOv8 will detect objects in real-time and draw bounding boxes.
-Close the window or press Ctrl+C in terminal to stop.

## 📂 Project Structure
├── app.py              # Main script with YOLO + Tkinter integration
├── requirements.txt    # Dependencies
├── README.md           # Project documentation

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

