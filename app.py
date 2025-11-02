from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Load YOLO model
detect_objects = False  # State variable to track detection

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        global detect_objects
        if detect_objects:  # Perform detection only if the button is clicked
            results = model(frame)  # Run YOLO detection
            frame_with_boxes = results[0].plot()  # Processed frame with detections
        else:
            frame_with_boxes = frame  # Show plain video feed if detection is off

        ret, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detect_objects
    detect_objects = True
    return "Detection started", 200

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detect_objects
    detect_objects = False
    return "Detection stopped", 200

if __name__ == "__main__":
    app.run(debug=True)
