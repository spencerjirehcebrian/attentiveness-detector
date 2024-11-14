import cv2
import numpy as np
import onnxruntime
from flask import Flask, Response, render_template
import threading
from queue import Queue
import time

app = Flask(__name__)

# Global variables
input_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_thread = False

# Initialize ONNX Runtime
session = onnxruntime.InferenceSession("yolov5_best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Classes
CLASSES = ['attentive', 'unattentive', 'phone']

def preprocess_image(frame):
    # Resize and normalize image
    img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32) / 255.0
    return img

def post_process(output, conf_threshold=0.25, iou_threshold=0.45):
    # Process YOLO output
    # Assuming standard YOLOv5 output format
    predictions = output[0]
    
    boxes = []
    scores = []
    class_ids = []
    
    for pred in predictions:
        if pred[4] > conf_threshold:  # Confidence score
            class_id = np.argmax(pred[5:])
            score = pred[4] * pred[5 + class_id]
            
            if score > conf_threshold:
                x, y, w, h = pred[0:4]
                boxes.append([x, y, w, h])
                scores.append(float(score))
                class_ids.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]

def inference_thread():
    global stop_thread
    
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Preprocess
        input_data = preprocess_image(frame)
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Post-process
        boxes, scores, class_ids = post_process(outputs[0])
        
        # Draw results
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            label = f"{CLASSES[class_id]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Update output queue
        if not output_queue.full():
            output_queue.put(frame_bytes)
            
    cap.release()

def gen_frames():
    while True:
        if not output_queue.empty():
            frame_bytes = output_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>YOLOv5 Inference</title>
        </head>
        <body>
            <h1>YOLOv5 Live Inference</h1>
            <img src="/video_feed">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start inference thread
    inference_thread = threading.Thread(target=inference_thread)
    inference_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
    # Cleanup
    stop_thread = True
    inference_thread.join()