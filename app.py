import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template
import time
from threading import Thread

app = Flask(__name__)

# Configuration
CAMERA_INDEX = 0
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

class_names = ['attentive', 'unattentive', 'phone']

# Initialize ONNX Runtime
session = ort.InferenceSession("yolov5_best.onnx", providers=['CPUExecutionProvider'])

def preprocess_image(image):
    """Prepare image for inference"""
    original_height, original_width = image.shape[:2]
    
    # Resize and normalize
    input_img = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
    
    return input_img, original_height, original_width

def post_process(output, orig_h, orig_w):
    """Process YOLO output"""
    boxes = []
    class_ids = []
    confidences = []
    
    # Reshape output to [num_boxes, classes+5]
    output = output[0].reshape((-1, len(class_names) + 5))
    
    # Process each detection
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > CONFIDENCE_THRESHOLD:
            # Extract bounding box coordinates
            x, y, w, h = detection[0:4]
            
            # Convert to corner coordinates
            left = int((x - w/2) * orig_w)
            top = int((y - h/2) * orig_h)
            width = int(w * orig_w)
            height = int(h * orig_h)
            
            boxes.append([left, top, width, height])
            class_ids.append(class_id)
            confidences.append(float(confidence))
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    
    results = []
    for i in indices:
        results.append({
            'box': boxes[i],
            'class_id': class_ids[i],
            'confidence': confidences[i]
        })
    
    return results

def draw_detections(image, results):
    """Draw detection results on image"""
    for result in results:
        box = result['box']
        class_id = result['class_id']
        confidence = result['confidence']
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for attentive, customize colors as needed
        if class_names[class_id] == 'unattentive':
            color = (0, 165, 255)  # Orange
        elif class_names[class_id] == 'phone':
            color = (0, 0, 255)    # Red
            
        cv2.rectangle(image, (box[0], box[1]), 
                     (box[0] + box[2], box[1] + box[3]), 
                     color, 2)
        
        # Draw label
        label = f'{class_names[class_id]}: {confidence:.2f}'
        cv2.putText(image, label, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def generate_frames():
    """Generator function for streaming frames"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prepare image and run inference
        input_img, orig_h, orig_w = preprocess_image(frame)
        outputs = session.run(None, {'images': input_img})[0]
        
        # Process detections
        results = post_process(outputs, orig_h, orig_w)
        
        # Draw results
        frame = draw_detections(frame, results)
        
        # Convert frame to jpg for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)