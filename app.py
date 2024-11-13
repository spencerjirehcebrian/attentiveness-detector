import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

class YOLOAttentivenessDetector:
    def __init__(self, model_path='best.pt'):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.eval()
        self.model.conf = 0.5  # Confidence threshold
        self.model.iou = 0.45  # NMS IOU threshold
        
        # Run a warmup inference
        self.model(torch.zeros(1, 3, 640, 640))
        
    def predict(self, image):
        if image is None:
            return None, None
            
        # Convert to RGB if needed
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Run inference
        results = self.model(image)
        
        # Draw on image
        rendered_img = results.render()[0]  # Get annotated image
        
        # Get detection info
        pred_df = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame
        
        if len(pred_df) > 0:
            # Get the prediction with highest confidence
            best_pred = pred_df.iloc[pred_df['confidence'].argmax()]
            confidence = float(best_pred['confidence'])
            label = best_pred['name']
        else:
            confidence = 0.0
            label = "No detection"
            
        # Convert back to BGR for OpenCV
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        
        return rendered_img, {label: confidence}

# Create detector instance
try:
    print("Loading YOLOv5 model...")
    detector = YOLOAttentivenessDetector('best.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# YOLOv5 Attentiveness Detection")
    gr.Markdown("Real-time detection of attentiveness, unattentiveness, and phone usage")
    
    with gr.Row():
        with gr.Column():
            # Input webcam feed
            input_image = gr.Image(
                label="Webcam Input",
                type="numpy",
                sources=["webcam"],
            )
        
        with gr.Column():
            # Output detection image
            output_image = gr.Image(
                label="Detection Result",
                type="numpy"
            )
            # Output classification label
            output_label = gr.Label(
                label="Classification",
                num_top_classes=1
            )
    
    # Set up the webcam feed to continuously update
    input_image.stream(
        fn=detector.predict,
        inputs=input_image,
        outputs=[output_image, output_label],
        preprocess=True,
        postprocess=True,
        batch=False,
        max_batch_size=1
    )

if __name__ == "__main__":
    print("Starting Gradio server...")
    # For local development
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True if you want a public URL
    )