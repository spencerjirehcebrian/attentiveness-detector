# Real-time Attentiveness Detection

This project uses YOLOv5 and Gradio to create a real-time attentiveness detection system using your webcam. The system can detect if a person is attentive, unattentive, or using a phone.

## Quick Start 🚀

### Prerequisites

Before you begin, make sure you have:
- A computer with a webcam
- Python installed (Python 3.8 or higher recommended)
- Git installed (for downloading YOLOv5)

### Step-by-Step Installation Guide 📝

1. **Create a new project folder**
   ```bash
   mkdir attentiveness-detection
   cd attentiveness-detection
   ```

2. **Set up a Python virtual environment**
   
   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   On Linux/Mac:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install gradio==4.19.2
   pip install torch torchvision
   pip install opencv-python
   pip install pandas
   pip install pillow
   ```

4. **Project Structure**
   
   Place your files in this structure:
   ```
   attentiveness-detection/
   ├── app.py           # Main application file
   ├── best.pt          # Your YOLOv5 model file
   ├── venv/            # Virtual environment
   └── README.md        # This file
   ```

5. **Create the application file**
   
   Create a new file called `app.py` and copy the code from above into it.

6. **Place your model**
   
   Put your `best.pt` file in the project root directory.

### Running the Application 🎯

1. **Activate the virtual environment** (if not already activated)
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Access the interface**
   - Open your web browser
   - Go to `http://localhost:7860`
   - Allow webcam access when prompted

### Troubleshooting Common Issues 🔧

1. **"No module found" error**
   ```bash
   pip install [module_name]
   ```

2. **Webcam not detected**
   - Make sure your webcam is connected
   - Allow browser access to webcam
   - Try restarting the application

3. **CUDA/GPU errors**
   - The application will run on CPU by default
   - For GPU support, install CUDA toolkit and corresponding PyTorch version

4. **Model loading error**
   - Ensure `best.pt` is in the correct location
   - Check if the model path in `app.py` matches your file location

### Customization 🎨

You can modify these parameters in the code:

```python
# In YOLOAttentivenessDetector class
self.model.conf = 0.5  # Confidence threshold (0.0 to 1.0)
self.model.iou = 0.45  # IOU threshold for NMS
```

### Features ✨

- Real-time webcam detection
- Class confidence display
- Bounding box visualization
- Easy-to-use web interface

## Support and Updates 💡

For updates or issues:
1. Check if your packages are up to date
2. Ensure your webcam is working properly
3. Try restarting the application

## Requirements

```txt
gradio==4.19.2
torch
torchvision
opencv-python
pandas
pillow
```

## Optional: Running on GPU 🚀

If you have a NVIDIA GPU and want to use it:

1. Install CUDA toolkit from NVIDIA website
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace cu118 with your CUDA version)

## Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## Safety Notes ⚠️

- The application requires webcam access
- All processing is done locally on your machine
- No data is sent to external servers

## License

This project is open-source and available under the MIT License.