from src.inference import load_model, process_model

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

import torch
import os

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Init
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
 
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = 'cpu'

models = {
    0: "yolov8n.pt",
    1: "yolov8s.pt",
    2: "yolov8m.pt",
    3: "yolov8l.pt",
    4: "yolov8x.pt"
}

model = load_model(models[0])
labels = model.names

# Check EXTENSIONS
def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file has an extension and it is one of the allowed types ('png', 'jpg', 'jpeg'), False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload image file, Process model, Save results
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle image upload, run inference with YOLOv8 model, and display the results.

    For POST requests:
        - Verify if a file is included in the request.
        - Retrieve parameters from the form:
            * model_input (int or None): Select which YOLOv8 model to use (default is 0).
            * confidence_input (float): Confidence threshold for detections (default 0.5).
            * imgsz_input (int): Input image size for the model (default 640).
            * label_input (int or None): Specific class label to detect, if provided.
        - Validate the filename and ensure it has an allowed extension.
        - Save the uploaded file to the upload folder.
        - Perform YOLOv8 model inference with the specified parameters.
        - Save the processed result image to the results folder.
        - Extract detection details including:
            * cls: predicted object classes.
            * conf: confidence scores for each detected object.
            * orig_shape: original dimensions of the uploaded image.
            * preprocess_time, inference_time, postprocess_time: timing metrics for each processing stage.
            * count_obj: total number of detected objects.
        - Render the 'index.html' template displaying the uploaded image, result image, detection info, confidence scores, and labels.

    For GET requests:
        - Render the 'index.html' template with the upload form and label list, without detection results.

    Returns:
        Rendered HTML page showing the upload interface and, if applicable, detection results.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        # Get parameters
        model_input = int(request.form.get('model_input', None)) if request.form.get('model_input') else None
        confidence_input = float(request.form.get('confidence_input', 0.5))
        imgsz_input = int(request.form.get('imgsz_input', 640))
        label_input = int(request.form.get('label_input', None)) if request.form.get('label_input') else None
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if model_input in models:
            model = load_model(models[model_input])
        else:
            model = load_model(models[0])

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # YOLOv8 processing
            results = process_model(model=model, 
                                    filepath=filepath, 
                                    imgsz=imgsz_input, 
                                    conf=confidence_input, 
                                    device=device, 
                                    classes=label_input)   

            # Save results
            result_filename = f"processed_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            for result in results:
                cls = result.boxes.cls
                conf = result.boxes.conf.tolist()
                orig_shape = result.orig_shape
                preprocess_time = result.speed['preprocess']
                inference_time = result.speed['inference']
                postprocess_time = result.speed['postprocess']

                result.save(filename=result_path) 
            count_obj = cls.numel() 
                
            return render_template('index.html', filename=filename, result_filename=result_filename, 
                                   orig_shape=orig_shape,
                                   preprocess_time=preprocess_time,
                                   inference_time=inference_time,
                                   postprocess_time=postprocess_time,
                                   count_obj=count_obj,
                                   conf=conf,
                                   labels=labels,
                                   model_input=model_input,
                                   models=models)
        
    return render_template('index.html', filename=None,  labels=labels, model_input=None, models=models)

# Download input to static/uploads
@app.route('/uploads/<filename>')
def download_file(filename):
    """
    Serve the original uploaded image file from the upload directory.

    Args:
        filename (str): The filename of the uploaded image to serve.

    Returns:
        A response to send the requested file from the UPLOAD_FOLDER directory.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Save output to results
@app.route('/results/<filename>')
def result_file(filename):
    """
    Serve the processed result image file from the results directory.

    Args:
        filename (str): The filename of the processed image to serve.

    Returns:
        A response to send the requested file from the RESULT_FOLDER directory.
    """
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888, use_reloader=True)


