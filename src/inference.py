from ultralytics import YOLO
import os 

def load_model(model):
    """
    Load the YOLOv8 model from the specified weights file.
    
    If the model weights file does not exist at the given path ("./weights/yolov8n.pt"),
    the function attempts to create the directory and initialize the model, 
    which may trigger downloading of the weights automatically by the YOLO class.
    
    Returns:
        YOLO: An instance of the loaded YOLOv8 model ready for inference.
    """
    model_file = "./weights/"
    model_path = os.path.join(model_file, model)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Downloading Model")
        model = YOLO(model_path)
    
    model = YOLO(model_path)

    return model

def process_model(model, filepath, imgsz, conf, device, classes):
    """
    Run inference on an input image using the provided YOLOv8 model.
    
    Args:
        model (YOLO): The loaded YOLOv8 model instance.
        filepath (str): Path to the input image file to run inference on.
        imgsz (int): The size to which the image should be resized for the model.
        conf (float): Confidence threshold for detection filtering.
        device (str): Device to run inference on, e.g., 'cpu' or 'cuda:0'.
        classes (int or list of int or None): Specific class indices to filter predictions,
            or None to detect all classes.
    
    Returns:
        Results: The prediction results object returned by the model's predict method,
        containing detected boxes, confidence scores, classes, and other metadata.
    """
    results = model.predict(filepath, imgsz=imgsz, conf=conf, device=device, classes=classes)
    return results

