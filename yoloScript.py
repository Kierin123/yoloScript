import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
import cv2
import torch
import os

# Define paths
dataset_path = "datasets/data.yaml"
model_path = "trainedModel/best.pt"
image_path = "datasets/test/images/"
onnx_path = "restul.onnx"

def load_images_from_dir(dir):
    images = []
    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename))
        if img is not None:
            images.append(img)
    return images

# Train
def train():
    model = YOLO("yolo11n.pt")  # Load a pretrained YOLO model (recommended for training)
    results = model.train(data=dataset_path, epochs=200, imgsz=640, batch=64)
    results.save(model_path)
    print("Training completed and model saved.")

# Validate
def validate():
    model = YOLO(model_path)  # Load the trained model
    results = model.val(data=dataset_path)
    print(results)

# Predict
def predict():
    model = YOLO(model_path)  # Load the trained model
    img = load_images_from_dir(image_path)
    model.predict(img, show=True, save=True, imgsz=320, conf=0.3)

# Convert to ONNX
def convert_to_onnx():
    model = YOLO(model_path)  # Load the trained model
    dummy_input = torch.randn(1, 3, 640, 640)  # Adjust the input size as per your model requirements
    torch.onnx.export(model.model, dummy_input, onnx_path, opset_version=11,
                      input_names=["input"], output_names=["output"])
    print(f"Model has been converted to ONNX and saved at {onnx_path}")


# GUI setup
def create_gui():
    root = tk.Tk()
    root.title("YOLOv11 Control Menu")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    train_button = tk.Button(frame, text="Train", command=train)
    train_button.grid(row=0, column=0, padx=5, pady=5)

    validate_button = tk.Button(frame, text="Validate", command=validate)
    validate_button.grid(row=0, column=1, padx=5, pady=5)

    predict_button = tk.Button(frame, text="Predict", command=predict)
    predict_button.grid(row=1, column=0, padx=5, pady=5)

    convert_button = tk.Button(frame, text="Convert to ONNX", command=convert_to_onnx)
    convert_button.grid(row=1, column=1, padx=5, pady=5)

    exit_button = tk.Button(frame, text="Exit", command=root.quit)
    exit_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    create_gui()


# Main function to run specific tasks
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="YOLOv11 Training, Validation, and Prediction")
#     parser.add_argument("--task", type=str, choices=["train", "validate", "predict", "convert"], required=True, help="Task to perform")
#
#     args = parser.parse_args()
#
#     if args.task == "train":
#         train()
#     elif args.task == "validate":
#         validate()
#     elif args.task == "predict":
#         predict()
#     elif args.task == "convert":
#         convert_to_onnx()
#