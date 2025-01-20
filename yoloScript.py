from ultralytics import YOLO
import cv2
import torch
import os

# Define paths
dataset_path = "./datasets/data.yaml"
trained_model_path = "runs/"
model_path = "trained/best.pt"
image_path = "datasets/test/images/"
onnx_path = "result.onnx"

import os
import cv2


def load_images_from_dir(dir):
    images = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if not os.path.isdir(dir):
        return images

    for filename in os.listdir(dir):
        if filename.lower().endswith(supported_extensions):
            img = cv2.imread(os.path.join(dir, filename))
            if img is not None:
                images.append(img)
            else:
                print(f"Could not read the image {filename}.")
        else:
            print(f"Unsupported file format: {filename}")

    return images

"""
train()

Function to perform training of pretrained model of yolo11. 
Function uses yolo11s.pt model - model to get better result but loose 
time performance. 
epochs = 250
optimizer = auto / AdamW
batch = 32
"""
def train():
    try:
        model = YOLO("yolo11s.pt")  # Load a pretrained YOLO model (recommended for training)
        model.info()
        results = model.train(data=dataset_path, epochs=250, imgsz=640, batch=32, patience=50,  device='cpu')
        results.save(trained_model_path)
        print("Training completed and model saved.")
    except Exception as e:
        print(f"Error during training: {e}")

"""
validate()

Function to perform validation of trained model of yolo11. 
"""
def validate():
    try:
        model = YOLO(model_path)  # Load the trained model
        results = model.val(data=dataset_path)
        print(results)
    except Exception as e:
        print(f"Error during validation: {e}")


"""
predict()

Function to perform prediction signs from provides photos in "datasets/test/images/"
"""
def predict():
    try:
        model = YOLO(model_path)  # Load the trained model
        img = load_images_from_dir(image_path)
        print(len(img))
        model.predict(img, save=True, conf=0.75)
    except Exception as e:
        print(f"Error during prediction: {e}")

"""
convert_to_onnx()

Function to create result.onnx file.
"""
def convert_to_onnx():
    try:
        model = YOLO(model_path)  # Load the trained model
        dummy_input = torch.randn(1, 3, 640, 640)  # Adjust the input size as per your model requirements
        torch.onnx.export(model.model, dummy_input, onnx_path, opset_version=11,
                          input_names=["input"], output_names=["output"])
        # model.export(format="onnx")
        print(f"Model has been converted to ONNX and saved at {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")


# Main function with a control menu
def main():
    print('Current location: ' + os.getcwd())
    while True:
        print("\nYOLOv11 Control Menu:")
        print("1. Train")
        print("2. Validate")
        print("3. Predict")
        print("4. Convert to ONNX")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            train()
        elif choice == "2":
            validate()
        elif choice == "3":
            predict()
        elif choice == "4":
            convert_to_onnx()
        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()