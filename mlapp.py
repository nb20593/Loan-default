import os
import onnx
import onnxruntime
import time
import numpy as np
import cv2
import mlflow

# Set your MLflow tracking URI if it's not already configured
# mlflow.set_tracking_uri("your_tracking_uri")

cwd = os.getcwd()

ROOT_DIR = os.path.join(cwd)
image_path = []
folder = "test_images"
for file in os.listdir(os.path.join(ROOT_DIR, folder)):
    full_name = os.path.join(ROOT_DIR, folder, file)
    image_path.append(full_name)

# Load the ONNX model.
model = onnx.load('model_recruit.onnx')

# Serialize the ONNX model to a file.
onnx.save(model, 'my_model.onnx.pb')

#model_path = os.path.join(cwd, "model", "model_recruit.onnx")

sess = onnxruntime.InferenceSession('my_model.onnx.pb')

threshold = 0.7  # Prediction threshold
highlight_color = (0, 0, 255, 100)  # Light gray with alpha channel (0-255 for transparency)
border_color = (0, 0, 255)  # Red color for the border
border_thickness = 2  # Border thickness

# Start a new MLflow run to log the results
with mlflow.start_run():
    for image_path in image_path:
        # Load and preprocess the input image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (500, 500))
        img = img.reshape(1, 1, 500, 500).astype(np.float32) / 255.0

        # Make predictions and measure inference time
        start_time = time.time()
        predictions = sess.run(None, {"modelInput": img})
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time} seconds")

        # Log the inference time as a metric
        mlflow.log_metric("inference_time", inference_time)

        # Get the prediction grid
        prediction_grid = predictions[0][0, :, :, 0]

        # Create a copy of the original image for visualization
        result_image = cv2.imread(image_path)

        # Iterate through the grid and apply a transparent gray shade to cells with predictions > threshold
        for h in range(40):
            for w in range(20):
                prediction = prediction_grid[h, w]
                if prediction > threshold:
                    # Calculate cell coordinates on the original image
                    x1 = w * (result_image.shape[1] // 20)
                    x2 = (w + 1) * (result_image.shape[1] // 20)
                    y1 = h * (result_image.shape[0] // 40)
                    y2 = (h + 1) * (result_image.shape[0] // 40)

                    # Draw a transparent gray rectangle
                    overlay = result_image.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_color, -1)  # -1 for a filled rectangle
                    cv2.addWeighted(overlay, 0.5, result_image, 0.5, 0, result_image)  # Adjust alpha to control transparency

        # Iterate through the grid again to add a thick border to the highlighted cells
        for h in range(40):
            for w in range(20):
                prediction = prediction_grid[h, w]
                if prediction > threshold:
                    # Calculate cell coordinates on the original image
                    x1 = w * (result_image.shape[1] // 20)
                    x2 = (w + 1) * (result_image.shape[1] // 20)
                    y1 = h * (result_image.shape[0] // 40)
                    y2 = (h + 1) * (result_image.shape[0] // 40)

                    # Draw a thick border
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), border_color, border_thickness)

        if image_path.lower().endswith(".png"):
            output_extension = "_output.png"
        elif image_path.lower().endswith(".jpg"):
            output_extension = "_output.jpg"
        else:
            output_extension = "_output"  # Default to this if the extension is not recognized

        # Create the output path by replacing the input extension
        output_path = image_path.rsplit('.', 1)[0] + output_extension

        # Save the result image with the dynamically determined output path
        cv2.imwrite(output_path, result_image)
        mlflow.log_artifact(output_path)
        
        # Log the input image as an artifact
        mlflow.log_artifact(image_path)

        # Display the output in Colab (for visualization)
        cv2.imshow("Predictions", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

