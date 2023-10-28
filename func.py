import os
import onnx
import onnxruntime
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
from onnxruntime.quantization import quantize_dynamic , QuantType
from onnxruntime.quantization import quantize, QuantizationMode, quantize_dynamic, quantize_static, QuantType
from openvino.inference_engine import IECore

# Function to load image paths
def load_image_paths(folder):
    cwd = os.getcwd()
    ROOT_DIR = os.path.join(cwd)
    image_paths = []
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        full_name = os.path.join(ROOT_DIR, folder, file)
        image_paths.append(full_name)
    return image_paths

# Function to load ONNX model
def load_onnx_model(model_path,options=None):
    if options is not None:
        return onnxruntime.InferenceSession(model_path, options=options)
    else:
        return onnxruntime.InferenceSession(model_path)

# Function to preprocess an image
def preprocess_image(image_path, input_shape):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, input_shape)
    img = img.reshape(1, 1, input_shape[0], input_shape[1]).astype(np.float32) / 255.0
    return img

# Function to make predictions
def make_predictions(session, input_data):
    start_time = time.time()
    predictions = session.run(None, {"modelInput": input_data})
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return predictions[0][0, :, :, 0]

def record_inference_time(inference_times,start_time,image_path,predictions):
    inference_time = time.time() - start_time
    inference_times.append(inference_time)

def predict_images(image_paths, model, threshold, input_shape,inference_times):
    for image_path in image_paths:
        img = preprocess_image(image_path, input_shape)
        start_time = time.time()
        predictions = make_predictions(model, img)
        record_inference_time(inference_times,start_time,image_path,predictions)
        result_image = cv2.imread(image_path)
        output_image = highlight_grid(predictions,result_image)
        #save_output(image_path,output_image)

def highlight_grid(prediction_grid,result_image):
    for h in range(40):
        for w in range(20):
            prediction = prediction_grid[h, w]
            if prediction > threshold:
                x1 = w * (result_image.shape[1] // 20)
                x2 = (w + 1) * (result_image.shape[1] // 20)
                y1 = h * (result_image.shape[0] // 40)
                y2 = (h + 1) * (result_image.shape[0] // 40)

                overlay = result_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_color, -1) 
                cv2.addWeighted(overlay, 0.5, result_image, 0.5, 0, result_image)  

    for h in range(40):
        for w in range(20):
            prediction = prediction_grid[h, w]
            if prediction > threshold:
                x1 = w * (result_image.shape[1] // 20)
                x2 = (w + 1) * (result_image.shape[1] // 20)
                y1 = h * (result_image.shape[0] // 40)
                y2 = (h + 1) * (result_image.shape[0] // 40)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), border_color, border_thickness)
    return result_image

def save_output(image_path,result_image):
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
    #Display the output in Colab (for visualization)
    cv2.imshow("Predictions", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_openvino_model(model_xml, model_bin):
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
    return exec_net

def make_openvino_predictions(exec_net, input_data):
    start_time = time.time()
    exec_net.infer(inputs={"modelInput": input_data})
    inference_time = time.time() - start_time
    print(f"OpenVINO Inference time: {inference_time} seconds")
    return inference_time

if __name__ == "__main__":
    folder = "test_images"
    model_path = 'my_model.onnx.pb'
    quant_model_path = 'my_quant_model.onnx.pb'
    threshold = 0.7
    input_shape = (500, 500)
    highlight_color = (0, 0, 255, 100)  
    border_color = (0, 0, 255)  
    border_thickness = 2  
    inference_times_1 = []
    inference_times_2 = []
    inference_times_3 = []
    openvino_inference_times = []
    
    image_paths = load_image_paths(folder)
    model = load_onnx_model(model_path)
    predict_images(image_paths, model, threshold, input_shape,inference_times_1)

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_optimized = load_onnx_model(model_path, options)
    predict_images(image_paths, model_optimized, threshold, input_shape,inference_times_2)

    quantized_model = quantize_dynamic(model_path,quant_model_path,weight_type= QuantType.QUInt8)
    quantized_model = load_onnx_model(model_path)
    predict_images(image_paths, quantized_model, threshold, input_shape,inference_times_3)



    openvino_model_xml = "path_to_openvino_model.xml"
    openvino_model_bin = "path_to_openvino_model.bin"
    openvino_model = load_openvino_model(openvino_model_xml, openvino_model_bin)

    openvino_inference_times = []
    
    for image_path in image_paths:
        img = preprocess_image(image_path, input_shape)
        
        # Inference with OpenVINO
        openvino_inference_time = make_openvino_predictions(openvino_model, img)
        openvino_inference_times.append(openvino_inference_time)

    plt.plot(inference_times_1, label='Model 1')
    plt.plot(inference_times_2, label='Model 2')
    plt.plot(inference_times_3, label='Model 2')
    plt.plot(openvino_inference_times, label='OpenVINO')
    plt.xlabel('Image Index')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time per Image for Different Models')
    plt.legend()
    plt.show()

    