import os

import cv2
from flask import Flask, jsonify, request
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES']=""

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Flask webserver
app = Flask(__name__)

CONF_THRES = 0.3

@app.route('/od',methods = ['POST'])
def od():
    img = load_images_to_opencv_from_flask(request)
    # Inference
    results = model(img)
    response = list()
    preds = results.xyxy[0].tolist()
    for pred in preds:
        if pred[-2] > CONF_THRES and results.names[int(pred[-1])] != "person":
            obj = dict()
            obj["bb"]= pred[0:4]
            obj["pred"] = results.names[int(pred[-1])]
            obj["score"] = pred[-2]
            response.append(obj)
    return jsonify(response)

def load_images_to_opencv_from_flask(request):
    """Use cv2.imdecode to load images directly to memory from flask request.files
    Args:
        request ([Flask.request]): Flask request object
    Returns:
        [tuple]: tuple of cv loaded images
    """
    #query_img
    img1 = request.files["image"].read()
    npimg1 = np.frombuffer(img1, np.uint8)
    img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
    return img1

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
