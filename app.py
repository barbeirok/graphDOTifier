import base64
import socket
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import Model.Classifier as classifier

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    if not request.files:
        return {'error': 'No files uploaded'}, 400
    graph = request.files['graph']
    if graph.filename == '':
        return jsonify({"error": "No selected file"}), 400
    graph = classifier.resize(graph)
    class_predictions, bbox_predictions = classifier.predict_image(graph)
    if class_predictions is None or bbox_predictions is None:
        return jsonify({"error": "Error processing image"}), 500

    fig = classifier.visualize_predictions(graph, class_predictions, bbox_predictions)
    return jsonify(fig), 200



if __name__ == '__main__':
    app.run()
