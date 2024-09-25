from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import io
import cv2
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Initialize SAM model
CHECKPOINT_PATH = r"C:\Users\Administrator\Downloads\sam_vit_h_4b8939.pth"  # Replace with the actual path to your checkpoint file
MODEL_TYPE = "vit_h"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read the image in memory
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream).convert('RGB')
        image_rgb = np.array(image)

        # Run the segmentation code
        sam_result = mask_generator.generate(image_rgb)

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

        # Convert the annotated image to a format that can be sent as a response
        _, buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        response_image = io.BytesIO(buffer)

        logging.debug("Image processed and ready to be sent")

        return send_file(response_image, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)