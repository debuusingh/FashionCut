from flask import Flask,render_template,request
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # For displaying images


app = Flask(__name__)
# model = keras.models.load_model('model/model.h5')

@app.route('/')
def index():    
    return render_template('index.html')

# @app.route('/upload', methods=['GET''POST'])
# def upload():
#     pass

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['myfile']
    image.save(os.path.join("static", image.filename))
    img = cv2.imread('static/' + image.filename)
    img = cv2.resize(img, (240, 240))
    img = img / 255.0
    img = img.reshape(1, 240, 240, 3)
    # label = model.predict(img)
    # label[0][0]=0.1
    label = 'man'
    # label = 'man' if label[0][0] >= 0.8 else 'woman'
    
    # Dynamically set the collection based on prediction
    if label == 'man':
        collection_images = [
            "static/man_stock/shirt_01.png",
            "static/man_stock/shirt_02.jpg",
            "static/man_stock/shirt_03.jpg",
            "static/man_stock/shirt_04.jpg",
            "static/man_stock/tshirt_05.jpg",
        ]
    else:
        collection_images = [
            "static/women_stock/long_sleeve_02.jpg",
            "/static/women_stock/stock_03_jpg.jpeg",
            "/static/women_stock/stock_04.jpg",
            "/static/women_stock/stock_05.jpg",
            "/static/women_stock/stock_01.jpg",
        ]
    # return  img_path='static/' + image.filename,
    return render_template(
        'index.html',
        label=label,
        img_path=fn('static/' + image.filename,4),
        collection_images=collection_images
    )


def fn(path, class_to_test):
    IMAGE_FILENAMES = [path]

    # Define distinct colors for each class (3-channel RGB)
    MASK_COLORS = [
        (255, 0, 0),  # Red for class 0
        (0, 255, 0),  # Green for class 1
        (0, 0, 255),  # Blue for class 2
        (255, 255, 0),  # Yellow for class 3
        (255, 0, 255),  # Magenta for class 4
        (0, 255, 255)  # Cyan for class 5
    ]

    # Create the options for the ImageSegmenter
    base_options = python.BaseOptions(model_asset_path='model/selfie_multiclass_256x256.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)

    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        for image_file_name in IMAGE_FILENAMES:
            # Create the MediaPipe image file that will be segmented
            image = mp.Image.create_from_file(image_file_name)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.category_mask.numpy_view()

            # Convert the original image to RGB for display
            original_image = image.numpy_view()
            if original_image.shape[-1] == 4:  # If RGBA, convert to RGB
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

            # Create a copy of the original image to keep the background as is
            output_image = original_image.copy()

            # Apply color for the selected class
            mask_color = np.array(MASK_COLORS[class_to_test], dtype=np.uint8)
            selected_mask = category_mask == class_to_test

            # Overlay the selected segment with its color
            overlay_image = np.zeros_like(original_image, dtype=np.uint8)
            overlay_image[selected_mask] = mask_color
            output_image = cv2.addWeighted(output_image, 0.7, overlay_image, 0.3, 0)

            # Find contours for the selected segment
            selected_mask_uint8 = selected_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(selected_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw blue borders on the contours
            cv2.drawContours(output_image, contours, -1, (255, 0, 0), thickness=2)  # Blue border

            # Save the output image
            output_image_path = f'static/segmented_{os.path.basename(image_file_name)}'
            cv2.imwrite(output_image_path, output_image)

    return output_image_path


if __name__ == "__main__":
    app.run(debug=True,port=5001)


