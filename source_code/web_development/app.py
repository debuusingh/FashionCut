from flask import Flask,render_template,request
import os
import cv2
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('model/model.h5')

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
    label = model.predict(img)
    label[0][0]=0.1
    label = 'man' if label[0][0] >= 0.8 else 'woman'
    
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

    return render_template(
        'index.html',
        label=label,
        img_path='static/' + image.filename,
        collection_images=collection_images
    )


if __name__ == "__main__":
    app.run(debug=True,port=5001)
