from flask import Flask, render_template, url_for, redirect, request
from flask_bootstrap import Bootstrap
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import os
import json
import cv2
from keras import backend as K

app = Flask(__name__)
Bootstrap(app)

OUTPUT_DIR = 'static'
SIZE = 28
data =[]

def get_prediction(image):
    image = cv2.imread(image)
    image = cv2.resize(image, (28, 28))
    #image = image.astype("float") / 255.0
    image = img_to_array(image)
    image =np.array(image, dtype="float")/255.0
    image = np.expand_dims(image, axis=0)

    model = load_model("santa_not_santa.model")
    (notSanta, santa) = model.predict(image)[0] 
    label = "Santa" if santa > notSanta else "Not Santa"
    K.clear_session()
    return label
    
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join(OUTPUT_DIR, uploaded_file.filename)
                uploaded_file.save(image_path)                
                class_name = get_prediction(image_path)
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path,
                    'size': SIZE
                }
                print(result)
                return render_template('static.html', result=result)
    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)
