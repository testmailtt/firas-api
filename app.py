from flask import Flask,request
import numpy as np
import re
import base64
import os
from mnist_model import model
from scipy.misc import imread, imresize,imshow

app = Flask(__name__)
model = model()

@app.route('/', methods=['GET'])
def home():
    return {'message':'home'}

@app.route('/predict/', methods=['POST'])
def predict():
    img = request.files['image']
    img.save(os.path.join('./', 'output.png'))
    
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)
    
    out = model.predict(x)
    response = np.array_str(np.argmax(out, axis=1))
    
    
    
    
    return {'response':response}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)