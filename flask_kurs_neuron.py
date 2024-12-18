import os

import torch
import cv2

from PIL import Image
from click.testing import Result

from googletrans import Translator

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
# from flask_restx import Api, Resource, fields

from keras.src.legacy.preprocessing import image
from keras import utils

app = Flask(__name__)

menu = [
    {"name": "Главная", "url": "/"},
    {"name": "Галерея", "url": "/gallery"},
]

@app.route("/")
def index():
    return render_template('index.html', title="Костин М.М. - ИСТ-301", menu=menu)


@app.route("/gallery", methods=['POST', 'GET'])
def upload_clothes():
    pass


@app.route('/api', methods=['get'])
def test_api():
    return {"hello": 123, "test": 321, "heloo": "looma"}


if __name__ == "__main__":
    app.run(debug=True)


