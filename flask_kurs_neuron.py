import base64
import os
import uuid

import torch
import cv2

import numpy as np
from flask import Flask, render_template, request, jsonify
# from flask_restx import Api, Resource, fields

app = Flask(__name__)

static_path = './static/images/'

gallery_forders = [
    {"name": "Glioma", "url": "./static/images/Glioma"},
    {"name": "Meningioma", "url": "./static/images/Meningioma"},
    {"name": "No Tumor", "url": "./static/images/No Tumor"},
    {"name": "Pituitary", "url": "./static/images/Pituitary"},
]

model_yolo = torch.hub.load(repo_or_dir='./models/first_model/yolov5', model='custom',
                                 path='./models/first_model/yolov5/runs/train/mrt_yolov5s_results/weights/best.pt', source = 'local', force_reload=True)

# api-документация
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/api/gallery", methods=['GET'])
def load_gallery():
    folder_name = request.args.get('folder')
    folder_path = os.path.join(static_path, folder_name)
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_links = [{"url": 'http://192.168.0.166:5000/static/images/' + folder_name + '/' + f} for f in image_files]

    print(image_links)

    return image_links


@app.route('/api/test', methods=['get'])
def test_api():
    return {"hello": 123, "test": 321, "heloo": "looma"}


@app.route('/api/detect', methods=['POST'])
def api_detect():
    try:
        base64_image = request.form['image']
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'message': "Возникла ошибка при отправке фото!"})

    results = model_yolo(img)
    processed_img = results.render()[0]  # Отрисованное изображение с обнаруженными объектами

    classes = model_yolo.names

    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_base64 = base64.b64encode(buffer).decode('utf-8')

    detections = results.xyxy[0]

    if len(detections) > 0:
        sorted_detections = detections[detections[:, 4].argsort(descending=True)]

        most_probable_class_id = int(sorted_detections[0, 5].item())

        most_probable_class_name = classes[most_probable_class_id]
    else:
        return jsonify({'image': processed_base64, "text": "Ничего не найдено", "class": "!"})

    return jsonify({'image': processed_base64, "text": "Определено: ", "class": most_probable_class_name})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        class_yolo = request.form['class']
        base64_image = request.form['image']
        image_data = base64.b64decode(base64_image)
    except Exception:
        return jsonify({'message': "Возникла ошибка при отправке фото!"})

    class_url = None
    for folder in gallery_forders:
        if folder["name"] == class_yolo:
            class_url = folder["url"]
            break

    file_name = f"{uuid.uuid4()}.jpg"
    with open(os.path.join(class_url, file_name), "wb") as file:
        file.write(image_data)

    return jsonify({'message': "Изображение успешно загружено!"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)