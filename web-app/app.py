import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


import torch
import flask
from flask import Flask, request, render_template
import json
import base64
import uuid
import os
import sys
sys.path.append("../")
import deeplab_resnet
import deeplab_xception
import centermask2

UPLOAD_FOLDER = './static/uploads'
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        file = request.files.get('file')
        filename = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # SEGMENT resnet
        files_resnet, classes_resnet = deeplab_resnet.generate_segments(file_path, UPLOAD_FOLDER)
        classes_resnet[0] = 'all'
        if len(classes_resnet) == 2:
            files_resnet.pop(0)
            classes_resnet.pop(0)

        # SEGMENT resnet
        files_xception, classes_xception = deeplab_xception.generate_segments(file_path, UPLOAD_FOLDER)
        classes_xception[0] = 'all'
        if len(classes_xception) == 2:
            files_xception.pop(0)
            classes_xception.pop(0)

        # SEGMENT centermask2
        file_centermask = centermask2.generate_segments(file_path, UPLOAD_FOLDER)

        return flask.jsonify({'input_file': file_path,
                              'segment_resnet': files_resnet,
                              'classes_resnet': classes_resnet,
                              'segment_xception': files_xception,
                              'classes_xception': classes_xception,
                              'segment_centermask': file_centermask})

    except Exception as ex:
        erro = str(ex)
        print(erro)
        return app.response_class(response=json.dumps(erro), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
