import os

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from celery import Celery
from celery.result import AsyncResult

import matplotlib.pyplot as plt
from skimage.transform import resize

from ml_models.beheaded_inception3 import beheaded_inception_v3
from ml_models.caption_net import load_model, generate_caption


redis_host = os.environ.get('REDIS_HOST') or 'localhost'
celery_app = Celery('app', backend=f'redis://{redis_host}', broker=f'redis://{redis_host}')
app = Flask(__name__)

# load model
mlflow_host = os.environ.get('MLFLOW_HOST') or 'localhost'
mlflow_port = os.environ.get('MLFLOW_PORT') or 5000
mlflow_url = f'http://{mlflow_host}:{mlflow_port}'
experiment_name = 'lsml2_demo'
model = load_model(mlflow_url, experiment_name)
model.eval()

# load vectorizer
vectorizer = beheaded_inception_v3().eval()


@celery_app.task
def get_caption(img_path):
    img = plt.imread(img_path)
    img = img[:, :, :3]
    img = resize(img, (299, 299))
    caption = generate_caption(model, vectorizer, img, t=5.)
    return str(caption)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get file from post request
        f = request.files['file']

        # save file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(img_path)

        # launch celery task and get the task id
        task_id = get_caption.delay(img_path).id
        response = {'task_id': task_id}

        return jsonify(response)

    return None


@app.route('/task/<task_id>', methods=['GET'])
def task(task_id):
    task = AsyncResult(task_id, app=celery_app)
    response = {
        'ready': task.ready(),
        'result': str(task.result) if task.ready() else None
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3000)
