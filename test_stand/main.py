from flask import Flask, render_template, request
import random
from estimator import Estimator
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'something very secret'

@app.route('/')
def start():
    return render_template("index.html")


@app.route('/estimate_df', methods=['GET', 'POST'])
def estimate_df():
    if request.method == "POST":
        method = request.form.get('method')
        dataset = request.form.get('dataset')
        if dataset == 'celebdf':
            data_path = '/home/woghan/projects/diploma/diploma/datasets/celebdf'
        elif dataset == 'dftimit_hq':
            data_path = '/home/woghan/projects/diploma/diploma/datasets/DeepfakeTIMIT'
        elif dataset == 'dftimit_lq':
            data_path = '/home/woghan/projects/diploma/diploma/datasets/DeepfakeTIMIT'
        elif dataset == 'dfdc':
            data_path = '/home/woghan/projects/diploma/diploma/datasets/dfdcdataset'
        elif dataset == 'uadfv':
            data_path = '/home/woghan/projects/diploma/diploma/datasets/uadfv/fake_videos'
            
        res = Estimator.benchmark(dataset, method, data_path)
        print(res)
        return render_template('estimate_df.html', prediction=res, method=method, dataset=dataset)
    return render_template('estimate_df.html', prediction=None, method=None, dataset=None)


@app.route('/estimate_vid', methods=['GET', 'POST'])
def estimate_vid():
    if request.method == "POST":
        method = request.form.get('method')
        video = request.files['video']
        label = request.form.get('type')
        if video:
            video_location = os.path.join(
                '/home/woghan/projects/diploma/diploma/test_stand/',
                video.filename
            )
            video.save(video_location)
            
        method, pred = Estimator.detect_single(video_path=video_location, method=method, label=label)
        return render_template('estimate_vid.html', prediction=pred)
    return render_template('estimate_vid.html', prediction=None)


if __name__ == "__main__":
    app.run(port=8080, debug=True)