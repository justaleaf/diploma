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
        data_path = '/home/woghan/projects/diploma/diploma/datasets/celebdf'
            
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

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            select = request.form.get('methodchoice')
            if image_file.filename.endswith(".jpg"):
                method, pred = dfdetector.DFDetector.detect_single(image_path=image_location, method=select)
            elif image_file.filename.endswith(".mp4") or image_file.filename.endswith(".avi"):
                method, pred = dfdetector.DFDetector.detect_single(video_path=image_location, method=select)
            # add random number to circumvent browser image caching of images with the same name
            randn = random.randint(0, 1000)
            return render_template("index.html", prediction = pred, method=method, image_loc=image_file.filename[:-4] + '.jpg', random_num = randn)
    return render_template("index.html", prediction = 0, method=None,image_loc = None,random_num=None)


if __name__ == "__main__":
    app.run(port=8080, debug=True)