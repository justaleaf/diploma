from flask import Flask, render_template, request
import random
from estimator import Estimator

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


@app.route('/estimate_vid')
def estimate_vid():
    return render_template("estimate_vid.html")


if __name__ == "__main__":
    app.run(port=8080, debug=True)