
from real_time import Real_Time_Analysis

from flask import Flask, render_template, jsonify

app = Flask(__name__)

analysis = Real_Time_Analysis()

@app.route('/data')
def data():
    total_views = 1234
    comments = 567
    likes = 890

    data = {
        'total_views': total_views,
        'comments': comments,
        'likes': likes,
    }

    return jsonify(data=data)

@app.route('/FER')
def fer():
    return jsonify(data=analysis._get_FER_pc())

@app.route('/EYE_EMOTION')
def eye_emotion():
    return jsonify(data=analysis._get_eye_emotion_pc())

@app.route('/EYE_GAZE')
def eye_gaze():
    return jsonify(data=analysis._get_eye_gaze_pc())

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)