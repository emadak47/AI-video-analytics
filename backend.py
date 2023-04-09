import re
import os
import signal


from utils import is_file_allowed
from real_time import Real_Time_Analysis, gen_frames
from static import Static_Analysis

from flask import *
from utils import *

app = Flask(__name__, 
            static_folder='static/',
            template_folder='templates')

rt_analysis = Real_Time_Analysis()
st_analysis = Static_Analysis()


@app.route('/st-data')
def st_data():
    (attentivness_pc, deep_thinking_pc, confidence_pc, potential_lie_pc) = \
        st_analysis.run()

    FER_mp = st_analysis._get_FER_pc()
    EYE_EMOTION_mp = st_analysis._get_eye_emotion_pc()
    EYE_GAZE_mp = st_analysis._get_eye_gaze_pc()
    AUDIO_mp = st_analysis._get_audio_pc()

    return jsonify(data={
        "FER": FER_mp,
        "EYE_EMOTION": EYE_EMOTION_mp,
        "EYE_GAZE":  {"_".join(key): value for key, value in EYE_GAZE_mp.items()},
        "AUDIO": AUDIO_mp,
        "Scores": {
            "attentivness": attentivness_pc,
            "deep_thinking": deep_thinking_pc,
            "confidence": confidence_pc,
            "potential_lie": potential_lie_pc,
        }
    })

@app.route('/rt-data')
def rt_data():
    (attentivness_pc, deep_thinking_pc, confidence_pc, potential_lie_pc) = \
        rt_analysis.run()

    FER_mp = rt_analysis._get_FER_pc()
    EYE_EMOTION_mp = rt_analysis._get_eye_emotion_pc()
    EYE_GAZE_mp = rt_analysis._get_eye_gaze_pc()
    AUDIO_mp = rt_analysis._get_audio_pc()

    return jsonify(data={
        "FER": FER_mp,
        "EYE_EMOTION": EYE_EMOTION_mp,
        "EYE_GAZE":  {"_".join(key): value for key, value in EYE_GAZE_mp.items()},
        "AUDIO": AUDIO_mp,
        "Scores": {
            "attentivness": attentivness_pc,
            "deep_thinking": deep_thinking_pc,
            "confidence": confidence_pc,
            "potential_lie": potential_lie_pc,
        }
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/acknowledgement')
def acknowledgement():
    return render_template('acknowledgement.html', name = "smth", status="success")

@app.route('/uploads', methods=['POST', 'GET'])
def uploads():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        f = request.files['file']

        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if f and is_file_allowed(f.filename): 
            f.save(f.filename)
            return render_template('acknowledgement.html', name = f.filename, status="success")
    else:
        return render_template('page_not_found.html')

@app.route('/video-feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/terminate', methods=['POST'])
def terminate():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({'message': 'Server terminated.'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)