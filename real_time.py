import cv2 
import numpy as np
import os
import librosa
import pandas as pd
import time
import threading

from threading import Thread
from datetime import datetime
from settings import PATHS, CONSTANTS
from utils import *
from model import GazeTracking, Eye_Emotion_Model, Audio_Model
from typing import Optional, Tuple, List
from random import randrange, randint
from collections import Counter

from multiprocessing.pool import ThreadPool

lock = threading.Lock()
outputFrame = None
count = 0

def gen_frames():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


class Real_Time_Analysis:
    VIDEO_MODEL_NAME = 'default_video'
    AUDIO_MODEL_NAME = 'default_audio'
    EYE_REACTION_MODEL_NAME = 'default_eye_reaction'

    VIDEO_RECORDING_OUTPUT = "frames/"
    AUDIO_RECORDING_OUTPUT = "audio/"

    FER_data = []
    EYE_EMOTION_data = []
    EYE_GAZE_data = []
    AUDIO_data = [] 

    FER_mp = {"Angry": 0, "Disgusted": 0, "Fearful": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprised": 0, "count": 0}    
    EYE_EMOTION_mp = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "count": 0}
    EYE_GAZE_mp = {
        ("top", "right"): 0, 
        ("top", "left"): 0, 
        ("top", "center"): 0, 
        ("center", "right"): 0, 
        ("center", "left"): 0, 
        ("center", "center"): 0, 
        ("bottom", "right"): 0, 
        ("bottom", "left"): 0, 
        ("bottom", "center"): 0,
        "count": 0}
    AUDIO_mp = {"calm": 0, "happy": 0, "sad": 0, "angry": 0, "fearful": 0, "count": 0}


    def __init__(self) -> None:
        self.video_model = load_model(self.VIDEO_MODEL_NAME)
        self.audio_model = load_model(self.AUDIO_MODEL_NAME)
        self.eye_reaction_model = load_model(self.EYE_REACTION_MODEL_NAME)
        self.eye_gaze_model = GazeTracking()

    def run(self):
        global outputFrame, lock

        cap = cv2.VideoCapture(0)
        
        audio_files_counter = 0
        start_time = datetime.now()

        while True:
            success, frame = cap.read()
            if not success: break 
            
            now = datetime.now()
            if int((now - start_time).total_seconds()) % 10 == 0: 
                thread = Thread(target=audio_recorder, args=(audio_files_counter, ))
                thread.daemon = True
                thread.start()
                thread.join()
                audio_files_counter += 1
                start_time = now

            gaze_data = self._get_eye_gaze_data(frame)
            if gaze_data: self.EYE_GAZE_data.append(gaze_data)
            
            fer_data = self._get_FER_data(frame)
            if fer_data: self.FER_data.append(fer_data)

            self.save_frame(frame)

            self.FER_mp = self._convert_list_to_mp(self.FER_data)
            self.EYE_GAZE_mp = self._convert_list_to_mp(self.EYE_GAZE_data)

            with lock: outputFrame = frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        return self.analyse()

    def analyse(self): 
        (EYE_EMOTION_data, AUDIO_data) = self._get_data()

        EYE_EMOTION_mp = self._convert_list_to_mp(EYE_EMOTION_data)
        AUDIO_mp = self._convert_list_to_mp(AUDIO_data)

        self.EYE_EMOTION_mp = EYE_EMOTION_mp
        self.AUDIO_mp = AUDIO_mp

        return self._get_scores(self.FER_mp, EYE_EMOTION_mp, self.EYE_GAZE_mp, AUDIO_mp)

    def _get_data(self) -> Tuple[List, List]: 
        EYE_EMOTION_data = self._get_eye_emotion_data(self.eye_reaction_model)
        AUDIO_data = self._get_audio_data(self.audio_model)

        return (EYE_EMOTION_data, AUDIO_data)

    def _get_scores(self, FER_mp, EYE_EMOTION_mp, EYE_GAZE_mp, AUDIO_mp) -> Tuple[float, float, float, float]: 
        attentivness_pc = self._calc_attentivness_pc(EYE_GAZE_mp, EYE_EMOTION_mp)
        deep_thinking_pc = self._calc_deep_thinking_pc(EYE_GAZE_mp, EYE_EMOTION_mp)
        confidence_pc = self._calc_confidence_pc(FER_mp, EYE_EMOTION_mp)
        potential_lie_pc = self._calc_potential_lie_pc(AUDIO_mp)

        return (attentivness_pc, deep_thinking_pc, confidence_pc, potential_lie_pc)

    def _calc_attentivness_pc(self, EYE_GAZE_mp: dict, EYE_EMOTION_mp: dict) -> float:
        gaze_indicator = EYE_GAZE_mp.get(('center', 'center'), 0)
        eye_indicator = EYE_EMOTION_mp.get('happy', 0)
        return (gaze_indicator + eye_indicator) / (EYE_GAZE_mp.get('count', 1) + EYE_EMOTION_mp.get('count', 1))

    def _calc_deep_thinking_pc(self, EYE_GAZE_mp: dict, EYE_EMOTION_mp: dict) -> float: 
        gaze_indicator = EYE_GAZE_mp.get('count', 0) - EYE_GAZE_mp.get(('center', 'center'), 0)
        eye_indicator = EYE_EMOTION_mp.get('surprise', 0)
        return (gaze_indicator + eye_indicator) / (EYE_GAZE_mp.get('count', 1) + EYE_EMOTION_mp.get('count', 1))
    
    def _calc_confidence_pc(self, FER_mp: dict, EYE_EMOTION_mp: dict) -> float:
        fer_indicator = FER_mp.get('Happy', 0) + FER_mp.get('Neutral', 0)
        eye_indicator = EYE_EMOTION_mp.get('happy', 0)
        return (fer_indicator + eye_indicator) / (FER_mp.get('count', 1) + EYE_EMOTION_mp.get('count', 1))

    def _calc_potential_lie_pc(self, AUDIO_mp: dict) -> float:
            audio_indicator = AUDIO_mp.get('happy', 0) + AUDIO_mp.get('calm', 0)
            return (AUDIO_mp.get('count', 1) - audio_indicator) / AUDIO_mp.get('count', 1)

    def _get_FER_data(self, frame) -> Optional[str]: 
        FER = []

        frame = cv2.resize(frame, (1280, 720))
        face_detector = cv2.CascadeClassifier(PATHS.HAARCASCADES.frontal_face)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces: 
            roi_gray_frame = gray_frame[y: y + h, x: x + w]

            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(
                roi_gray_frame,
                (48, 48),
                ), -1), 0)

            fer_prediction = self.video_model.predict(cropped_img)

            matidx = int(np.argmax(fer_prediction))
            FER.append(CONSTANTS.Emotion_Catalogue[matidx])

        return FER[0] if len(FER) >= 1 else None

    def _get_FER_pc(self):
        return self.FER_mp

    def _get_eye_emotion_data(self, eye_reaction_model) -> List[str]: 
        if not os.path.isdir(self.VIDEO_RECORDING_OUTPUT):
            os.mkdir(self.VIDEO_RECORDING_OUTPUT)
        
        eye_emotion = Eye_Emotion_Model()

        images = []
        for image in os.listdir(self.VIDEO_RECORDING_OUTPUT):
            try:
                image = cv2.imread(os.path.join(
                    self.VIDEO_RECORDING_OUTPUT, image))
                image = cv2.resize(image, eye_emotion._img_size)
                images.append(image)
            
            except Exception as e:
                prettify_print(f"Error: {e}", "load eye emotion data error")
        
        images = np.array(images)

        features = eye_emotion.get_features(images)

        pred = eye_reaction_model.predict(features)
        y_decode = np.argmax(pred, axis=1)

        for idx in [randrange(len(y_decode)) for i in range(0, int(len(y_decode)*0.05+1))]:
            y_decode[idx] = randint(0, 5)

        return [eye_emotion.ENCODER[i] for i in y_decode] 
        
    def _get_eye_emotion_pc(self):
        return self.EYE_EMOTION_mp

    def _get_audio_data(self, audio_model) -> List:
        AUDIO = []
        for audio in os.listdir(self.AUDIO_RECORDING_OUTPUT):
            try:
                X, sample_rate = librosa.load(
                    os.path.join(self.AUDIO_RECORDING_OUTPUT, audio),
                    duration=2.5,
                    sr=22050*2,
                    offset=0)
            
                mfccs = np.mean(librosa.feature.mfcc(
                    y=X, sr=np.array(sample_rate), n_mfcc=20),axis=0)
                data = (pd.DataFrame(data=mfccs)).stack().to_frame().T
                if data.shape != (1, 216):
                    continue
                else: 
                    flat_data = np.expand_dims(data, axis=2)
                    pred = audio_model.predict(flat_data, batch_size=32)
                    y_decode = np.argmax(pred, axis=1)
                    AUDIO.append(Audio_Model().ENCODER[y_decode[0]].split('_', 1)[1])

            except Exception as e: 
                prettify_print(f"Error: {e}", "load audio data error")
        
        return AUDIO

    def _get_audio_pc(self):
        return self.AUDIO_mp

    def _get_eye_gaze_data(self, frame) -> Optional[Tuple[str, str]]:
        self.eye_gaze_model.refresh(frame)

        v_ratio = self.eye_gaze_model.vertical_ratio()
        h_ratio = self.eye_gaze_model.horizontal_ratio()

        horizontal, vertical = "", ""
        v_dir = self._get_direction(v_ratio)
        h_dir = self._get_direction(h_ratio)

        mp = {
            "positive": {"y": "top", "x": "right"},
            "zero": {"y": "center", "x": "center"},
            "negative": {"y": "bottom", "x": "left"}
        }

        if v_dir: vertical = mp[v_dir]["y"]
        if h_dir: horizontal = mp[h_dir]["x"]
        
        if vertical != "" and horizontal != "":
            return (vertical, horizontal)
        else: 
            return None
    
    def _get_eye_gaze_pc(self):
        return self.EYE_GAZE_mp

    def _get_direction(self, ratio) -> Optional[str]: 
        if ratio is None: return None
        if ratio <= 0.35 and ratio >= 0:
            return "positive"
        elif ratio > 0.35 and ratio < 0.65:
            return "zero"
        elif ratio >= 0.65 and ratio <= 1:
            return "negative"
        else:
            return None
        
    def save_frame(self, frame): 
        global count

        face_detector = cv2.CascadeClassifier(PATHS.HAARCASCADES.frontal_face)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color_frame = frame[y: y + h, x: x + w]

            cv2.imwrite(
                f"{self.VIDEO_RECORDING_OUTPUT}/frame_{count}.jpg",
                roi_color_frame)
            
        count += 1

    def _convert_list_to_mp(self, data: List) -> dict:  
        data_mp = Counter(data)
        data_mp['count'] = len(data)

        return dict(data_mp)