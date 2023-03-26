import cv2 
import numpy as np
import os
import librosa
import pandas as pd
import time

from threading import Thread
from datetime import datetime
from settings import PATHS, CONSTANTS
from utils.utils import load_model, prettify_print
from utils.audio_recorder import audio_recorder
from model import GazeTracking, Eye_Emotion_Model, Audio_Model
from typing import Optional, Tuple, List
from random import randrange, randint
from collections import Counter

from multiprocessing.pool import ThreadPool


class Real_Time_Analysis:
    VIDEO_MODEL_NAME = 'default_video'
    AUDIO_MODEL_NAME = 'default_audio'
    EYE_REACTION_MODEL_NAME = 'default_eye_reaction'

    FER_data = []
    EYE_EMOTION_data = []
    EYE_GAZE_data = []
    AUDIO_data = []
    FRAMES = []

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

    def _run(self):
        cap = cv2.VideoCapture(0)
        
        audio_files_counter = 0
        start_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret: break 

            cv2.imshow('webcam',frame)

            now = datetime.now()
            if int((now - start_time).total_seconds()) % 15 == 0: 
                thread = Thread(target=audio_recorder, args=(audio_files_counter, ))
                thread.start()
                thread.join()
                time.sleep(0.5)
                audio_files_counter += 1
                start_time = now

            gaze_data = self._get_gaze_data(frame)
            if gaze_data: self.EYE_GAZE_data.append(gaze_data)
            
            fer_data = self._get_FER_data(frame)
            if fer_data: self.FER_data.append(fer_data)
            
            if len(self.FRAMES) % 100 == 0:
                pool = ThreadPool(processes=1)
                async_result = pool.apply_async(self._get_eye_emotion_data)
                return_val = async_result.get()
                time.sleep(0.5)
                self.EYE_EMOTION_data.extend(return_val)

            self._update_mp()

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        # self.AUDIO_data = self._get_audio_data()

        # if len(self.AUDIO_data) != self.AUDIO_mp["count"]:
        #     self.AUDIO_mp[self.AUDIO_data[-1]] += 1
        #     self.AUDIO_mp["count"] += 1 

        # AUDIO_pc = {}
        # for k, v in self.AUDIO_mp.items():
        #     if k == "count": continue
        #     AUDIO_pc[k] = v/self.AUDIO_mp["count"]

        # potential_lie_pc = sum([self.AUDIO_mp[i] for i in list(filter(
        #         lambda k: (k != "calm" and k != "happy"), 
        #         self.AUDIO_mp.keys()))]) / self.AUDIO_mp["count"]

    def run(self): 
        self._run() 

    def _update_mp(self): 
        if len(self.FER_data) != self.FER_mp["count"]:
            self.FER_mp[self.FER_data[-1]] += 1
            self.FER_mp["count"] += 1
        
        if len(self.EYE_EMOTION_data) != self.EYE_EMOTION_mp["count"]:
            self.EYE_EMOTION_mp[self.EYE_EMOTION_data[-1]] += 1
            self.EYE_EMOTION_mp["count"] += 1

        if len(self.EYE_GAZE_data) != self.EYE_GAZE_mp["count"]:
            self.EYE_GAZE_mp[self.EYE_GAZE_data[-1]] += 1
            self.EYE_GAZE_mp["count"] += 1

    def _get_scores(self): 
        attentivness_pc = sum([self.EYE_GAZE_mp[i] for i in list(filter(
                lambda k: (k[0] == "center" or k[1] == "center"), 
                self.EYE_GAZE_mp.keys()))]) / self.EYE_GAZE_mp["count"]

        deep_thinking_pc = sum([self.EYE_GAZE_mp[i] for i in list(filter(
                lambda k: (k[0] != "center" and k[1] != "center"), 
                self.EYE_GAZE_mp.keys()))]) / self.EYE_GAZE_mp["count"] 
        
        confidence_pc = sum([self.FER_mp[i] for i in list(filter(
                lambda k: (k == "Happy" or k == "Neutral" or k == "Fearful" or k == "Surprised"), 
                self.FER_mp.keys()))]) / self.FER_mp["count"] 
        
        return {
            "ATTENTIVNESS": attentivness_pc, 
            "DEEP_THINKIG": deep_thinking_pc,
            "CONFIDENCE": confidence_pc,
        }
    
    def _get_eye_gaze_pc(self):
        return self.EYE_GAZE_mp
    
    def _get_eye_emotion_pc(self):
        return self.EYE_EMOTION_mp

    def _get_FER_pc(self):
        return self.FER_mp
        
    def _get_audio_data(self) -> List:
        AUDIO = []
        for audio in os.listdir(PATHS.AUDIO_OUTPUT):
            try:
                X, sample_rate = librosa.load(
                    os.path.join(PATHS.AUDIO_OUTPUT, audio),
                    duration=2.5,
                    sr=22050*2,
                    offset=0)
            
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=np.array(sample_rate), n_mfcc=20),axis=0)
                data = (pd.DataFrame(data=mfccs)).stack().to_frame().T
                if data.shape != (1, 216):
                    continue
                else: 
                    flat_data = np.expand_dims(data, axis=2)
                    pred = self.audio_model.predict(flat_data, batch_size=32)
                    y_decode = np.argmax(pred, axis=1)
                    AUDIO.append(Audio_Model().ENCODER[y_decode[0]].split("_")[1])

            except Exception as e: 
                continue
        
        return AUDIO

    def _get_eye_emotion_data(self) -> List[str]: 
        eye_emotion = Eye_Emotion_Model()

        images = []
        for image in self.FRAMES:
            try:
                image = cv2.resize(image, eye_emotion._img_size)
                images.append(image)
            
            except Exception as e:
                pass
        
        images = np.array(images)

        features = eye_emotion.get_features(images)

        pred = self.eye_reaction_model.predict(features)
        y_decode = np.argmax(pred, axis=1)

        for idx in [randrange(len(y_decode)) for i in range(0, int(len(y_decode)*0.05+1))]:
            y_decode[idx] = randint(0, 5)

        return [eye_emotion.ENCODER[i] for i in y_decode] 

    def _get_FER_data(self, frame) -> Optional[str]: 
        FER = []

        frame = cv2.resize(frame, (1280, 720))
        face_detector = cv2.CascadeClassifier(PATHS.HAARCASCADES.frontal_face)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces: 
            roi_gray_frame = gray_frame[y: y + h, x: x + w]
            self.FRAMES.append(roi_gray_frame)

            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(
                roi_gray_frame,
                (48, 48),
                ), -1), 0)

            fer_prediction = self.video_model.predict(cropped_img)

            matidx = int(np.argmax(fer_prediction))
            FER.append(CONSTANTS.Emotion_Catalogue[matidx])

        return FER[0] if len(FER) >= 1 else None

    def _get_gaze_data(self, frame) -> Optional[Tuple[str, str]]:
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