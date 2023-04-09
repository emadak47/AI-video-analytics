import cv2
import os
import shutil
import numpy as np
import pandas as pd
import librosa
from typing import List, Tuple
from settings import PATHS, CONSTANTS
from utils import *
from model import Eye_Emotion_Model, Audio_Model, GazeTracking
from random import randrange, randint
from collections import Counter


class Static_Analysis:
    VIDEO_MODEL_NAME = 'default_video'
    AUDIO_MODEL_NAME = 'default_audio'
    EYE_REACTION_MODEL_NAME = 'default_eye_reaction'

    VIDEO_RECORDING_OUTPUT = "frames/"
    AUDIO_RECORDING_OUTPUT = "audio/"
    VIDEO_RECORDING = "trial.mp4"
    AUDIO_RECORDING = "trial.wav"

    FER_mp = {}
    EYE_EMOTION_mp = {}
    EYE_GAZE_mp = {}
    AUDIO_mp = {}

    def __init__(self) -> None:
        self.video_model = load_model(self.VIDEO_MODEL_NAME)
        self.audio_model = load_model(self.AUDIO_MODEL_NAME)
        self.eye_reaction_model = load_model(self.EYE_REACTION_MODEL_NAME)
        self.eye_gaze_model = GazeTracking()

    def run(self):
        self._preprocess()
        return self.analyse()

    def _preprocess(self) -> None:
        save_face_frames(self.VIDEO_RECORDING, self.VIDEO_RECORDING_OUTPUT)
        convert_video_to_audio(self.VIDEO_RECORDING)

    def analyse(self) -> List:
        (FER_data, EYE_EMOTION_data, EYE_GAZE_data, AUDIO_data) = self._get_data()

        FER_mp = self._convert_list_to_mp(FER_data)
        EYE_EMOTION_mp = self._convert_list_to_mp(EYE_EMOTION_data) 
        EYE_GAZE_mp = self._convert_list_to_mp(EYE_GAZE_data)
        AUDIO_mp = self._convert_list_to_mp(AUDIO_data)

        self.FER_mp = FER_mp
        self.EYE_EMOTION_mp = EYE_EMOTION_mp
        self.EYE_GAZE_mp = EYE_GAZE_mp
        self.AUDIO_mp = AUDIO_mp

        return self._get_scores(FER_mp, EYE_EMOTION_mp, EYE_GAZE_mp, AUDIO_mp)

    def _get_data(self) -> Tuple[List, List, List, List]:
        FER_data = self._get_FER_data(self.VIDEO_RECORDING, self.video_model)
        EYE_EMOTION_data = self._get_eye_emotion_data(self.eye_reaction_model)
        EYE_GAZE_data = self._get_eye_gaze_data(self.VIDEO_RECORDING, self.eye_gaze_model)
        AUDIO_data = self._get_audio_data(self.AUDIO_RECORDING, self.audio_model)

        return (FER_data, EYE_EMOTION_data, EYE_GAZE_data, AUDIO_data)

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
    
    def _get_FER_data(self, path_to_video_source: str, video_model) -> List:
        FER = []

        cap = cv2.VideoCapture(path_to_video_source)

        while cap.isOpened():
            try:
                success, frame = cap.read()
                frame = cv2.resize(frame, (1280, 720))
                if not success or frame is None:
                    break

                face_detector = cv2.CascadeClassifier(
                    PATHS.HAARCASCADES.frontal_face)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(
                    gray_frame, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi_gray_frame = gray_frame[y: y + h, x: x + w]

                    cropped_img = np.expand_dims(
                        np.expand_dims(cv2.resize(
                            roi_gray_frame,
                            (48, 48)
                        ), -1), 0)

                    fer_prediction = video_model.predict(cropped_img)

                    maxidx = int(np.argmax(fer_prediction))
                    FER.append(CONSTANTS.Emotion_Catalogue[maxidx])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                prettify_print(f"Error: {e}", "load FER data error")
                break

        cap.release()
        cv2.destroyAllWindows()

        return FER

    def _get_FER_pc(self):
        return self.FER_mp

    def _get_eye_emotion_data(self, eye_reaction_model) -> List:
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

    def _get_audio_data(self, path_to_audio_source: str, audio_model) -> List:
        prep_folder(self.AUDIO_RECORDING_OUTPUT)

        split_wav = SplitWavAudio(
            self.AUDIO_RECORDING_OUTPUT, path_to_audio_source)
        split_wav.multiple_split(sec_per_split=3)

        AUDIO = []
        for audio in os.listdir(self.AUDIO_RECORDING_OUTPUT):
            try:
                X, sample_rate = librosa.load(
                    os.path.join(self.AUDIO_RECORDING_OUTPUT, audio),
                    duration=2.5,
                    sr=22050*2,
                    offset=0)

                mfccs = np.mean(librosa.feature.mfcc(
                    y=X, sr=np.array(sample_rate), n_mfcc=20), axis=0)
                data = (pd.DataFrame(data=mfccs)).stack().to_frame().T
                if data.shape != (1, 216):
                    continue
                else:
                    flat_data = np.expand_dims(data, axis=2)
                    pred = audio_model.predict(flat_data, batch_size=32)
                    y_decode = np.argmax(pred, axis=1)
                    AUDIO.append(Audio_Model().ENCODER[y_decode[0]].rsplit('_', 1)[1])

            except Exception as e:
                prettify_print(f"Error: {e}", "load audio data error")

        return AUDIO

    def _get_audio_pc(self):
        return self.AUDIO_mp

    def _get_eye_gaze_data(self, path_to_video_source: str, eye_gaze_model) -> Optional[Tuple[str, str]]:
        EYE_GAZE = []

        cap = cv2.VideoCapture(path_to_video_source)

        while cap.isOpened():
            try:
                success, frame = cap.read()
                if not success or frame is None:
                    break

                eye_gaze_model.refresh(frame)

                v_ratio = eye_gaze_model.vertical_ratio()
                h_ratio = eye_gaze_model.horizontal_ratio()

                horizontal, vertical = "", ""
                v_dir = self._get_direction(v_ratio)
                h_dir = self._get_direction(h_ratio)

                mp = {
                    "positive": {"y": "top", "x": "right"},
                    "zero": {"y": "center", "x": "center"},
                    "negative": {"y": "bottom", "x": "left"}
                }

                if v_dir:
                    vertical = mp[v_dir]["y"]
                if h_dir:
                    horizontal = mp[h_dir]["x"]

                if vertical != "" and horizontal != "":
                    EYE_GAZE.append((vertical, horizontal))

            except Exception as e:
                prettify_print(f"Error: {e}", "load eye gaze data error")

        cap.release()
        cv2.destroyAllWindows()

        return EYE_GAZE

    def _get_eye_gaze_pc(self):
        return self.EYE_GAZE_mp

    def _get_direction(self, ratio) -> Optional[str]:
        if ratio is None:
            return None
        if ratio <= 0.35 and ratio >= 0:
            return "positive"
        elif ratio > 0.35 and ratio < 0.65:
            return "zero"
        elif ratio >= 0.65 and ratio <= 1:
            return "negative"
        else:
            return None

    def _convert_list_to_mp(self, data: List) -> dict:  
        data_mp = Counter(data)
        data_mp['count'] = len(data)

        return dict(data_mp)