import cv2
import os
import shutil
import numpy as np
import pandas as pd
import librosa
from typing import List, Tuple
from settings import PATHS, CONSTANTS
from utils.utils import load_model, save_face_frames, convert_video_to_audio, SplitWavAudio
from model import Eye_Emotion_Model, Audio_Model
from random import randrange, randint


class Static_Analysis:
    VIDEO_MODEL_NAME = 'default_video'
    AUDIO_MODEL_NAME = 'default_audio'
    EYE_REACTION_MODEL_NAME = 'default_eye_reaction'

    VIDEO_RECORDING = PATHS.VIDEO_RECORDING
    AUDIO_RECORDING = PATHS.AUDIO_RECORDING

    def __init__(self) -> None:        
        self.video_model = load_model(self.VIDEO_MODEL_NAME)
        self.audio_model = load_model(self.AUDIO_MODEL_NAME)
        self.eye_reaction_model = load_model(self.EYE_REACTION_MODEL_NAME)
        self._preprocess()

    def _preprocess(self) -> None:
        save_face_frames(self.VIDEO_RECORDING, PATHS.FRAMES)
        convert_video_to_audio(self.AUDIO_RECORDING)

    def _get_data(self) -> Tuple[List, List, List]:
        return (self._get_FER_data(self.VIDEO_RECORDING, self.video_model),
                self._get_eye_emotion_data(self.eye_reaction_model), 
                self._get_audio_data(self.AUDIO_RECORDING, self.audio_model))
        
    # TO BE COMPLETED 
    def analyse(self) -> None:
        (FER_data, EYE_EMOTION_data, AUDIO_data) = self._get_data()
        pass

    def _get_FER_data(self, path_to_video_source: str, video_model) -> List:
        FER = []

        cap = cv2.VideoCapture(path_to_video_source)

        while cap.isOpened():
            success, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not success: break 

            face_detector = cv2.CascadeClassifier(PATHS.HAARCASCADES.frontal_face)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

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

            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

        return FER

    def _get_eye_emotion_data(self, eye_reaction_model) -> List:
        eye_emotion = Eye_Emotion_Model()

        images = []
        for image in os.listdir(PATHS.FRAMES):
            try:
                image = cv2.imread(os.path.join(PATHS.FRAMES, image))
                image = cv2.resize(image, eye_emotion._img_size)
                images.append(image)

            except Exception as e:
                pass
        
        images = np.array(images)

        features = eye_emotion.get_features(images)

        pred = eye_reaction_model.predict(features)
        y_decode = np.argmax(pred, axis=1)

        for idx in [randrange(len(y_decode)) for i in range(0, len(y_decode)*0.05+1)]:
            y_decode[idx] = randint(0, 5)

        return [eye_emotion.ENCODER[i] for i in y_decode]

    def _get_audio_data(self, path_to_audio_source: str, audio_model) -> List:
        if os.path.isdir(PATHS.AUDIO_OUTPUT):
            if len(os.listdir(PATHS.AUDIO_OUTPUT)) != 0:
                for filename in os.listdir(PATHS.AUDIO_OUTPUT):
                    file_path = os.path.join(PATHS.AUDIO_OUTPUT, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            os.mkdir(PATHS.AUDIO_OUTPUT)

        split_wav = SplitWavAudio(PATHS.AUDIO_OUTPUT, path_to_audio_source)
        split_wav.multiple_split(sec_per_split=3)

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
                    pred = audio_model.predict(flat_data, batch_size=32)
                    y_decode = np.argmax(pred, axis=1)
                    AUDIO.append(Audio_Model().ENCODER[y_decode[0]])

            except Exception as e: 
                continue
        
        return AUDIO