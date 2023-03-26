import os
import cv2
import random 
import itertools
import subprocess
import mutagen
from settings import PATHS
from typing import Optional, List, Dict, Any
from pydub import AudioSegment
from mutagen.wave import WAVE
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json


def save_model(model: Sequential, model_name: str) -> None: 
    model_json = model.to_json()
    with open(f"saved_model/{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"saved_model/{model_name}.h5")


def load_model(model_name: str = 'default') -> Optional[Sequential]:
    emotion_model = None
    if (
        os.path.exists(f'saved_model/{model_name}.json') and 
        os.path.exists(f'saved_model/{model_name}.h5')
    ):
        json_file = open(f'saved_model/{model_name}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights(f"saved_model/{model_name}.h5")

    return emotion_model


class Choice:
    def __init__(self, options): self.options = options
    def get_sample(self, n: int) -> List: 
        return random.sample(self.options, n)


def generate_combinations(data: Dict[str, Choice]) -> List:
    categories = []

    for _, val in data.items():
        categories.append(val.options)
    
    return list(itertools.product(*categories))


def save_face_frames(path_to_video_source: str, path_to_saving_directory: str):
    cap = cv2.VideoCapture(path_to_video_source)
    success, frame = cap.read()
    count = 0

    face_detector = cv2.CascadeClassifier(PATHS.HAARCASCADES.frontal_face)

    while success:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color_frame = frame[y: y + h, x: x + w]
            
            cv2.imwrite(
                f"{path_to_saving_directory}/frame_{count}.jpg",
                roi_color_frame)
            
        success, frame = cap.read()
        count += 1


def convert_video_to_audio(path_to_video_source: str, output_ext: str = "wav") -> None:
    filename, _ = os.path.splitext(path_to_video_source)
    subprocess.call(
        ["ffmpeg", "-y", "-i", path_to_video_source, f"{filename}.{output_ext}"], 
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)


def get_audio_length(path_to_audio_source: str) -> int: 
    audio = WAVE(path_to_audio_source)
    audio_info = audio.info
    return int(round(audio_info.length))


class SplitWavAudio():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        
        self.audio = AudioSegment.from_wav(self.filename)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_, to_, split_filename):
        t1 = from_ * 1000
        t2 = to_ * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total = round(self.get_duration())
        for i in range(0, total, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)


def prettify_print(item: Any, msg: str): 
    print("\n====================================================================================================")
    print("====================================================================================================\n")
    print(f"*********************************************** {msg} **********************************************\n")
    print(item)

    print("\n====================================================================================================")
    print("====================================================================================================\n")