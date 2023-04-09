import os, shutil
import cv2
import random 
import itertools

from settings import PATHS
from typing import List, Dict, Any


class Choice:
    def __init__(self, options): self.options = options
    def get_sample(self, n: int) -> List: 
        return random.sample(self.options, n)


def generate_combinations(data: Dict[str, Choice]) -> List:
    categories = []

    for _, val in data.items():
        categories.append(val.options)
    
    return list(itertools.product(*categories))


def prep_folder(path_to_saving_directory: str): 
    if not os.path.isdir(path_to_saving_directory):
        os.mkdir(path_to_saving_directory)
    else: 
        for filename in os.listdir(path_to_saving_directory):
            file_path = os.path.join(path_to_saving_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_face_frames(path_to_video_source: str, path_to_saving_directory: str):
    prep_folder(path_to_saving_directory)

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


def prettify_print(item: Any, msg: str = ""): 
    print("\n====================================================================================================")
    print("====================================================================================================\n")
    print(f"****************************************** {msg} ****************************************\n")
    print(item)
    print("\n====================================================================================================")
    print("====================================================================================================\n")


def is_file_allowed(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'wav'}