from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class DATASET_TYPE(Enum):
    Train = 1
    Test = 2

class VIDEO_MODEL_PATH:
    JSON: str = "model/emotion_model.json"
    H5: str = "model/emotion_model.h5" 

class AUDIO_MODEL_PATH: 
    JSON: str 

class FER_PATH:
    TRAIN: str = "datasets/FER/train"
    TEST: str = "datasets/FER/test"

class EYE_REACTION_PATH:
    TRAIN: str = "datasets/Eye_Reaction/train"
    TEST: str = "datasets/Eye_Reaction/test"

class RAVDESS_PATH: 
    TRAIN: str = "datasets/RAVDESS"

class HAARCASCADES_PATH:
    frontal_face: str = "haarcascades/haarcascade_frontalface_default.xml" 
    eyes: str = "haarcascades/haarcascade_eye_tree_eyeglasses.xml"

class PATHS:
    VIDEO_MODEL = VIDEO_MODEL_PATH
    AUDIO_MODEL = AUDIO_MODEL_PATH
    FER = FER_PATH
    RAVDESS = RAVDESS_PATH
    EYE_REACTION = EYE_REACTION_PATH
    HAARCASCADES = HAARCASCADES_PATH


class CONSTANTS:
    TARGET_SIZE = 48
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 28700// 64
    EPOCH = 50
    VAL_STEPS = 7178 // 64
    Emotion_Catalogue = {
        0: "Angry", 
        1: "Disgusted", 
        2: "Fearful", 
        3: "Happy", 
        4: "Neutral", 
        5: "Sad", 
        6: "Surprised"}