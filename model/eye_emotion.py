import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Lambda, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical 

from typing import Optional, Tuple, List, Any
from utils import save_model, load_model
from settings import PATHS, CONSTANTS
from sklearn.preprocessing import LabelEncoder


class Eye_Emotion_Model:
    ENCODER = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise'
    }
    
    def __init__(
            self, 
            model_name: Optional[str] = None, 
            default: bool = True,
            is_gray: bool = False,
            img_size: Tuple[int, int] = (350, 350)
        )-> None:
        self._default = default
        self._custom_model_name = model_name

        self._is_gray = is_gray
        self._img_size = img_size
    
    def _load_images(self, path: str) -> Tuple[List, List]:
        images = []
        labels = []

        for emotion in os.listdir(path):
            _p = os.path.join(path, emotion)

            for image in os.listdir(_p):
                if self._is_gray: 
                    image = cv2.imread(os.path.join(_p, image), cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(os.path.join(_p, image))
                image = cv2.resize(image,self._img_size)
                images.append(image)
                labels.append(emotion)

            if self._is_gray:
                images = np.array(images)
                labels = np.array(labels)
                images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
                images = images.astype("float")/255  

            return images, labels
    
    def _load_data(self) -> Tuple[List, List, List, List]:
        train_x, train_y = self._load_images(PATHS.EYE_REACTION.TRAIN)
        test_x, test_y = self._load_images(PATHS.EYE_REACTION.TEST)
            
        return train_x, train_y, test_x, test_y
    
    def _get_features(self, keras_model, data, width=350) -> Any:
        cnn_model = keras_model(include_top = False, input_shape=(width, width, 3), weights='imagenet')

        inputs = Input((width, width, 3))
        cnn_model = Model(
            inputs, 
            GlobalAveragePooling2D()(cnn_model(Lambda(preprocess_input, name='preprocessing')(inputs))))
        
        return cnn_model.predict(data, batch_size=5, verbose=1)
    
    def _build_default_model(self, features) -> Sequential:
        model = Sequential()
        
        model.add(Dense(1020,activation= "relu",input_shape=(features.shape[1],)))
        model.add(Dense(900,activation = "relu"))
        model.add(Dense(800,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(6,activation="softmax"))

        return model

    def _initiate_custom_model(self):
        pass

    def _split_train_test(self):
        train_x, train_y, test_x, test_y = self._load_data()   

        train_y = np.array(train_y)
        train_x = np.array(train_x)
        test_y = np.array(test_y)
        test_x = np.array(test_x)

        le = LabelEncoder()

        train_y = to_categorical(le.fit_transform(train_y))
        test_y = to_categorical(le.fit_transform(test_y))

        return (train_x, test_x, train_y, test_y)

    def get_features(self, data):
        return np.concatenate([
            self._get_features(InceptionV3, data), 
            self._get_features(Xception, data)
            ], axis=1) 

    def _train_model(self, save: bool = True) -> None:
        (train_x, train_y, test_x, test_y) = self._split_train_test()

        features = self.get_features(train_x)

        if self._default:
            emotion_model = self._build_default_model(features)
        else:
            emotion_model = self._initiate_custom_model(self._custom_model_name)

        emotion_model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
            metrics=['accuracy'])

        emotion_model.fit(
            features,
            train_y,
            batch_size=CONSTANTS.BATCH_SIZE,
            epochs=CONSTANTS.EPOCH,
            validation_data=(test_x, test_y))
        
        if save:
            save_model(
                model=emotion_model, 
                model_name=self._custom_model_name if self._custom_model_name != None else 'default_video')

        return emotion_model

    def get_model(self, model_name: str = 'default') -> Sequential:
        model = load_model(model_name)
        
        if model: emotion_model = model
        else:
            emotion_model = self._train_model()

        return emotion_model
