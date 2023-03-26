import cv2

from typing import Optional
from settings import CONSTANTS, PATHS
from utils.utils import save_model, load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import MobileNetV2, InceptionV3


class Video_Model:
    def __init__(self, model_name: Optional[str], default: bool = True) -> None: 
        self._default = default
        self._custom_model_name = model_name
            
    def _build_default_model(self) -> Sequential:
        emotion_model = Sequential()
        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(0.5))
        emotion_model.add(Dense(7, activation='softmax'))
        
        return emotion_model

    def _initiate_custom_model(self, model_name: str) -> Sequential:
        if model_name == "MobileNetV2":
            emotion_model = MobileNetV2()
        elif model_name == "InceptionV3":
            emotion_model = InceptionV3()

        return emotion_model

    def _preprocess_generators(self):
        train_data_gen = ImageDataGenerator(rescale=1./255)
        validation_data_gen = ImageDataGenerator(rescale=1./255)

        train_generator = train_data_gen.flow_from_directory(
            PATHS.FER.TRAIN,
            target_size=(CONSTANTS.TARGET_SIZE, CONSTANTS.TARGET_SIZE),
            batch_size=CONSTANTS.BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')
        
        validation_generator = validation_data_gen.flow_from_directory( 
            PATHS.FER.TEST,
            target_size=(CONSTANTS.TARGET_SIZE, CONSTANTS.TARGET_SIZE),
            batch_size=CONSTANTS.BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')

        return (train_generator, validation_generator)
    
    def _train_model(self, save: bool = True) -> None: 
        train_generator, validation_generator = self._preprocess_generators()

        if self._default:
            emotion_model = self._build_default_model()
        else:
            emotion_model = self._initiate_custom_model(self._custom_model_name)

        cv2.ocl.setUseOpenCL(False)

        emotion_model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
            metrics=['accuracy'])

        emotion_model.fit(
            train_generator,
            steps_per_epoch=CONSTANTS.STEPS_PER_EPOCH,
            epochs=CONSTANTS.EPOCH,
            validation_data=validation_generator,
            validation_steps=CONSTANTS.VAL_STEPS)

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