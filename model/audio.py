import librosa
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop 
        
from typing import List, Optional
from utils import Choice, generate_combinations, save_model, load_model
from settings import PATHS, CONSTANTS
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from os import listdir
from os.path import isfile, join


class Audio_Model:
    LABELS = {
        'gender': Choice(['male', 'female']),
        'emotion': Choice(['calm', 'happy', 'sad', 'angry', 'fearful'])
    }

    ENCODER = {
        0: "female_calm",
        1: "male_calm",
        2: "female_happy",
        3: "male_happy",
        4: "female_sad",
        5: "male_sad",
        6: "female_angry",
        7: "male_angry",
        8: "female_fearful",
        9: "male_fearful",
    }
    
    def __init__(self, model_name: Optional[str] = None, default: bool = True) -> None:
        self._default = default
        self._custom_model_name = model_name
        self.train_labels = ['_'.join(i) for i in generate_combinations(self.LABELS)] 

    def _get_audio_files(self) -> List:
        audio_files = []
        for folder in listdir(PATHS.RAVDESS.TRAIN):
            for f in listdir(join(PATHS.RAVDESS.TRAIN, folder)):
                if isfile(join(PATHS.RAVDESS.TRAIN, folder, f)):
                    audio_files.append(f)

        return audio_files
    
    def _preprocess(self):
        df = pd.DataFrame(columns=['feature'])
        bookmark = 0
        audio_files = self._get_audio_files()
        
        for f in audio_files: 
            audio_signal, sample_rate = librosa.load(
                f, 
                res_type='kaiser_fast', 
                duration=2.5,
                sr=22050*2,
                offset=0.5)

            mfccs = np.mean(librosa.feature.mfcc(
                y=audio_signal, 
                sr=np.array(sample_rate), 
                n_mfcc=40, 
                axis=0))
        

            df.loc[bookmark] = [mfccs]
            bookmark = bookmark + 1     

        temp = pd.concat(
            [
                pd.DataFrame(df['feature'].values.tolist()),
                pd.DataFrame(self.train_labels)
            ], axis=1).rename(index=str, columns={"0": "label"})

        
        rnewdf = shuffle(temp)
        rnewdf = rnewdf.fillna(0)

        return rnewdf

    def _split_train_test(self):
        df = self._preprocess()
        split = np.random.rand(len(df)) < 0.8
        train = df[split]
        test = df[~split]

        X_train = np.array(train.iloc[:, :-1])
        y_train = np.array(train.iloc[:, -1:])
        X_test = np.array(test.iloc[:, :-1])
        y_test = np.array(test.iloc[:, -1:])

        le = LabelEncoder()

        y_train = to_categorical(le.fit_transform(y_train))
        y_test = to_categorical(le.fit_transform(y_test))

        return (X_train, X_test, y_train, y_test)
    
    def _build_default_model(self) -> Sequential: 
        model = Sequential()

        model.add(Conv1D(256, 5, padding='same', input_shape=(216,1)))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(MaxPooling1D(pool_size=(8)))
        model.add(Conv1D(128, 5, padding='same',))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same',))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same',))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv1D(128, 5, padding='same',))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model

    def _initiate_custom_model(self):
        pass
    
    def _train_model(self, save: bool = True) -> None: 
        (X_train, X_test, y_train, y_test) = self._split_train_test()

        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        if self._default:
            emotion_model = self._build_default_model()
        else:
            emotion_model = self._initiate_custom_model(self._custom_model_name)

        emotion_model.compile(
            loss='categorical_crossentropy', 
            optimizer=RMSprop(lr=0.00001, decay=1e-6),
            metrics=['accuracy'])

        emotion_model.fit(
            x_traincnn, 
            y_train, 
            batch_size=CONSTANTS.BATCH_SIZE, 
            epochs=CONSTANTS.EPOCH, 
            validation_data=(x_testcnn, y_test))

        if save:
            save_model(
                model=emotion_model, 
                model_name=self._custom_model_name if self._custom_model_name != None else 'default_audio')

        return emotion_model

    def get_model(self, model_name: str = 'default') -> Sequential:
        model = load_model(model_name)
        
        if model: emotion_model = model
        else:
            emotion_model = self._train_model()

        return emotion_model
