import os

from typing import Optional

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