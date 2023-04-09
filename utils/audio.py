import subprocess
import os 

from pydub import AudioSegment
from mutagen.wave import WAVE

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

