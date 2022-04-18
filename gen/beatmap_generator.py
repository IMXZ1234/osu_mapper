from abc import abstractmethod


class BeatmapGenerator:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path

    @abstractmethod
    def generate(self):
        raise NotImplementedError
