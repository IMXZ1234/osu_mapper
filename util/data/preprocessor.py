from abc import abstractmethod
import slider
import torchaudio
import shutil

from util import audio_util

TEMP_WAV_FILE_PATH = r'E:\temp.wav'


class OsuTrainDataAudioPreprocessor:
    """
    This class is intended to work together with OsuTrainDB.
    Preprocess audio for train use. Since audio data size is quite large without compression and
    preprocessing is time consuming in many cases, we preprocess them in advance and save them back
    in compressed formats(usually as mp3 files, if training is done with time domain features).
    During training, preprocessed files are read on the fly in Datasets __getitem__.
    """
    EXTRA_COLUMNS = []
    EXTRA_COLUMNS_TYPE = []

    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, beatmap: slider.Beatmap, path_from: str, path_to: str):
        """
        Takes path of an audio file(usually in osu! songs dir) and related information in `.osu`,
        preprocess it and save to a different path(usually one under control of OsuTrainDB).
        a dict containing extra information during preprocessing is returned,
        which will also be saved in OsuTrainDB.
        """
        raise NotImplementedError


class CopyPreprocessor(OsuTrainDataAudioPreprocessor):

    def preprocess(self, beatmap: slider.Beatmap, path_from: str, path_to: str):
        """
        Simply copies file.
        """
        shutil.copy(path_from, path_to)
        return []


class ResamplePreprocessor(OsuTrainDataAudioPreprocessor):
    EXTRA_COLUMNS = ['RESAMPLE_RATE']
    EXTRA_COLUMNS_TYPE = ['INT']

    def __init__(self, beat_feature_frames):
        super(ResamplePreprocessor, self).__init__()
        self.beat_feature_frames = beat_feature_frames

    def preprocess(self, beatmap, path_from, path_to):
        """
        This will ensure that audio have same number of frames for each beat after resampling.
        Essential for beat/snap based classification.
        """
        bpm = beatmap.bpm_min()
        resample_rate = round(self.beat_feature_frames * bpm / 60)  # * feature_frames_per_second
        audio_data, sample_rate = audio_util.audioread_get_audio_data(path_from)
        resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, resample_rate)
        audio_util.save_audio(path_to, resampled_audio_data, resample_rate, temp_wav_file_path=TEMP_WAV_FILE_PATH)
        return [resample_rate]
