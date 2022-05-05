import pickle
from abc import abstractmethod
import slider
import torchaudio
import shutil
import torch

from util import audio_util, beatmap_util

TEMP_WAV_FILE_PATH = r'E:\temp.wav'


class OsuAudioPreprocessor:
    """
    This class is intended to work together with OsuDB.
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
    def preprocess(self, beatmap: slider.Beatmap, path_from: str):
        """
        Takes path of an audio file(usually in osu! songs dir) and related information in `.osu`,
        preprocess it. Result of proprocessing is returned with a dict containing extra information
        during preprocessing is returned,
        which will also be saved in OsuDB.
        """
        raise NotImplementedError


class OsuAudioFilePreprocessor:
    """
    This class is intended to work together with OsuDB.
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
        preprocess it and save to a different path(usually one under control of OsuDB).
        a dict containing extra information during preprocessing is returned,
        which will also be saved in OsuDB.
        """
        raise NotImplementedError


class CopyPreprocessor(OsuAudioFilePreprocessor):

    def preprocess(self, beatmap: slider.Beatmap, path_from: str, path_to: str):
        """
        Simply copies file.
        """
        shutil.copy(path_from, path_to)
        return []


class ResamplePreprocessor(OsuAudioFilePreprocessor):
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


# class MelPreprocessor(OsuAudioPreprocessor):
#     EXTRA_COLUMNS = ['RESAMPLE_RATE']
#     EXTRA_COLUMNS_TYPE = ['INT']
#
#     def __init__(self, beat_feature_frames):
#         super().__init__()
#         self.beat_feature_frames = beat_feature_frames
#
#     def preprocess(self, beatmap, path_from):
#         """
#         This will ensure that audio have same number of frames for each beat after resampling.
#         Essential for beat/snap based classification.
#         """
#         bpm = beatmap.bpm_min()
#         resample_rate = round(self.beat_feature_frames * bpm / 60)  # * feature_frames_per_second
#         audio_data, sample_rate = audio_util.audioread_get_audio_data(path_from)
#         mel_spectrogram = torchaudio.transforms.MelSpectrogram(audio_data, sample_rate, resample_rate)
#         return [resample_rate]


class MelPreprocessor(OsuAudioFilePreprocessor):
    EXTRA_COLUMNS = ['RESAMPLE_RATE', 'CROP_START_TIME', 'SNAP_MEL', 'MEL_FRAME_TIME']
    EXTRA_COLUMNS_TYPE = ['REAL', 'REAL', 'INT', 'REAL']

    def __init__(self,
                 beat_feature_frames,
                 snap_offset=0,
                 n_fft=1024,
                 win_length=None,
                 hop_length=512,
                 n_mels=128,
                 from_resampled=True):
        """
        If from_resampled, in self.preprocess resampling is skipped and
        mel spectrogram is calculated directly from read audio data.
        """
        super().__init__()
        self.beat_feature_frames = beat_feature_frames
        self.snap_offset = snap_offset

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.from_resampled = from_resampled

    def preprocess(self, beatmap, path_from, path_to):
        """
        This will ensure that audio have same number of frames for each beat after resampling.
        Essential for beat/snap based classification.
        """
        audio_data, sample_rate = audio_util.audioread_get_audio_data(path_from)

        # do resampling if necessary
        if self.from_resampled:
            resample_rate = sample_rate
            resampled = audio_data
        else:
            bpm = beatmap.bpm_min()
            resample_rate = int(self.beat_feature_frames * bpm / 60)
            resampled = torchaudio.functional.resample(audio_data, sample_rate, resample_rate)

        # # intensity normalization
        # resampled /= torch.mean(resampled)

        snap_frames = self.beat_feature_frames // 8
        start_time = beatmap_util.get_first_hit_object_time_microseconds(beatmap)

        snap_frame_offset = round(snap_frames * self.snap_offset)
        # put snap to the center
        start_frame = round(start_time * resample_rate / 1000000) - snap_frame_offset

        # align with snap
        crop_start_frame = start_frame % snap_frames

        crop_end_frame = resampled.shape[1]
        # align with snap
        crop_end_frame = round((crop_end_frame - crop_start_frame - snap_frame_offset) / snap_frames) * snap_frames + crop_start_frame
        cropped = resampled[:, crop_start_frame:crop_end_frame]

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=resample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=self.n_mels,
            mel_scale="htk",
        )
        melspec = mel_spectrogram(cropped)
        # print(melspec.shape)
        with open(path_to, 'wb') as f:
            pickle.dump(melspec, f)

        crop_start_time = crop_start_frame / resample_rate * 1000000
        snap_mel = snap_frames // self.hop_length
        mel_frame_time = snap_frames / resample_rate * 1000000 / snap_mel
        print('snap_mel')
        print(snap_mel)
        print('mel_frame_time')
        print(mel_frame_time)
        return [resample_rate, crop_start_time, snap_mel, mel_frame_time]
