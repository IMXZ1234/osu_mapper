import os
import pickle
from datetime import timedelta

import slider
import torch
import torchaudio
from torch.utils.data import Dataset

from util import audio_util, beatmap_util
from util.data import audio_osu_data


def hit_objects_to_label(beatmap, audio_start_time_offset, snap_per_microsecond, total_snap_num):
    # print(audio_start_time_offset)
    label = [0 for _ in range(total_snap_num)]
    for hit_obj in beatmap.hit_objects():
        snap_idx = (hit_obj.time / timedelta(microseconds=1) - audio_start_time_offset) * snap_per_microsecond
        # print('snap_idx')
        # print(snap_idx)
        label[round(snap_idx)] = 1
    return label


class CNNv1Dataset(Dataset):
    """
    Using index_file_path as reference.
    Load and convert mp3 file to wav on the fly to save disk space.
    Mel freq features are used.
    """
    def __init__(self, index_file_path, beatmap_obj_file_path=None, feature_frames_per_beat=512*4):
        super(CNNv1Dataset, self).__init__()
        self.audio_osu_data = audio_osu_data.AudioOsuData.from_path(index_file_path, beatmap_obj_file_path)
        # adjust to multiples of 8, which is common snap divisor value
        self.feature_frames_per_beat = feature_frames_per_beat - feature_frames_per_beat % 8

    def __len__(self):
        return len(self.audio_osu_data.audio_osu_list)

    @staticmethod
    def preprocess(audio_data, sample_rate, speed_stars, bpm, start_time, end_time,
                   feature_frames_per_beat=512, snap_divisor=8):
        print('preprocess bpm %f, first_beat %f, last_beat %f' % (bpm, start_time, end_time))
        snap_per_microsecond = bpm * snap_divisor / 60000000
        print('microsecond per snap')
        print(1 / snap_per_microsecond)

        valid_time_interval = [start_time, end_time]
        if valid_time_interval[0] < 0:
            valid_time_interval[0] = valid_time_interval[0] % (1 / snap_per_microsecond)
        if 0 < valid_time_interval[0] < 1 / snap_per_microsecond:
            audio_start_time_offset = valid_time_interval[0]
        else:
            audio_start_time_offset = valid_time_interval[0] % (1 / snap_per_microsecond)

        # Note that not the whole length of the audio is mapped,
        # that is, expected to be accompanied by hit_objects aligned with snaps.
        # The audio is mapped only after the first timing_point.
        # Crop the beginning of audio_data to align the start of audio_data with a snap,
        # which is not necessarily the first timing_point.
        # Even unmapped periods of audio may contain useful information for hit_object prediction.
        audio_start_frame = round(audio_start_time_offset * sample_rate / 1000000)
        audio_data = audio_data[:, audio_start_frame:]
        target_sample_rate = round(feature_frames_per_beat * bpm / 60)  # * feature_frames_per_second
        # alter length of audio with respective to bpm to ensure different audio have same number of frames for each
        # beat
        resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
        # crop to multiples of snaps
        feature_frames_per_snap = int(feature_frames_per_beat / snap_divisor)
        feature_frame_num = resampled_audio_data.shape[1]
        extra_frames = feature_frame_num % feature_frames_per_snap
        if extra_frames != 0:
            feature_frame_num = resampled_audio_data.shape[1] + feature_frames_per_snap - extra_frames
            new_resampled_audio_data = torch.zeros([resampled_audio_data.shape[0], feature_frame_num])
            new_resampled_audio_data[:, :resampled_audio_data.shape[1]] = resampled_audio_data
            resampled_audio_data = new_resampled_audio_data

        total_snap_num = feature_frame_num // feature_frames_per_snap
        print('total_snap_num feature')
        print(total_snap_num)
        # label is given for every snap, but only those in the valid interval which is mapped is counted in loss
        # calculation
        # last snap here is valid
        valid_interval = (round((valid_time_interval[0] - audio_start_time_offset) * snap_per_microsecond),
                          round((valid_time_interval[1] - audio_start_time_offset) * snap_per_microsecond) + 1)
        return [resampled_audio_data, speed_stars, valid_interval]

    def __getitem__(self, index):
        beatmap = self.audio_osu_data.beatmaps[index]
        audio_file_path = self.audio_osu_data.audio_osu_list[index][0]
        # # torchaudio do not support mp3, have to first convert audio file to wav
        # temp_wav_path = audio_file_path[:-4] + '.wav'
        # audio_util.audio_to(audio_file_path, temp_wav_path)
        # audio_data, sample_rate = torchaudio.backend.soundfile_backend.load(temp_wav_path)
        # os.remove(temp_wav_path)
        audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)

        snap_divisor = 8
        snap_per_microsecond = beatmap.bpm_min() * snap_divisor / 60000000

        valid_time_interval = [beatmap_util.get_first_hit_object_time_microseconds(beatmap), beatmap_util.get_last_hit_object_time_microseconds(beatmap)]
        # total_snap_num = round((valid_time_interval[1] - valid_time_interval[0]) * snap_per_microsecond) + 1
        # print('total_snap_num time')
        # print(total_snap_num)
        if valid_time_interval[0] < 0:
            valid_time_interval[0] = valid_time_interval[0] % (1 / snap_per_microsecond)
        if 0 < valid_time_interval[0] < 1 / snap_per_microsecond:
            # print('time before first timing_point shorter than a snap')
            audio_start_time_offset = valid_time_interval[0]
            # print('valid_time_interval')
            # print(valid_time_interval)
            # print('audio_start_time_offset')
            # print(audio_start_time_offset)
        else:
            audio_start_time_offset = valid_time_interval[0] % (1 / snap_per_microsecond)


        # Note that not the whole length of the audio is mapped,
        # that is, expected to be accompanied by hit_objects aligned with snaps.
        # The audio is mapped only after the first timing_point.
        # Crop the beginning of audio_data to align the start of audio_data with a snap,
        # which is not necessarily the first timing_point.
        # Even unmapped periods of audio may contain useful information for hit_object prediction.
        audio_start_frame = round(audio_start_time_offset * sample_rate / 1000000)
        audio_data = audio_data[:, audio_start_frame:]
        target_sample_rate = round(self.feature_frames_per_beat * beatmap.bpm_min() / 60)  # * feature_frames_per_minute
        # alter length of audio with respective to bpm to ensure different audio have same number of frames for each
        # beat
        resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
        # crop to multiples of snaps
        feature_frames_per_snap = int(self.feature_frames_per_beat / snap_divisor)
        feature_frame_num = resampled_audio_data.shape[1]
        extra_frames = feature_frame_num % feature_frames_per_snap
        if extra_frames != 0:
            feature_frame_num = resampled_audio_data.shape[1] + feature_frames_per_snap - extra_frames
            new_resampled_audio_data = torch.zeros([resampled_audio_data.shape[0], feature_frame_num])
            new_resampled_audio_data[:, :resampled_audio_data.shape[1]] = resampled_audio_data
            resampled_audio_data = new_resampled_audio_data
        total_snap_num = feature_frame_num // feature_frames_per_snap
        # print('total_snap_num feature')
        # print(total_snap_num)

        # label is given for every snap, but only those in the valid interval which is mapped is counted in loss
        # calculation
        # last snap here is valid
        valid_interval = (round((valid_time_interval[0] - audio_start_time_offset) * snap_per_microsecond),
                          round((valid_time_interval[1] - audio_start_time_offset) * snap_per_microsecond) + 1)
        # print('valid_interval')
        # print(valid_interval)
        label = hit_objects_to_label(beatmap, audio_start_time_offset, snap_per_microsecond, total_snap_num)
        # print('len(label)')
        # print(len(label))
        try:
            speed_stars = beatmap.speed_stars()
        except ZeroDivisionError:
            # slider may encounter ZeroDivisionError during star calculation
            # speed_stars have the closest relationship with hit_object determination
            # if speed_stars are unavailable, try overall stars or overall difficulty(this is always available)
            # higher the speed_stars, the more likely a hit_object exists at that snap
            try:
                speed_stars = beatmap.stars()
            except ZeroDivisionError:
                speed_stars = beatmap.overall_difficulty
        if valid_interval[1] > len(label):
            print('valid interval out of bound!')
            print(valid_interval)
            print(len(label))
        # print(beatmap.beatmap_set_id)
        return [resampled_audio_data, speed_stars, valid_interval], label, index


if __name__ == '__main__':
    beatmap = slider.Beatmap.from_path(r'C:\Users\asus\AppData\Local\osu!\Songs\beatmap-637787043532047046-27_ライムライトの残火piano_version\- - - (IMXZ123) [Hard].osu')
    objs = beatmap.hit_objects()

    print(objs[0].time)
    print(beatmap.hit_object_difficulty())
    audio_file_path = r'C:\Users\asus\AppData\Local\osu!\Songs\beatmap-637787043532047046-27_ライムライトの残火piano_version\audio.mp3'
    print(audio_file_path)
    temp_wav_path = audio_file_path[:-4] + '.wav'
    if not os.path.exists(temp_wav_path):
        audio_util.audio_convert(audio_file_path, temp_wav_path)
    audio_data, sample_rate = torchaudio.backend.soundfile_backend.load(temp_wav_path)
    print(audio_data.shape)
    audio_meta = torchaudio.backend.soundfile_backend.info(temp_wav_path)
    feature_frames_per_beat = 512
    target_sample_rate = round(feature_frames_per_beat * beatmap.bpm_min())
    resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
    print(resampled_audio_data.shape)
    n_fft = 64
    transform = torchaudio.transforms.MelSpectrogram(target_sample_rate, n_fft=n_fft, hop_length=1, n_mels=n_fft)
    mel = transform(resampled_audio_data)
    print(mel.shape)
    with open(r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\spectrogram\mel.pkl', 'wb') as f:
        pickle.dump(mel, f)
    # # time_domain_frames_per_beat = sample_rate / beatmap.bpm_min()
    # # target_sample_rate = round(sample_rate * self.feature_frames_per_beat / time_domain_frames_per_beat)
    # target_sample_rate = round(512 * beatmap.bpm_min())
    # # resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
    # resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, sample_rate * 2)
    # save_path = audio_file_path[:-4] + '_resampled2x.wav'
    # torchaudio.backend.soundfile_backend.save(save_path, resampled_audio_data, sample_rate)
    # resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, sample_rate / 2)
    # save_path = audio_file_path[:-4] + '_resampled0.5x.wav'
    # torchaudio.backend.soundfile_backend.save(save_path, resampled_audio_data, sample_rate)

        # n_fft = 64
        # transform = torchaudio.transforms.MelSpectrogram(target_sample_rate, n_fft=n_fft, hop_length=1)
        # mel = transform(resampled_audio_data)
        # print(mel.shape)

        # print(audio_meta)
        # time_domain_frames_per_beat = sample_rate / beatmap.bpm_min()
        # target_sample_rate = round(sample_rate * self.feature_frames_per_beat / time_domain_frames_per_beat)
