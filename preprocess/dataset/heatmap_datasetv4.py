import sys

sys.argv.append(r'/home/xiezheng/osu_mapper')
import os
import pickle
import traceback
import zipfile
import json
from datetime import timedelta
import numpy as np

import slider
import torch
import torchaudio
import multiprocessing
import math

import audio_util
from plt_util import plot_signal
import itertools


def calculate_meta(beatmap: slider.Beatmap, frame_times):
    """
    ms_per_beat, snap_divisor, ho_density, occupied_proportion
    """
    total_frames = len(frame_times)
    # same length as mel frame data
    occupied_proportion = np.zeros([total_frames])
    ho_density = np.zeros([total_frames])
    ms_per_beat = np.zeros([total_frames])
    frame_ms = frame_times[1] - frame_times[0]

    non_inherited_tps = [tp for tp in beatmap.timing_points if tp.parent is None]
    ho_idx = 0
    # statistics of hit_objects under control of timing_points
    # a dummy timing_points is added at the beginning
    # [total_snaps, hit_object_num, occupied_snap_num]
    # a single hit_object Slider/Spinner may occupy multiple snaps
    tp_statistics_list = [[0, 0, 0]] + [[0, 0, 0] for tp in non_inherited_tps]
    for tp_idx, (tp1, tp2) in enumerate(zip([None] + non_inherited_tps, non_inherited_tps + [None])):
        if tp1 is None:
            tp_ms_per_snap = tp2.ms_per_beat / beatmap.beat_divisor
            start_ms = 0
            end_ms = tp2.offset / timedelta(milliseconds=1)
        elif tp2 is None:
            tp_ms_per_snap = tp1.ms_per_beat / beatmap.beat_divisor
            start_ms = tp1.offset / timedelta(milliseconds=1)
            end_ms = frame_times[-1]
        else:
            tp_ms_per_snap = tp1.ms_per_beat / beatmap.beat_divisor
            start_ms = tp1.offset / timedelta(milliseconds=1)
            end_ms = tp2.offset / timedelta(milliseconds=1)
        tp_ms_itv = end_ms - start_ms
        cur_tp_statistics = tp_statistics_list[tp_idx]
        cur_tp_statistics[0] = round(tp_ms_itv / tp_ms_per_snap)
        # count hit_objects under control of this timing_point
        while ho_idx < len(beatmap._hit_objects):
            ho = beatmap._hit_objects[ho_idx]
            ho_time = ho.time / timedelta(milliseconds=1)
            if ho_time > end_ms:
                break
            if isinstance(ho, slider.beatmap.Circle):
                cur_tp_statistics[1] += 1
                cur_tp_statistics[2] += 1
            else:
                assert isinstance(ho, (slider.beatmap.Slider, slider.beatmap.Spinner))
                ho_end_time = ho.end_time / timedelta(milliseconds=1)
                cur_tp_statistics[1] += 1
                ho_snaps = round((ho_end_time - ho_time) / tp_ms_per_snap)
                cur_tp_statistics[2] += ho_snaps
            ho_idx += 1

    for tp_idx, (tp1, tp2) in enumerate(zip([None] + non_inherited_tps, non_inherited_tps + [None])):
        if tp1 is None:
            tp_ms_per_beat = tp2.ms_per_beat
            start_mel = 0
            end_mel = round(tp2.offset / timedelta(milliseconds=1) / frame_ms)
            cur_tp_statistics = tp_statistics_list[tp_idx + 1]
        elif tp2 is None:
            tp_ms_per_beat = tp1.ms_per_beat
            start_mel = round(tp1.offset / timedelta(milliseconds=1) / frame_ms)
            end_mel = len(frame_times)
            cur_tp_statistics = tp_statistics_list[tp_idx]
        else:
            tp_ms_per_beat = tp1.ms_per_beat
            start_mel = round(tp1.offset / timedelta(milliseconds=1) / frame_ms)
            end_mel = round(tp2.offset / timedelta(milliseconds=1) / frame_ms)
            cur_tp_statistics = tp_statistics_list[tp_idx]
        occupied_proportion[start_mel:end_mel] = cur_tp_statistics[2] / cur_tp_statistics[0] if cur_tp_statistics[
                                                                                                    0] != 0 else 0
        ho_density[start_mel:end_mel] = cur_tp_statistics[1] / cur_tp_statistics[0] if cur_tp_statistics[0] != 0 else 0
        ms_per_beat[start_mel:end_mel] = tp_ms_per_beat
    return ms_per_beat, ho_density, occupied_proportion, beatmap.beat_divisor


def calculate_hitobject_meta(beatmap: slider.Beatmap, frame_times):
    """
    -> 5, L
    meta information of hit objects
    """
    total_frames = len(frame_times)
    # hit_object count within each mel frame
    # sliders or spinners are counted only once in the closest mel frame
    # circle count, slider count, spinner count, slider occupied, spinner occupied
    ho_meta = np.zeros([5, total_frames])
    frame_bounds = (frame_times[:-1] + frame_times[1:]) / 2
    itv = np.diff(frame_bounds)
    itv = np.concatenate([np.array([frame_bounds[0]]), itv, itv[-1:]])
    frame_idx = 0
    for ho in beatmap._hit_objects:
        time_start = ho.time // timedelta(milliseconds=1)
        while frame_idx < len(frame_bounds) and time_start > frame_bounds[frame_idx]:
            frame_idx += 1
        if isinstance(ho, slider.beatmap.Circle):
            ho_meta[0, frame_idx] += 1
        else:
            time_end = ho.end_time // timedelta(milliseconds=1)
            if isinstance(ho, slider.beatmap.Slider):
                ho_meta[1, frame_idx] += 1
                i = 3
            elif isinstance(ho, slider.beatmap.Spinner):
                ho_meta[2, frame_idx] += 1
                i = 4
            accumulate_start = time_start
            while frame_idx < len(frame_bounds) and time_end > frame_bounds[frame_idx]:
                length_within_frame = frame_bounds[frame_idx] - accumulate_start
                ho_meta[i, frame_idx] += length_within_frame
                accumulate_start = frame_bounds[frame_idx]
                frame_idx += 1
            ho_meta[i, frame_idx] += time_end - accumulate_start
    ho_meta[3, :] = ho_meta[3, :] / itv
    ho_meta[4, :] = ho_meta[4, :] / itv
    return ho_meta, beatmap.beat_divisor


def plot_label(label):
    for signal, name in zip(
            label.T,
            [
                'signal_type', 'cursor_x', 'cursor_y'
            ]
    ):
        plot_signal(signal, name)
    for signal, name in zip(
            label.T,
            [
                'signal_type', 'cursor_x', 'cursor_y'
            ]
    ):
        plot_signal(signal[:512], name)


def plot_meta(meta_data):
    for signal, name in zip(
            meta_data,
            [
                'circle_count', 'slider_count', 'spinner_count', 'slider_occupy', 'spinner_occupy'
            ]
    ):
        plot_signal(signal, name)
    # for signal, name in zip(
    #         meta_data,
    #         [
    #             'circle_count', 'slider_count', 'spinner_count', 'slider_occupy', 'spinner_occupy'
    #         ]
    # ):
    #     plot_signal(signal[:5120], name)


def check_integrity(file_path):
    try:
        # check if downloaded beatmapsets are corrupted
        f = zipfile.ZipFile(file_path, 'r', )
        # print(f.namelist())
        for fn in f.namelist():
            f.read(fn)
    except Exception:
        return False
    return True


def check_beatmap_suitable(beatmap: slider.Beatmap, beat_divisor=8):
    if beat_divisor % beatmap.beat_divisor != 0:
        return False
    assert beatmap.timing_points[0].parent is None
    for tp in beatmap.timing_points[1:]:
        if tp.parent is None:
            return False
    if len(beatmap._hit_objects) <= 1:
        return False
    return True


def cal_mel_spec(audio_data,
                 frame_start, frame_length,
                 window='hamming',
                 nfft=None, n_mel=40, sample_rate=22000):
    """
    calculate mel spec for frames starting from specified pos
    audio_data: ..., sample
    return time, ..., n_mel
    """
    if nfft is None:
        nfft = frame_length
    frames = np.stack([audio_data[..., start:start + frame_length] for start in frame_start], axis=0)
    if window == 'hamming':
        frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, nfft, axis=-1))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mel + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_mel, int(np.floor(nfft / 2 + 1))))
    for m in range(1, n_mel + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    # filter_banks_shape = list(pow_frames.shape)
    # filter_banks_shape[-1] = n_mel
    # filter_banks = np.zeros(filter_banks_shape)
    # for n in range(len(frames)):
    #     filter_banks[n, ...] = np.dot(pow_frames[n, ...], fbank.T)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks


class HeatmapDataset:
    """
    Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames.
    Popular settings are **25 ms** for the frame size, frame_size = 0.025 and a 10 ms stride (15 ms overlap), frame_stride = 0.01.

    commonly seen bpm is within 60 ~ 120
    with beat_divisor 4, snap per minute is 240 ~ 480,
    a typical snap therefore takes 250ms ~ 125ms, and can hold 10 ~ 20 mel frames (with stride of half window length)
    """
    def __init__(self, mel_args, feature_args):
        self.mel_args = mel_args
        self.feature_args = feature_args

        self.sample_rate = mel_args.get('sample_rate', 22000)
        self.mel_frame_per_snap = mel_args.get('mel_frame_per_snap', 16)
        self.n_mels = mel_args.get('n_mels', 40)

        self.beat_divisor = feature_args.get('beat_divisor', 8)
        self.off_snap_threshold = feature_args.get('off_snap_threshold', 0.15)

        self.label_idx_to_beat_label_seq = list(itertools.product(*[(0, 1, 2, 3) for _ in range(self.beat_divisor)]))
        self.beat_label_seq_to_label_idx = {seq: idx for idx, seq in enumerate(self.label_idx_to_beat_label_seq)}
        self.label_idx_count = []

    def from_meta_file(self, save_dir, meta_filepath, osz_dir,
                       temp_dir='.'
                       ):
        """
        will save processed label under data_dir
        """

        with open(os.path.join(meta_filepath), 'r') as f:
            meta_dict_list = json.load(f)
        for sample_idx, beatmap_meta_dict in enumerate(meta_dict_list[3:4]):
            self.process_beatmap_meta_dict(
                beatmap_meta_dict, save_dir, osz_dir, temp_dir, skip_exist=False,
            )

    def process_beatmap_meta_dict(self, beatmap_meta_dict, save_dir, osz_dir, temp_dir, skip_exist=True):
        # print('in process')
        beatmap_id, beatmapset_id = beatmap_meta_dict['beatmap_id'], beatmap_meta_dict['beatmapset_id']
        dirs, paths, processed = {}, {}, {}
        items = {
            'mel': beatmapset_id,
            'meta': beatmap_id,
            'label': beatmap_id,
            'label_idx': beatmap_id,
            'info': beatmap_id,
        }
        for item_name, sample_name in items.items():
            dirs[item_name] = os.path.join(save_dir, item_name)
            os.makedirs(dirs[item_name], exist_ok=True)
            paths[item_name] = os.path.join(dirs[item_name], sample_name + '.pkl')
            processed[item_name] = os.path.exists(paths[item_name])
        if skip_exist and all(os.path.exists(path) for path in paths.values()):
            print('skipping %s' % beatmap_id)
            return

        osz_path = os.path.join(osz_dir, beatmapset_id + '.osz')
        if not os.path.exists(osz_path):
            print('osz not exist!')
            return
        if not check_integrity(osz_path):
            print('osz corrupted %s' % beatmapset_id)
            return

        osz_file = zipfile.ZipFile(osz_path, 'r')
        try:
            all_beatmaps = list(slider.Beatmap.from_osz_file(osz_file).values())
        except Exception:
            print('beatmap parse failed %s' % beatmapset_id)
            return
        # print([(bm.beatmap_id, bm.beatmap_set_id) for bm in all_beatmaps])
        beatmap = None
        for bm in all_beatmaps:
            if bm.version == beatmap_meta_dict['version'] or bm.beatmap_id == int(beatmap_id):
                beatmap = bm
        if beatmap is None:
            print('beatmap %s not found in osz %s' % (beatmap_id, osz_path))
            return

        if not check_beatmap_suitable(beatmap, self.beat_divisor):
            print('beatmap not suitable %s' % beatmap_id)
            return

        bpm = beatmap.timing_points[0].bpm
        beat_length_s = 60 / bpm
        snap_length_s = beat_length_s / self.beat_divisor
        snap_length_samples_f = snap_length_s * self.sample_rate
        beat_length_samples_f = beat_length_s * self.sample_rate
        hop_length_s = snap_length_s / self.mel_frame_per_snap
        hop_length_samples_f = hop_length_s * self.sample_rate
        hop_length = math.ceil(hop_length_samples_f)
        # since we align mel frames with snaps,
        # the center of the first mel frame should depend on temporal position of the first timing point
        first_tp_s = beatmap.timing_points[0].offset / timedelta(seconds=1)
        first_tp_sample_f = first_tp_s * self.sample_rate

        first_ho = beatmap._hit_objects[0]
        last_ho = beatmap._hit_objects[-1]
        # time offset relative to tp
        occupied_start_s = first_ho.time / timedelta(seconds=1)
        if isinstance(last_ho, slider.beatmap.Circle):
            occupied_end_s = last_ho.time / timedelta(seconds=1)
        else:
            occupied_end_s = last_ho.end_time / timedelta(seconds=1)
        # hit objects should be well aligned with snaps
        # occupied_totol_beats_f = (occupied_end_s - occupied_start_s) / beat_length_s
        occupied_totol_snaps_f = (occupied_end_s - occupied_start_s) / snap_length_s
        occupied_start_snaps_f = (occupied_start_s - first_tp_s) / snap_length_s
        if abs(occupied_start_snaps_f - round(occupied_start_snaps_f)) > 0.15:
            print('hit objects not well aligned %s' % beatmap_id)
            # print(occupied_start_snaps_f, occupied_start_snaps_f)
            return
        if abs(occupied_totol_snaps_f - round(occupied_totol_snaps_f)) > 0.15:
            print('hit objects not well aligned %s' % beatmap_id)
            # print('beat_length_s', beat_length_s)
            # print('snap_length_s', snap_length_s)
            # print('first_tp_s', first_tp_s)
            # print('occupied_start_s', occupied_start_s)
            # print('occupied_end_s', occupied_end_s)
            # print('occupied_totol_beats_f', occupied_totol_beats_f)
            # print('occupied_totol_snaps_f', occupied_totol_snaps_f)
            return

        print('processing beatmap %s' % beatmap_id)
        # retrieve audio data
        audio_filename = all_beatmaps[0].audio_filename

        for beatmap in all_beatmaps:
            if beatmap.audio_filename != audio_filename:
                print('multiple audio file in same beatmapset!')
                continue

        on_disk_audio_filepath = os.path.join(temp_dir, beatmapset_id + os.path.splitext(audio_filename)[1])
        try:
            with open(on_disk_audio_filepath, 'wb') as f:
                f.write(osz_file.read(audio_filename))
        except Exception:
            print('no audio %s' % audio_filename)
            return
        try:
            audio_data, sr = audio_util.audioread_get_audio_data(on_disk_audio_filepath, ret_tensor=False)
        except Exception:
            print('fail to load audio %s' % beatmapset_id)
            return
        os.remove(on_disk_audio_filepath)

        total_sample = audio_data.shape[1]
        # count by beat_length_samples_f backward and forward to retrieve aligned audio
        beats_before_first_tp = math.floor(first_tp_sample_f / beat_length_samples_f)
        crop_start_sample_f = first_tp_sample_f - beats_before_first_tp * beat_length_samples_f
        snaps_before_first_tp = beats_before_first_tp * self.beat_divisor
        beats_after_first_tp = math.floor((total_sample - first_tp_sample_f) / beat_length_samples_f)
        crop_end_sample_f = first_tp_sample_f + beats_after_first_tp * beat_length_samples_f
        # crop_start_sample, crop_end_sample = round(crop_start_sample_f), round(crop_end_sample_f)

        total_beats_f = (crop_end_sample_f - crop_start_sample_f) / beat_length_samples_f
        total_snaps_f = total_beats_f * self.beat_divisor
        total_snaps = round(total_snaps_f)
        if abs(total_snaps - total_snaps_f) > 0.15:
            print('hit objects not well aligned %s' % beatmap_id)
            # print('total_beats_f', total_beats_f)
            # print('total_snaps_f', total_snaps_f)
            # print('total_snaps', total_snaps)
            return

        total_mel_frames = total_snaps * self.mel_frame_per_snap

        """
        process mel
        """
        if not (skip_exist and processed['mel']):
            try:
                # mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                #     sample_rate=self.sample_rate,
                #     n_fft=2 * hop_length,
                #     win_length=2 * hop_length,
                #     hop_length=hop_length,
                #     # center=True,
                #     # pad_mode="reflect",
                #     # we shall satisfy "center" by padding outside this function manually
                #     center=False,
                #     pad_mode="reflect",
                #     power=2.0,
                #     norm="slaney",
                #     onesided=True,
                #     n_mels=self.n_mels,
                #     mel_scale="htk",
                # )
                # # pad left and right with half window length = hop length
                # audio_for_mel = audio_data[:, max(0, crop_start_sample-hop_length):min(total_sample, crop_end_sample+hop_length)]
                # pad_left = hop_length - min(crop_start_sample, hop_length)
                # if pad_left > 0:
                #     audio_for_mel = torch.cat([torch.zeros([2, pad_left]), audio_for_mel], dim=1)
                # pad_right = hop_length - min(total_sample - crop_end_sample, hop_length)
                # if pad_right > 0:
                #     audio_for_mel = torch.cat([audio_for_mel, torch.zeros([2, pad_right])], dim=1)

                # # channel, n_mels, T -> T, n_mels
                # mel_spec = torch.mean(mel_spectrogram(audio_for_mel), dim=0).numpy().T
                frame_length = round(2 * hop_length_samples_f)
                frame_start_f = np.linspace(crop_start_sample_f - hop_length_samples_f, crop_end_sample_f - hop_length_samples_f, total_mel_frames, endpoint=False)
                frame_start = np.round(frame_start_f).astype(int)
                audio_for_mel = audio_data
                # pad and crop left
                if frame_start[0] < 0:
                    audio_for_mel = np.concatenate([np.zeros([2, -frame_start[0]]), audio_for_mel], axis=1)
                else:
                    audio_for_mel = audio_for_mel[..., frame_start[0]:]
                frame_start -= frame_start[0]
                # pad right, generally this will not happen
                if frame_start[-1] + frame_length > audio_for_mel.shape[1]:
                    audio_for_mel = np.concatenate([audio_for_mel, np.zeros([2, frame_start[-1] + frame_length - audio_for_mel.shape[1]])], axis=1)
                else:
                    audio_for_mel = audio_for_mel[..., :frame_start[-1] + frame_length]
                mel_spec = cal_mel_spec(
                    audio_for_mel,
                    frame_start, frame_length,
                    window='hamming',
                    nfft=frame_length,
                    n_mel=frame_length,
                    sample_rate=self.sample_rate
                )
                mel_spec = np.mean(mel_spec, axis=1)
                assert mel_spec.shape[0] == total_mel_frames, 'mel_spec.shape[0] %s != total_mel_frames %s' % (mel_spec.shape[0], total_mel_frames)
            except Exception:
                traceback.print_exc()
                print('failed process_mel %s_%s' % (beatmapset_id, beatmap_id))
                return
            # mel_spec = self.add_populated_indicator(mel_spec, beatmap)
            with open(paths['mel'], 'wb') as f:
                pickle.dump(mel_spec, f)

        def get_ho_pos_snap(ho_, end_time=False):
            if end_time and isinstance(ho_, (slider.beatmap.Slider, slider.beatmap.Spinner)):
                ho_pos_s_ = ho_.end_time / timedelta(seconds=1)
            else:
                ho_pos_s_ = ho_.time / timedelta(seconds=1)
            ho_pos_snap_f_ = (ho_pos_s_ - first_tp_s) / snap_length_s
            ho_pos_snap_ = round(ho_pos_snap_f_)
            # assert strict alignment between ho and snaps
            # more lenient
            assert abs(ho_pos_snap_f_ - ho_pos_snap_) < 0.15
            ho_pos_snap_ += snaps_before_first_tp
            return ho_pos_snap_

        if not (skip_exist and processed['label'] and processed['label_idx']):
            """
            process label
            """
            try:
                # snap type
                # no hitobject: 0
                # circle: 1
                # slider: 2
                # spinner: 3
                snap_type = np.zeros(total_snaps)
                for i, ho in enumerate(beatmap._hit_objects):
                    # print(i)
                    ho_pos_snap = get_ho_pos_snap(ho, end_time=False)
                    if isinstance(ho, slider.beatmap.Circle):
                        snap_type[ho_pos_snap] = 1
                    else:
                        ho_end_pos_snap = get_ho_pos_snap(ho, end_time=True)
                        if isinstance(ho, slider.beatmap.Slider):
                            snap_type[ho_pos_snap: ho_end_pos_snap+1] = 2
                        else:
                            assert isinstance(ho, slider.beatmap.Spinner)
                            snap_type[ho_pos_snap: ho_end_pos_snap+1] = 3
                # cursor signal
                x_pos_seq, y_pos_seq = np.zeros(total_snaps), np.zeros(total_snaps)
                for ho_a, ho_b in zip([None] + beatmap._hit_objects, beatmap._hit_objects + [None]):
                    if ho_a is None:
                        # before first hit object
                        # start point for following interpolation
                        ho_a_end_pos_snap = 0
                        ho_a_end_x = ho_b.position.x
                        ho_a_end_y = ho_b.position.y
                    else:
                        ho_a_pos_snap = get_ho_pos_snap(ho_a, end_time=False)
                        if isinstance(ho_a, slider.beatmap.Circle):
                            ho_a_end_pos_snap = ho_a_pos_snap
                            ho_a_end_x = ho_a.position.x
                            ho_a_end_y = ho_a.position.y
                        else:
                            # calculate cursor pos within span of ho_a
                            ho_a_end_pos_snap = get_ho_pos_snap(ho_a, end_time=True)
                            # end snap is not counted
                            ho_span_snaps = ho_a_end_pos_snap - ho_a_pos_snap
                            if isinstance(ho_a, slider.beatmap.Slider):
                                assert ho_span_snaps % ho_a.repeat == 0
                                single_repeat_span_snaps = ho_span_snaps / ho_a.repeat
                                for snap_offset in range(ho_span_snaps):
                                    snap = ho_a_pos_snap + snap_offset
                                    # consider repeat
                                    r, p = snap_offset // single_repeat_span_snaps, (snap_offset % single_repeat_span_snaps) / single_repeat_span_snaps
                                    if r % 2 == 1:
                                        p = 1 - p
                                    curve_pos = ho_a.curve(p)
                                    # if curve_pos.x > 1000 or curve_pos.y > 1000:
                                    #     print('invalid curve')
                                    #     print(r, p, curve_pos)
                                    x_pos_seq[snap] = curve_pos.x
                                    y_pos_seq[snap] = curve_pos.y
                                last_position = ho_a.curve(1)
                                ho_a_end_x = last_position.x
                                ho_a_end_y = last_position.y
                            else:
                                assert isinstance(ho_a, slider.beatmap.Spinner)
                                ho_a_end_x = ho_a.position.x
                                ho_a_end_y = ho_a.position.y
                                x_pos_seq[ho_a_pos_snap:ho_a_end_pos_snap] = ho_a_end_x
                                y_pos_seq[ho_a_pos_snap:ho_a_end_pos_snap] = ho_a_end_y
                    if ho_b is None:
                        # after the last ho
                        assert ho_a is not None
                        ho_b_pos_snap = total_snaps
                        ho_b_x = ho_a_end_x
                        ho_b_y = ho_a_end_y
                    else:
                        ho_b_pos_snap = get_ho_pos_snap(ho_b, end_time=False)
                        ho_b_x = ho_b.position.x
                        ho_b_y = ho_b.position.y

                    # do linear interpolation until the snap before the first snap occupied by hit_object ho_b
                    num_itp = ho_b_pos_snap - ho_a_end_pos_snap
                    x_pos_seq[ho_a_end_pos_snap:ho_b_pos_snap] = np.linspace(ho_a_end_x, ho_b_x, num=num_itp, endpoint=False)
                    y_pos_seq[ho_a_end_pos_snap:ho_b_pos_snap] = np.linspace(ho_a_end_y, ho_b_y, num=num_itp, endpoint=False)
                    if ho_a_end_x > 1000 or ho_a_end_y > 1000:
                        print('invalid ho_a')
                        print(ho_a)
                        print(ho_a.position)
                    if ho_b_x > 1000 or ho_b_y > 1000:
                        print('invalid ho_b')
                        print(ho_b)
                        print(ho_b.position)
                x_pos_seq = (x_pos_seq + 180) / (691 + 180)
                y_pos_seq = (y_pos_seq + 82) / (407 + 82)
                label = np.stack([snap_type, x_pos_seq, y_pos_seq], axis=1)
            except Exception:
                traceback.print_exc()
                print('failed process_label %s_%s' % (beatmapset_id, beatmap_id))
                return
            with open(paths['label'], 'wb') as f:
                pickle.dump(label, f)
            # plot_label(label)

            # """
            # create label idx
            # """
            beat_snap_type = snap_type.reshape([total_snaps // self.beat_divisor, self.beat_divisor])
            label_idx = [self.beat_label_seq_to_label_idx[tuple(beat_label_seq.tolist())] for beat_label_seq in beat_snap_type]
            with open(paths['label_idx'], 'wb') as f:
                pickle.dump(label_idx, f)

            # vis
            # if sample_idx in range(1):
            #     plot_label(sample_label)
            # # process data from mel
            # sample_data = self.process_data(beatmap, mel_spec)
            # with open(data_path, 'wb') as f:
            #     pickle.dump(sample_data, f)
            # process meta from mel
        if not (skip_exist and processed['meta']):
            try:
                meta_data = (beatmap.stars(), beatmap.cs())
            except Exception:
                traceback.print_exc()
                print('failed process_meta %s_%s' % (beatmapset_id, beatmap_id))
                return
            with open(paths['meta'], 'wb') as f:
                pickle.dump(meta_data, f)
            # if sample_idx in range(1):
            #     plot_meta(meta_data)

        if not (skip_exist and processed['info']):
            try:
                first_occupied_snap = get_ho_pos_snap(beatmap._hit_objects[0], end_time=False)
                last_occupied_snap = get_ho_pos_snap(beatmap._hit_objects[-1], end_time=True)
                sample_info = (total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap)
                # print(sample_info)
            except Exception:
                traceback.print_exc()
                print('failed process_info %s_%s' % (beatmapset_id, beatmap_id))
                return
            with open(paths['info'], 'wb') as f:
                pickle.dump(sample_info, f)


def worker(q: multiprocessing.Queue, i):
    # print('worker!')
    mel_args = {
        'sample_rate': 22000,
        'mel_frame_per_snap': 16,
        'n_mels': 40,
    }
    feature_args = {
        'beat_divisor': 8,
        'off_snap_threshold': 0.15,
    }
    ds = HeatmapDataset(mel_args, feature_args)
    temp_dir = r'/home/data1/xiezheng/osu_mapper/temp/%d' % i
    os.makedirs(temp_dir, exist_ok=True)
    while True:
        beatmap_meta_dict = q.get(block=True, timeout=None)
        if beatmap_meta_dict is None:
            # print('end msg')
            return 0
        ds.process_beatmap_meta_dict(
            beatmap_meta_dict,
            r'/home/data1/xiezheng/osu_mapper/preprocessed_v4',
            r'/home/data1/xiezheng/osu_mapper/beatmapsets',
            temp_dir,
        )


def multiprocessing_prepare_data(nproc=32, target=worker):
    meta_root = r'/home/xiezheng/osu_mapper/resources/data/osz'
    update_exist = True
    all = True
    print('updating exist data')
    all_meta = os.listdir(meta_root)
    all_meta_file_path = [os.path.join(meta_root, fn) for fn in all_meta]

    preprocessed_root = r'/home/data1/xiezheng/osu_mapper/preprocessed_v4'
    os.makedirs(preprocessed_root, exist_ok=True)

    processed_beatmapid_log_file = r'/home/data1/xiezheng/osu_mapper/preprocessed_v4/processed_ids.txt'
    if not os.path.exists(processed_beatmapid_log_file):
        open(processed_beatmapid_log_file, 'w')
    processed_beatmapid = set()
    with open(processed_beatmapid_log_file, 'r') as logf:
        for line in logf.readlines():
            processed_beatmapid.add(int(line))

    q = multiprocessing.Queue(maxsize=256)
    processes = [multiprocessing.Process(target=target, args=(q, i)) for i in range(nproc)]

    for process in processes:
        process.start()

    with open(processed_beatmapid_log_file, 'a') as logf:
        for meta_filepath in all_meta_file_path:
            with open(os.path.join(meta_filepath), 'r') as f:
                try:
                    meta_dict_list = json.load(f)
                except Exception:
                    continue
            for sample_idx, beatmap_meta_dict in enumerate(meta_dict_list):
                beatmap_id, beatmapset_id = int(beatmap_meta_dict['beatmap_id']), int(
                    beatmap_meta_dict['beatmapset_id'])
                if all:
                    if beatmap_id not in processed_beatmapid:
                        processed_beatmapid.add(beatmap_id)
                        logf.writelines([str(beatmap_id) + '\n'])
                    # print('add task %d' % beatmap_id)
                    q.put(beatmap_meta_dict, block=True, timeout=None)
                else:
                    if not update_exist:
                        if beatmap_id not in processed_beatmapid:
                            processed_beatmapid.add(beatmap_id)
                            logf.writelines([str(beatmap_id) + '\n'])
                            # print('add task %d' % beatmap_id)
                            q.put(beatmap_meta_dict, block=True, timeout=None)
                    else:
                        if beatmap_id in processed_beatmapid:
                            q.put(beatmap_meta_dict, block=True, timeout=None)

    for i in range(nproc):
        q.put(None)


if __name__ == '__main__':
    multiprocessing_prepare_data(16)
    # ds = HeatmapDataset(
    #     {
    #         'sample_rate': 22000,
    #         'mel_frame_per_snap': 16,
    #         'n_mels': 40,
    #     },
    #     {
    #         'beat_divisor': 8,
    #         'off_snap_threshold': 0.15,
    #     }
    # )
    # ds.from_meta_file(
    #     r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\heatmapv1',
    #     r'C:\Users\asus\coding\python\osu_mapper\resources\data\meta20230320.json',
    #     r'F:\beatmapsets',
    # )
    # ds.from_meta_file(
    #     r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\processed_v4',
    #     r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\osz\meta_2010-04-01.json',
    #     r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\beatmapsets',
    # )
