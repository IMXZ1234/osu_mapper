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

from util import audio_util
from util.plt_util import plot_signal


def frames_to_time(length, sr, hop_length, n_fft):
    """
    output is in ms
    """
    frame_samples = np.arange(length) * hop_length + n_fft // 2
    return frame_samples / sr * 1000


def smooth_hit(x: np.ndarray, mu: "float | [float, float]", sigma: float = 5):
    """
    a smoothed impulse
    modelled using a normal distribution with mean `mu` and std. dev `sigma`
    evaluated at values in `x`
    """

    if isinstance(mu, (float, int)):
        z = (x - mu) / sigma
    elif isinstance(mu, tuple):
        a, b = mu
        z = np.where(x < a, x - a, np.where(x < b, 0, x - b)) / sigma
    else:
        raise NotImplementedError(f"`mu` must be float or tuple, got {type(mu)}")

    return np.exp(-.5 * z ** 2)


def hit_signal(beatmap: slider.Beatmap, frame_times: "L,") -> "L, 3":
    """
    returns an array encoding the hits occurring at the times represented by `frames`
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    # - [3] represents new combos

    - `frame_times`: array of times at each frame in ms
    """

    sig = np.zeros((3, len(frame_times)))
    for ho in beatmap._hit_objects:
        if isinstance(ho, slider.beatmap.Circle):
            sig[0] += smooth_hit(frame_times, ho.time // timedelta(milliseconds=1))
        elif isinstance(ho, slider.beatmap.Slider):
            sig[1] += smooth_hit(frame_times,
                                 (ho.time // timedelta(milliseconds=1), ho.end_time // timedelta(milliseconds=1)))
        elif isinstance(ho, slider.beatmap.Spinner):
            sig[2] += smooth_hit(frame_times,
                                 (ho.time // timedelta(milliseconds=1), ho.end_time // timedelta(milliseconds=1)))

    return sig.T


def cursor_signal(beatmap: slider.Beatmap, frame_times: "L,") -> "L, 2":
    """
    return [2,L] where [{0,1},i] is the {x,y} position at the times represented by `frames`

    - `frame_times`: array of times at each frame in ms
    """
    # print(beatmap._hit_objects)
    length = frame_times.shape[0]
    # print(frame_times)
    pos = np.zeros([length, 2])
    ci = 0
    for a, b in zip([None] + beatmap._hit_objects, beatmap._hit_objects + [None]):
        if a is None:
            # before first hit object
            while frame_times[ci] < b.time / timedelta(milliseconds=1):
                pos[ci] = np.array([b.position.x, b.position.y])
                ci += 1
        elif b is None:
            if isinstance(a, (slider.beatmap.Circle, slider.beatmap.Spinner)):
                ho_pos = a.position
            else:
                # is Slider, take end pos
                ho_pos = a.curve(1)
            # after last hit object
            while ci < length:
                pos[ci] = np.array([ho_pos.x, ho_pos.y])
                ci += 1
        else:
            a_time, b_time = a.time / timedelta(milliseconds=1), b.time / timedelta(milliseconds=1)
            last_pos = np.array([a.position.x, a.position.y])
            last_time = a_time
            if isinstance(a, slider.beatmap.Slider):
                a_end_time = a.end_time / timedelta(milliseconds=1)
                a_len = a_end_time - a_time
                one_repeat_a_len = a_len / a.repeat
                while frame_times[ci] < a_end_time:
                    offset = frame_times[ci] - a_time
                    # consider repeat
                    r, p = offset / one_repeat_a_len, (offset % one_repeat_a_len) / one_repeat_a_len
                    if r % 2 == 1:
                        p = 1 - p
                    curve_pos = a.curve(p)
                    pos[ci] = np.array([curve_pos.x, curve_pos.y])
                    ci += 1
                last_pos = np.array([curve_pos.x, curve_pos.y])
                last_time = a_end_time
            elif isinstance(a, slider.beatmap.Spinner):
                a_end_time = a.end_time / timedelta(milliseconds=1)
                while frame_times[ci] < a_end_time:
                    pos[ci] = np.array([a.position.x, a.position.y])
                    ci += 1
                last_pos = np.array([a.position.x, a.position.y])
                last_time = a_end_time
            # do linear interpolation until hit_object b
            interpolate_itv = b_time - last_time
            b_pos = np.array([b.position.x, b.position.y])
            pos_change = b_pos - last_pos
            while frame_times[ci] < b_time:
                offset = frame_times[ci] - last_time
                pos[ci] = offset / interpolate_itv * pos_change + last_pos
                ci += 1
    pos[:, 0] = (pos[:, 0] + 180) / (691 + 180)
    pos[:, 1] = (pos[:, 1] + 82) / (407 + 82)
    return pos


def generate_label(beatmap, frame_times, add_noise=True):
    """
    -> L, 5
    circle hit, slider hit, spinner hit, cursor x, cursor y
    """
    cursor_sig = cursor_signal(beatmap, frame_times)
    hit_sig = hit_signal(beatmap, frame_times)
    if add_noise:
        # noise's value for cursor_sig should be small
        cursor_sig += np.random.randn(*cursor_sig.shape) / 1024
        # noise's value for hit_signal could be a litte larger
        hit_sig += np.random.randn(*hit_sig.shape) / 32
    return np.concatenate([hit_sig, cursor_sig], axis=1)


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


def plot_label(label):
    for signal, name in zip(
            label.T,
            [
                'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
            ]
    ):
        plot_signal(signal, name)
    for signal, name in zip(
            label.T,
            [
                'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
            ]
    ):
        plot_signal(signal[:5120], name)


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


class HeatmapDatasetv1:
    def __init__(self, mel_args, feature_args):
        self.mel_args = mel_args
        self.feature_args = feature_args

        self.sample_rate = mel_args.get('sample_rate', 22000)
        self.n_fft = mel_args.get('n_fft', 2048)
        self.hop_length = mel_args.get('hop_length', (self.sample_rate // 1000) * 6)
        self.n_mels = mel_args.get('n_mels', 64)

        self.coeff_approach_rate = feature_args.get('coeff_approach_rate', 0.25)
        self.coeff_snap_divisor = feature_args.get('coeff_snap_divisor', 0.1)
        self.coeff_ms_per_beat = feature_args.get('coeff_ms_per_beat', 60 / 60000)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=self.n_mels,
            mel_scale="htk",
        )

    def process_sample(self, audio_data, sr, beatmap=None, meta_dict=None):
        mel_spec = self.process_audio(audio_data, sr)
        if beatmap is None:
            return mel_spec
        label = self.process_label(beatmap, mel_spec)
        if meta_dict is None:
            return mel_spec, label
        meta = self.process_meta(beatmap, mel_spec, meta_dict)
        return mel_spec, label, meta

    def process_meta(self, beatmap, mel_spec, meta_dict):
        frame_times = frames_to_time(mel_spec.shape[0], self.sample_rate, self.hop_length, self.n_fft)
        ms_per_beat, ho_density, occupied_proportion, snap_divisor = calculate_meta(beatmap, frame_times)
        diff_size = float(meta_dict['diff_size'])
        diff_overall = float(meta_dict['diff_overall'])
        diff_approach = float(meta_dict['diff_approach'])
        diff_drain = float(meta_dict['diff_drain'])
        diff_aim = float(meta_dict['diff_aim'])
        diff_speed = float(meta_dict['diff_speed'])
        difficultyrating = float(meta_dict['difficultyrating'])
        return ms_per_beat, ho_density, occupied_proportion, snap_divisor,\
               diff_size, diff_overall, diff_approach, diff_drain, diff_aim, diff_speed, difficultyrating

    def process_label(self, beatmap, mel_spec):
        frame_times = frames_to_time(mel_spec.shape[0], self.sample_rate, self.hop_length, self.n_fft)
        sample_label = generate_label(beatmap, frame_times, add_noise=False)
        return sample_label

    def process_audio(self, audio_data, sr):
        audio_data = torchaudio.functional.resample(audio_data, sr, self.sample_rate)
        # to single channel
        mel_spec = torch.mean(self.mel_spectrogram(audio_data), dim=0).numpy().T
        # print(mel_spec.shape)
        return mel_spec

    def process_info(self, mel_spec, beatmapsetid):
        # (total frames, beatmapsetid)
        return mel_spec.shape[0], beatmapsetid

    def from_meta_file(self, save_dir, meta_filepath, osz_dir,
                       # tgt_mel_dir=None, ref_mel_dir=None,
                       # tgt_osu_dir=None, ref_osu_dir=None,
                       temp_dir='E:/'
                       ):
        """
        will save processed label under data_dir
        """
        tgt_mel_dir = os.path.join(save_dir, 'mel')
        tgt_meta_dir = os.path.join(save_dir, 'meta')
        tgt_label_dir = os.path.join(save_dir, 'label')
        tgt_info_dir = os.path.join(save_dir, 'info')
        os.makedirs(tgt_mel_dir, exist_ok=True)
        os.makedirs(tgt_label_dir, exist_ok=True)
        os.makedirs(tgt_meta_dir, exist_ok=True)
        os.makedirs(tgt_info_dir, exist_ok=True)

        with open(os.path.join(meta_filepath), 'r') as f:
            meta_dict_list = json.load(f)
        for sample_idx, beatmap_meta_dict in enumerate(meta_dict_list.values()):
            # str
            beatmap_id, beatmapset_id = beatmap_meta_dict['beatmap_id'], beatmap_meta_dict['beatmapset_id']
            mel_path = os.path.join(tgt_mel_dir, beatmapset_id + '.pkl')
            label_path = os.path.join(tgt_label_dir, beatmap_id + '.pkl')
            meta_path = os.path.join(tgt_meta_dir, beatmap_id + '.pkl')
            info_path = os.path.join(tgt_info_dir, beatmap_id + '.pkl')

            mel_processed = os.path.exists(mel_path)
            label_processed = os.path.exists(label_path)
            meta_processed = os.path.exists(meta_path)
            info_processed = os.path.exists(info_path)
            if mel_processed and label_processed and meta_processed and info_processed:
                print('skipping %s' % beatmap_id)
                continue

            osz_path = os.path.join(osz_dir, beatmapset_id + '.osz')
            if not check_integrity(osz_path):
                print('osz corrupted %s' % beatmapset_id)
                continue

            osz_file = zipfile.ZipFile(osz_path, 'r')
            try:
                all_beatmaps = list(slider.Beatmap.from_osz_file(osz_file).values())
            except Exception:
                print('beatmap parse failed %s' % beatmapset_id)
                continue
            # print([(bm.beatmap_id, bm.beatmap_set_id) for bm in all_beatmaps])
            beatmap = None
            for bm in all_beatmaps:
                if bm.version == beatmap_meta_dict['version'] or bm.beatmap_id == int(beatmap_id):
                    beatmap = bm
            if beatmap is None:
                print('beatmap %s not found in osz %s' % (beatmap_id, osz_path))
                continue
            print('processing beatmap %s' % beatmap_id)
            if not os.path.exists(mel_path):
                # process mel
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
                    continue
                try:
                    audio_data, sr = audio_util.audioread_get_audio_data(on_disk_audio_filepath)
                except Exception:
                    print('fail to load audio %s' % beatmapset_id)
                    continue
                os.remove(on_disk_audio_filepath)

                mel_spec = self.process_audio(audio_data, sr)
                with open(mel_path, 'wb') as f:
                    pickle.dump(mel_spec, f)
            else:
                with open(mel_path, 'rb') as f:
                    mel_spec = pickle.load(f)

            if not label_processed:
                try:
                    # process label
                    sample_label = self.process_label(beatmap, mel_spec)
                except Exception:
                    traceback.print_exc()
                    print('failed process_label %s_%s' % (beatmapset_id, beatmap_id))
                    continue
                with open(label_path, 'wb') as f:
                    pickle.dump(sample_label, f)

                # vis
                if sample_idx in range(10):
                    # for signal, name in zip(
                    #         [ms_per_beat, ho_density, occupied_proportion],
                    #         ['ms_per_beat', 'ho_density', 'blank_proportion']
                    # ):
                    #     plot_signal(signal, name)
                    plot_label(sample_label)
                # # process data from mel
                # sample_data = self.process_data(beatmap, mel_spec)
                # with open(data_path, 'wb') as f:
                #     pickle.dump(sample_data, f)
                # process meta from mel
            if not meta_processed:
                try:
                    sample_data = self.process_meta(beatmap, mel_spec, beatmap_meta_dict)
                except Exception:
                    traceback.print_exc()
                    print('failed process_meta %s_%s' % (beatmapset_id, beatmap_id))
                    continue
                with open(meta_path, 'wb') as f:
                    pickle.dump(sample_data, f)

            if not info_processed:
                try:
                    sample_info = self.process_info(mel_spec, beatmapset_id)
                except Exception:
                    traceback.print_exc()
                    print('failed process_meta %s_%s' % (beatmapset_id, beatmap_id))
                    continue
                with open(info_path, 'wb') as f:
                    pickle.dump(sample_info, f)


if __name__ == '__main__':
    ds = HeatmapDatasetv1({}, {})
    ds.from_meta_file(
        r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\heatmapv1',
        r'C:\Users\asus\coding\python\osu_mapper\resources\data\meta20230320.json',
        r'F:\beatmapsets',
    )