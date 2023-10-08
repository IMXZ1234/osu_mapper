# from gen.beatmap_generator import BeatmapGenerator
import math
import os

import numpy as np
import torch
import torchaudio
import yaml
import pickle

from gen import gen_util
from gen.interpreter import cursor_with_hit_embedding
from nn.net import acgan_embeddingv1
from util import audio_util, beatmap_util
from util.audio_util import cal_mel_spec
from util.general_util import recursive_to_cpu, recursive_to_ndarray, recursive_flatten_batch


class ACGANEmbeddingGenerator:
    def __init__(self):
        self.n_snaps = 32 * 8
        self.batch_size = 16
        model_param = {
            'n_snap': self.n_snaps,
            'tgt_embedding_dim': 16,
            'tgt_coord_dim': 2,
            'audio_feature_dim': 40,
            'noise_dim': 16,
            'norm': 'LN',
            'middle_dim': 128,
            'preprocess_dim': 32,
            'condition_coeff': 1.,
            # star
            'cls_label_dim': 1,
        }

        self.mel_args = {
            'sample_rate': 22000,
            'n_fft': 512,
            'hop_length': 220,
            'n_mels': 40,
        }
        self.sample_rate = 22000
        self.beat_divisor = 8
        self.mel_frame_per_snap = 16
        self.n_mels = 40

        self.model = acgan_embeddingv1.Generator(**model_param)
        model_path = r'./resources/pretrained_models/model_adv_-1_0.pt'
        state_dict = torch.load(model_path)
        # self.model.load_state_dict(state_dict)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        self.model.load_state_dict(new_state_dict)
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.label_interpreter = cursor_with_hit_embedding.CursorWithHitEmbeddingInterpreter(
            r'./resources/pretrained_models/embedding_center-1_-1.pkl',
            self.beat_divisor,
            r'./resources/pretrained_models/counter.pkl',
        )

    @staticmethod
    def get_bpm_start_end_time(audio_file_path, audio_info_path):
        if not os.path.exists(audio_info_path):
            print('extracting bpm, start_time, end_time...')
            audio_info = gen_util.extract_bpm(audio_file_path)
            audio_info_dict = {
                'bpm': audio_info[0],
                'start_time': audio_info[1],
                'end_time': audio_info[2],
            }
            print(audio_info_dict)
            with open(audio_info_path, 'wt') as f:
                yaml.dump(audio_info_dict, f)
            print('saved audio info to %s' % audio_info_path)
        else:
            with open(audio_info_path, 'rt') as f:
                audio_info_dict = yaml.load(f, Loader=yaml.FullLoader)
            print(audio_info_dict)
            audio_info = [
                audio_info_dict['bpm'],
                audio_info_dict['start_time'],
                audio_info_dict['end_time'],
            ]
        return audio_info

    def generate_beatmapset(self,
                            osz_path,
                            audio_file_path,
                            meta_list,
                            audio_info_path=None,
                            osu_dir=None,
                            **kwargs):
        print('generating for %s' % audio_file_path)
        audio_filename_no_ext = os.path.splitext(os.path.basename(audio_file_path))[0]
        if osu_dir is None:
            osu_dir = os.path.dirname(audio_file_path)
        os.makedirs(osu_dir, exist_ok=True)
        if audio_info_path is None:
            audio_info_path = os.path.join(os.path.dirname(audio_file_path), audio_filename_no_ext + 'yaml')

        try:
            audio_data, sr = audio_util.audioread_get_audio_data(audio_file_path, ret_tensor=True)
        except Exception:
            print('fail to load audio %s' % audio_file_path)
            return
        audio_data = torchaudio.functional.resample(audio_data, sr, self.sample_rate)
        audio_data = audio_data.numpy()

        bpm, start_time, end_time = self.get_bpm_start_end_time(audio_file_path, audio_info_path)
        # from ms to sec
        start_time, end_time = start_time / 1000, end_time / 1000

        beat_length_s = 60 / bpm
        snap_length_s = beat_length_s / self.beat_divisor
        snap_length_samples_f = snap_length_s * self.sample_rate
        beat_length_samples_f = beat_length_s * self.sample_rate
        hop_length_s = snap_length_s / self.mel_frame_per_snap
        hop_length_samples_f = hop_length_s * self.sample_rate
        hop_length = math.ceil(hop_length_samples_f)
        # since we align mel frames with snaps,
        # the center of the first mel frame should depend on temporal position of the first timing point
        first_tp_s = start_time
        first_tp_sample_f = first_tp_s * self.sample_rate

        total_sample = audio_data.shape[1]
        # count by beat_length_samples_f backward and forward to retrieve aligned audio
        beats_before_first_tp = math.floor(first_tp_sample_f / beat_length_samples_f)
        crop_start_sample_f = first_tp_sample_f - beats_before_first_tp * beat_length_samples_f
        snaps_before_first_tp = beats_before_first_tp * self.beat_divisor
        beats_after_first_tp = math.floor((total_sample - first_tp_sample_f) / beat_length_samples_f)
        snaps_after_first_tp = beats_after_first_tp * self.beat_divisor
        crop_end_sample_f = first_tp_sample_f + beats_after_first_tp * beat_length_samples_f
        # crop_start_sample, crop_end_sample = round(crop_start_sample_f), round(crop_end_sample_f)

        total_beats_f = (crop_end_sample_f - crop_start_sample_f) / beat_length_samples_f
        total_snaps_f = total_beats_f * self.beat_divisor
        total_snaps = round(total_snaps_f)

        total_mel_frames = total_snaps * self.mel_frame_per_snap

        """
        process mel
        """
        try:
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
                n_mel=self.n_mels,
                sample_rate=self.sample_rate
            )
            # -> n_frames, n_mels
            mel_spec = np.mean(mel_spec, axis=1)
            print(mel_spec)
            assert mel_spec.shape[0] == total_mel_frames, 'mel_spec.shape[0] %s != total_mel_frames %s' % (mel_spec.shape[0], total_mel_frames)
            print('mel spec shape: ', mel_spec.shape)
        except Exception:
            print('mel spec generation failed!')
            return

        with open(r'C:\Users\asus\coding\python\osu_mapper\resources\data\processed_v4\mel\999834.pkl', 'rb') as f:
            mel_spec = pickle.load(f)
        total_mel_frames = len(mel_spec)
        total_snaps = total_mel_frames // 16

        osu_path_list = []
        beatmap_list = []

        # pad tail to multiples of subseq length
        extra_snaps, num_subseq = total_snaps % self.n_snaps, total_snaps // self.n_snaps
        print(total_snaps, total_mel_frames)
        print(extra_snaps, num_subseq)
        if extra_snaps > 0:
            mel_spec = np.concatenate([mel_spec, np.zeros([(self.n_snaps - extra_snaps) * self.mel_frame_per_snap, self.n_mels])], axis=0)
            num_subseq += 1
        mel_spec = mel_spec.reshape([num_subseq, self.n_snaps * self.mel_frame_per_snap, self.n_mels])
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        print('mel spec shape: ', mel_spec.shape)
        if torch.cuda.is_available():
            mel_spec = mel_spec.to('cuda:0')

        # run for every meta
        for meta_dict in meta_list:
            # get and remove 'star'
            meta = meta_dict.pop('star') / 5
            meta = torch.tensor([[meta]]).expand([mel_spec.shape[0], -1])
            meta = meta.to('cuda:0')
            with torch.no_grad():
                self.model.eval()
                # [
                gen_output = self.model.forward((mel_spec, meta))

            if torch.cuda.is_available():
                gen_output = recursive_to_cpu(gen_output)
            gen_output = recursive_to_ndarray(gen_output)
            gen_output = recursive_flatten_batch(gen_output, cat_dim=1)

            # plot_gen_output(gen_output)

            beatmap = beatmap_util.get_empty_beatmap()
            beatmap_util.set_bpm(beatmap, bpm, self.beat_divisor)
            beatmap_util.set_meta(
                beatmap,
                meta_dict
            )
            self.label_interpreter.gen_hitobjects(
                beatmap, gen_output, bpm, start_time * 1000, end_time * 1000
            )

            beatmap_name = beatmap_util.osu_filename(beatmap)
            osu_path = os.path.join(osu_dir, beatmap_name)
            beatmap.write_path(osu_path)

            osu_path_list.append(osu_path)
            beatmap_list.append(beatmap)
        beatmap_util.pack_to_osz(audio_file_path, osu_path_list, osz_path, beatmap_list)
        print('saved to %s' % osz_path)
