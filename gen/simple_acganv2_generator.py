# from gen.beatmap_generator import BeatmapGenerator
import os

import numpy as np
import yaml
import torch

from gen import gen_util
from gen.interpreter import heatmap_pos
from preprocess.dataset import heatmap_datasetv2
from util import audio_util, general_util, beatmap_util, plt_util
from nn.net import simple_acganv2


class SimpleACGANGenerator:
    def __init__(self):
        model_param = {
                    'seq_len': 2560,
                    'tgt_dim': 5,
                    'noise_dim': 16,
                    'audio_feature_dim': 40,
                    'norm': 'LN',
                    'middle_dim': 128,
                    'preprocess_dim': 16,
                    'cls_label_dim': 3,
                }

        self.mel_args = {
            'sample_rate': 22000,
            'n_fft': 512,
            'hop_length': 220,
            'n_mels': 40,
        }
        self.dataset = heatmap_datasetv2.HeatmapDatasetv1(self.mel_args, {})

        self.model = simple_acganv2.Generator(**model_param)
        model_path = r'./resources/pretrained_models/model_adv_47_0.pt'
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        self.model.load_state_dict(new_state_dict)
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.label_interpreter = heatmap_pos.HeatmapPosInterpreter(
            self.mel_args['sample_rate'],
            self.mel_args['hop_length'],
            self.mel_args['n_fft'],
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

    def generate_meta(self, mel_spec, subseq_len):
        pass

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
        if audio_info_path is None:
            audio_info_path = os.path.join(os.path.dirname(audio_file_path), audio_filename_no_ext + 'yaml')
        audio_data, sr = audio_util.audioread_get_audio_data(audio_file_path)
        # L, C
        mel_spec = self.dataset.process_sample(audio_data, sr)
        bpm, start_time, end_time = self.get_bpm_start_end_time(audio_file_path, audio_info_path)
        print('bpm, start_time, end_time')
        print(bpm, start_time, end_time)
        subseq_len = 2560
        total_len = mel_spec.shape[0]
        start_frame = heatmap_datasetv2.time_to_frame(
            start_time,
            self.mel_args['sample_rate'],
            self.mel_args['hop_length'],
            self.mel_args['n_fft'],
        )
        num_subseq = (total_len - start_frame) // subseq_len
        audio_feature = mel_spec[start_frame: start_frame + num_subseq * subseq_len].reshape([num_subseq, subseq_len, -1])
        audio_feature = torch.tensor(audio_feature)
        ho_meta = torch.tensor([[0.01, 0.4, 0.]]).expand([num_subseq, -1])
        if torch.cuda.is_available():
            audio_feature = audio_feature.to(0)
            ho_meta = ho_meta.to(0)
        with torch.no_grad():
            labels = self.model.forward((audio_feature, ho_meta))
        if torch.cuda.is_available():
            labels = labels.to('cpu')
        labels = torch.cat([
            torch.zeros([start_frame, 5]),
            labels.reshape([num_subseq * subseq_len, -1]),
            torch.zeros([total_len - start_frame- num_subseq * subseq_len, 5]),
        ])
        labels = labels.numpy()
        plot_gen_output(labels)

        osu_path_list = []
        beatmap_list = []
        for beatmap_idx, beat_divisor in enumerate([8]):
            beatmap = beatmap_util.get_empty_beatmap()
            beatmap_util.set_bpm(beatmap, bpm, beat_divisor)
            beatmap_util.set_meta(
                beatmap,
                meta_list[beatmap_idx]
            )
            self.label_interpreter.gen_hitobjects(
                beatmap, labels, start_time, end_time, beat_divisor
            )
            beatmap_name = beatmap_util.osu_filename(beatmap)
            osu_path = os.path.join(osu_dir, beatmap_name)
            beatmap.write_path(osu_path)
            osu_path_list.append(osu_path)
            beatmap_list.append(beatmap)
        beatmap_util.pack_to_osz(audio_file_path, osu_path_list, osz_path, beatmap_list)
        print('saved to %s' % osz_path)


def plot_gen_output(gen_output):
    for signal, name in zip(
        gen_output.T,
        [
            'circle_hit', 'slider_hit', 'spinner_hit', 'cursor_x', 'cursor_y'
        ]
    ):
        fig_array, fig = plt_util.plot_signal(signal, name,
                                     save_path=None,
                                     show=True)
