import os

from gen.beatmap_generator import BeatmapGenerator
from gen.label_interpreter import BiLabelInterpreter
from util import beatmap_util


class SegMultiLabelGenerator(BeatmapGenerator):
    def __init__(self,
                 inference_config_path='./resources/config/inference/seg_mlp_bi_lr0.1.yaml',
                 prepare_data_config_path='./resources/config/prepare_data/inference/seg_multi_label_data.yaml'):
        print('using inference_config_path %s' % inference_config_path)
        print('using prepare_data_config_path %s' % prepare_data_config_path)
        super().__init__(inference_config_path, prepare_data_config_path)
        data_arg = self.config_dict['data_arg']
        dataset_arg = data_arg['test_dataset_arg']
        self.beat_feature_frames = dataset_arg['beat_feature_frames'] if 'beat_feature_frames' in dataset_arg else 16384
        self.snap_divisor = dataset_arg['snap_divisor'] if 'snap_divisor' in dataset_arg else 8
        self.sample_beats = dataset_arg['sample_beats'] if 'sample_beats' in dataset_arg else 16
        self.pad_beats = dataset_arg['pad_beats'] if 'pad_beats' in dataset_arg else 4
        self.pad_frames = self.pad_beats * self.beat_feature_frames
        self.batch_size = data_arg['batch_size'] if 'batch_size' in dataset_arg else 1

    # def preprocess(self, audio_file_path):
    #     """
    #     Resampling and padding/trimming.
    #     """
    #     audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
    #     bpm, start_time, end_time = gen_util.extract_bpm(audio_file_path)
    #     start_frame, end_frame = SegMultiLabelDBDataset.get_aligned_start_end_frame(
    #         bpm, start_time, end_time, self.beat_feature_frames, self.sample_beats
    #     )
    #     resample_rate = round(self.beat_feature_frames * bpm / 60)  # * feature_frames_per_second
    #     resampled_audio_data = torchaudio.functional.resample(
    #         audio_data, sample_rate, resample_rate
    #     )
    #     cond_data = SegMultiLabelDBDataset.preprocess(
    #         resampled_audio_data, start_frame, end_frame, self.pad_beats
    #     )
    #     return cond_data

    def generate_beatmapset(self, audio_file_path, speed_stars_list, meta_list,
                            osu_out_path_list=None, audio_info_path=None, audio_idx=0, **kwargs):
        # prepare inference cond_data
        print('preparing inference dara...')
        audio_info = BeatmapGenerator.get_bpm_start_end_time(audio_file_path, audio_info_path)
        # if multiple beatmaps of different difficulties are to be generated for a single audio,
        # meta should not be None, as `version` must be specified for each beatmap.
        if meta_list is None:
            meta_list = [
                {
                    'audio_filename': os.path.basename(audio_file_path),  # indispensable
                    'title': os.path.splitext(os.path.basename(audio_file_path))[0],  # indispensable
                    'artist': 'Various Artists',
                    'creator': 'osu_mapper',
                    'version': str(speed_stars),  # indispensable
                } for speed_stars in speed_stars_list
            ]
        beatmap_list = BeatmapGenerator.initialize_beatmaps(
            audio_info, speed_stars_list, meta_list, self.snap_divisor
        )

        # same audio for all beatmaps in beatmapset
        self.data_preparer.prepare_inference_data(
            [audio_file_path] * len(beatmap_list), beatmap_list,
            audio_idx, audio_idx + 1
        )

        # run inference and get beatmapset_label for each audio
        print('running inference...')
        # into batch of size 1 to conform to the model input format
        beatmapset_label = self.inference.run_inference()
        print(beatmapset_label)
        beatmapset_label_div = self.inference.data_iter.dataset.accumulate_audio_sample_num
        beatmapset_label_div = [0] + beatmapset_label_div
        # although all beatmaps in beatmapset share same audio,
        # they may have different cond_data due to different difficulty
        beatmap_label_list = [
            beatmapset_label[beatmapset_label_div[i - 1]:beatmapset_label_div[i]].reshape([-1])
            for i in range(1, len(beatmapset_label_div))
        ]
        print(beatmap_label_list)

        # generate .osu files
        print('generating .osu from predicted cond_data')
        if osu_out_path_list is None:
            osu_out_path_list = [
                # generate .osu under same directory as audio file
                os.path.join(beatmap_util.DEFAULT_OSU_DIR, beatmap_util.osu_filename(beatmap))
                for beatmap in beatmap_list
            ]
        else:
            assert len(osu_out_path_list) == len(beatmap_list)
        for beatmap_label, beatmap, out_path in \
                zip(beatmap_label_list, beatmap_list, osu_out_path_list):
            # in microseconds
            snap_microseconds = 60000000 / (audio_info[0] * self.snap_divisor)
            # clear formerly added two dummy hitobjects
            beatmap._hit_objects.clear()
            BiLabelInterpreter.gen_hitobjects(
                beatmap, beatmap_label, audio_info[1], snap_microseconds, self.snap_divisor
            )

            beatmap.write_path(out_path)
        beatmap_util.pack_to_osz(audio_file_path, osu_out_path_list, beatmap_list=beatmap_list)
