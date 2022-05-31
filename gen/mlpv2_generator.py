import os

from gen.beatmap_generator import BeatmapGenerator
from gen.label_interpreter import MultiLabelInterpreter
from util import beatmap_util


class MLPv2Generator(BeatmapGenerator):
    def __init__(self,
                 inference_config_path='./resources/config/inference/mlpv2_lr0.1.yaml',
                 prepare_data_config_path='./resources/config/prepare_data/inference/mlp_data.yaml'):
        print('using inference_config_path %s' % inference_config_path)
        print('using prepare_data_config_path %s' % prepare_data_config_path)
        super().__init__(inference_config_path, prepare_data_config_path)
        data_arg = self.config_dict['data_arg']
        dataset_arg = data_arg['test_dataset_arg']
        self.snap_divisor = dataset_arg['snap_divisor'] if 'snap_divisor' in dataset_arg else 8

    def generate_beatmapset(self, audio_file_path, speed_stars_list, meta_list,
                            osu_out_path_list=None, audio_info_path=None, audio_idx=0, **kwargs):
        # prepare inference data
        print('preparing inference data...')
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

        print('running inference...')
        beatmap_label_list = []
        # same audio for all beatmaps in beatmapset
        for beatmap in beatmap_list:
            gen = self.data_preparer.dataset.get_sample_data_generator(audio_file_path, beatmap)
            last_label = 1
            beatmap_labels = []
            for label_idx in range(len(gen)):
                feature = gen.get_next_sample_feature(last_label)
                last_label = int(self.inference.run_inference_sample(feature))
                beatmap_labels.append(last_label)
            beatmap_label_list.append(beatmap_labels)

        # generate .osu files
        print('generating .osu from predicted labels')
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
            # in milliseconds
            snap_ms = beatmap_util.get_snap_milliseconds(beatmap, self.snap_divisor)
            # clear formerly added two dummy hitobjects
            beatmap._hit_objects.clear()
            MultiLabelInterpreter.gen_hitobjects(
                beatmap, beatmap_label, beatmap_util.get_first_hit_object_time_milliseconds(beatmap), snap_ms, self.snap_divisor
            )

            beatmap.write_path(out_path)
        beatmap_util.pack_to_osz(audio_file_path, osu_out_path_list, beatmap_list=beatmap_list)
