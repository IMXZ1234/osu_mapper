import os

from gen.beatmap_generator import BeatmapGenerator
from gen.label_interpreter import HeatmapLabelWithPosInterpreter
from util import beatmap_util


class CDDPMGenerator(BeatmapGenerator):
    def __init__(self,
                 inference_config_path='./resources/config/inference/cganv7_within_batch.yaml',
                 prepare_data_config_path='./resources/config/prepare_data/inference/label_pos_datav3.yaml'):
        print('using inference_config_path %s' % inference_config_path)
        print('using prepare_data_config_path %s' % prepare_data_config_path)
        super().__init__(inference_config_path, prepare_data_config_path)
        data_arg = self.config_dict['data_arg']
        dataset_arg = data_arg['test_dataset_arg']
        self.snap_divisor = dataset_arg['snap_divisor'] if 'snap_divisor' in dataset_arg else 8

    def generate_beatmapset(self, audio_file_path, speed_stars_list, meta_list,
                            osu_out_path_list=None, audio_info_path=None, audio_idx=0, **kwargs):
        # prepare inference cond_data
        print('preparing inference data...')
        if meta_list is not None and 'title' in meta_list[0]:
            title = meta_list[0]['title']
        else:
            title = None
        audio_info = BeatmapGenerator.get_bpm_start_end_time(audio_file_path, audio_info_path, title)
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
        self.data_preparer.dataset.prepare_inference(audio_file_path, beatmap_list, save=True)

        # run inference and get beatmapset_label for each audio
        print('running inference...')
        # into batch of size 1 to conform to the model input format
        beatmap_label_list = self.inference.run_inference_cganv5()
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
            # in milliseconds
            snap_ms = beatmap_util.get_snap_milliseconds(beatmap, self.snap_divisor)
            print('snap_ms')
            print(snap_ms)
            start_time = beatmap_util.get_first_hit_object_time_milliseconds(beatmap)
            # clear formerly added two dummy hitobjects
            beatmap._hit_objects.clear()
            HeatmapLabelWithPosInterpreter.gen_hitobjects(
                beatmap, beatmap_label, start_time, snap_ms, self.snap_divisor
            )
            print(beatmap._hit_objects)
            beatmap.write_path(out_path)
        beatmap_util.pack_to_osz(audio_file_path, osu_out_path_list, beatmap_list=beatmap_list)
