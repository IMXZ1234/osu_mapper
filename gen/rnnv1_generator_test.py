import os

from gen.beatmap_generator import BeatmapGenerator
from gen.label_interpreter import MultiLabelInterpreter
from preprocess.prepare_data import DEFAULT_TRAIN_MEL_AUDIO_DIR, DEFAULT_TRAIN_MEL_DB_PATH
from util import beatmap_util


class RNNv1Generator(BeatmapGenerator):
    def __init__(self,
                 inference_config_path='./resources/config/inference/rnnv1_lr0.1.yaml',
                 prepare_data_config_path='./resources/config/prepare_data/inference/rnn_data.yaml'):
        print('using inference_config_path %s' % inference_config_path)
        print('using prepare_data_config_path %s' % prepare_data_config_path)
        super().__init__(inference_config_path, prepare_data_config_path)
        data_arg = self.config_dict['data_arg']
        dataset_arg = data_arg['test_dataset_arg']
        self.snap_divisor = dataset_arg['snap_divisor'] if 'snap_divisor' in dataset_arg else 8

    def generate_beatmapset(self, audio_file_path, speed_stars_list, meta_list,
                            osu_out_path_list=None, audio_info_path=None, audio_idx=0, **kwargs):
        # prepare inference cond_data
        print('preparing inference cond_data...')
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
            gen = self.data_preparer.dataset.get_db_sample_data_generator(
                DEFAULT_TRAIN_MEL_DB_PATH,
                DEFAULT_TRAIN_MEL_AUDIO_DIR,
            )
            state = self.inference.model.begin_state(batch_size=1, device=self.inference.output_device)
            last_label = 0
            beatmap_labels = []
            warmup_len = gen.warmup_len()
            print('warmup_len')
            print(warmup_len)
            period = [0] * self.snap_divisor
            period[1] = 1
            warmup_labels = [0] * (warmup_len % self.snap_divisor) + period * (warmup_len // self.snap_divisor)
            for label_idx in range(warmup_len):
                feature = gen.get_next_sample_feature(warmup_labels[label_idx], warmup=True)
                last_label, state = self.inference.run_inference_sample_rnn(feature, state)
                # print('warmup_label %d' % int(last_label))
            for label_idx in range(len(gen)):
                feature = gen.get_next_sample_feature(last_label, warmup=False)
                last_label, state = self.inference.run_inference_sample_rnn(feature, state)
                last_label = int(last_label)
                beatmap_labels.append(last_label)
            beatmap_label_list.append(beatmap_labels)
            true_beatmap_label = gen.audio_label.tolist()
            print(list(zip(beatmap_labels, true_beatmap_label)))
            # print('beatmap_labels')
            # print(beatmap_labels)
            # print('true_beatmap_label')
            # print(true_beatmap_label)
        # print(beatmap_label_list)

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
            # clear formerly added two dummy hitobjects
            start_time = beatmap_util.get_first_hit_object_time_milliseconds(beatmap)
            beatmap._hit_objects.clear()
            MultiLabelInterpreter.gen_hitobjects(
                beatmap, beatmap_label, start_time, snap_ms, self.snap_divisor
            )
            print(beatmap._hit_objects)
            beatmap.write_path(out_path)
        beatmap_util.pack_to_osz(audio_file_path, osu_out_path_list, beatmap_list=beatmap_list)
