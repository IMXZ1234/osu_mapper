import os
import pickle
from datetime import timedelta
import yaml

from util import beatmap_util
from util.data import audio_osu_data
from preprocess import db, filter
from util.general_util import dynamic_import

DEFAULT_TRAIN_AUDIO_DIR = r'./resources/data/audio'
DEFAULT_INFERENCE_AUDIO_DIR = r'./resources/data/inference_audio'

DEFAULT_TRAIN_DB_PATH = r'./resources/data/osu_train.db'
DEFAULT_INFERENCE_DB_PATH = r'./resources/data/osu_inference.db'

DEFAULT_BEATMAP_LIST_SAVE_PATH = r'./resources/data/inference_beatmap_list.pkl'


def old_prepare_dataset():
    # src_dir = os.getcwd()
    # proj_dir = os.path.dirname(src_dir)
    # print(proj_dir)
    # print(torchaudio.backend.list_audio_backends())
    # song_dir = prepare_dataset.OSUSongsDir()
    # song_dir.gen_index_file()
    # data = audio_osu_data.AudioOsuData.from_path(r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\local.pkl')
    data = audio_osu_data.AudioOsuData.from_path(r'/\resources\data\raw\div2n_sninhtp.pkl')

    num = 0
    num1 = 0
    num2 = 0
    for beatmap in data.beatmaps:
        inherited_before_non_inherited = False
        first_timing_point_non_inherited = None
        first_timing_point = None
        f = False
        fn = False
        for time_point in beatmap.timing_points:
            if time_point.bpm is not None:
                first_timing_point_non_inherited = time_point.offset / timedelta(microseconds=1)
                if first_timing_point is None:
                    first_timing_point = first_timing_point_non_inherited
                f = True
                fn = True
            else:
                inherited_before_non_inherited = True
                first_timing_point = time_point.offset / timedelta(microseconds=1)
                f = True
            if f and fn:
                break
        if inherited_before_non_inherited:
            print('before')
            print(beatmap.beatmap_set_id)
            print(first_timing_point)
            print(first_timing_point_non_inherited)
        first_obj_time = beatmap.hit_objects()[0].time / timedelta(microseconds=1)
        # if first_obj_time < first_timing_point_non_inherited:
        #     print('obj')
        #     print(beatmap.beatmap_set_id)
        #     print(first_timing_point)
        #     print(first_timing_point_non_inherited)
        #     print(first_obj_time)
        #     num += 1
        start = min(first_timing_point, first_obj_time)
        if start < 0:
            print('start')
            if first_timing_point < 0:
                print('timing')
                num += 1
            if first_obj_time < 0:
                print('obj')
                num1 += 1
            if max(first_timing_point, first_obj_time) < 0:
                print('both')
                num2 += 1
            print(beatmap.beatmap_set_id)
            print(first_timing_point)
            print(first_timing_point_non_inherited)
            print(first_obj_time)
    print(num)
    print(num1)
    print(num2)
    # data_filter = prepare_dataset.DataFilter(data)
    # data = data_filter.keep_satisfied(cnnv1_criteria,
    #                            r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\div2n_sninhtp.pkl')
    # # data = audio_osu_data.AudioOsuData.from_path(r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\div2n_sninh_tp.pkl')
    # fd = prepare_dataset.FoldDivider(data)
    # fd.div_folds(r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\fold\div2n_sninhtp')
    # data_filter = prepare_dataset.DataFilter(data)
    # data_filter.keep_satisfied(has_multiple_non_inherited_timepoints,
    #                            r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\div2n_sninh_tp.pkl')
    # print(len(data.beatmaps))
    # beat_divisor_list = [beatmap.beat_divisor for beatmap in data.beatmaps]
    # print(beat_divisor_list)
    # divisor_count = divisor_distribution(beat_divisor_list)
    # timing_points_list = [beatmap.timing_points for beatmap in data.beatmaps]
    # non_inherited_timing_points_num = [0 for _ in range(len(timing_points_list))]
    # timing_points_num = []
    # for i, timing_points in enumerate(timing_points_list):
    #     timing_points_num.append(len(timing_points))
    #     for ti, timing_point in enumerate(timing_points):
    #         if timing_point.bpm is not None:
    #             non_inherited_timing_points_num[i] += 1
    # print(non_inherited_timing_points_num)
    # print(len(non_inherited_timing_points_num))
    # print(timing_points_num)
    # print(len(timing_points_num))
    # counts, bins = count(non_inherited_timing_points_num, bin_len=1, value_on_interval=False)
    # print(counts)
    # print(bins)
    # counts, bins = count(timing_points_num, bin_len=1, value_on_interval=False)
    # print(counts)
    # print(bins)
    # print(timing_points_list)
    # print(divisor_count)
    # plt.hist(beat_divisor_list)
    # plt.plot()
    # satisfied_index = data.get_satisfied(lambda beatmap: beatmap.beat_divisor in [1, 2, 4, 8])
    # satisfied_index = data.get_satisfied(lambda beatmap: beatmap.timing in [1, 2, 4, 8])
    # satisfied_index = data.get_satisfied(lambda beatmap: len(beatmap.timing_points))
    # print(satisfied_index)
    # print(len(satisfied_index))
    # satisfied_beatmaps = data.beatmaps_by_index(satisfied_index)
    # satisfied_audio_osu_list = data.audio_osu_list_by_index(satisfied_index)

    # data.save_index_file(satisfied_audio_osu_list, r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\commondiv.pkl')
    # data.save_beatmap_obj(satisfied_beatmaps, r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\raw\commondiv_beatmaps.pkl')

    # inspector = data_inspect.DataInspector(data)
    # bpm_min_list = inspector.attr_distribution('bpm_min')
    # bpm_max_list = inspector.attr_distribution('bpm_max')
    # for i in range(len(bpm_min_list)):
    #     if bpm_max_list[i] != bpm_min_list[i]:
    #         print(data.beatmaps[i].beatmap_id)
    #         print(bpm_max_list[i])
    #         print(bpm_min_list[i])
    # prepare_raw.OSUSongsDir().gen_index_file()
    # prepare_raw.fetch_audio_osu()
    # lib = slider.library.Library(os.path.join(proj_dir, 'data/slider_db'))
    # prepare_raw.download_beatmap_with_specifications(lib, datetime.datetime(2021, 1, 1))


class DataPreparer:
    def __init__(self, preprocessor_arg=None, filter_arg=None, fold_divider_arg=None,
                 do_preprocess=True, do_filter=True, do_fold_divide=True,
                 db_path=None,
                 from_db_path=None,
                 save_dir=None,
                 from_dir=None,
                 save_ext=None,
                 inference=False,
                 **kwargs):
        self.inference = inference
        self.save_dir, self.db_path = save_dir, db_path
        self.from_db_path, self.from_dir = from_db_path, from_dir
        self.save_ext = save_ext
        self.do_preprocess = do_preprocess
        if self.do_preprocess:
            self.preprocessor = self.load_preprocessor(**preprocessor_arg)
        # when preparing inference data we do not need filtering or fold division
        if self.inference:
            if self.save_dir is None:
                self.save_dir = DEFAULT_INFERENCE_AUDIO_DIR
            if self.db_path is None:
                self.db_path = DEFAULT_INFERENCE_DB_PATH
        else:
            if self.save_dir is None:
                self.save_dir = DEFAULT_TRAIN_AUDIO_DIR
            if self.db_path is None:
                self.db_path = DEFAULT_TRAIN_DB_PATH
            self.do_filter = do_filter
            if self.do_filter:
                self.filter = self.load_filter(**filter_arg)
            self.do_fold_divide = do_fold_divide
            if self.do_fold_divide:
                self.fold_divider = self.load_fold_divider(**fold_divider_arg)

    def load_preprocessor(self, preprocessor_type, **kwargs):
        p = dynamic_import(preprocessor_type)(**kwargs)
        return p

    def load_fold_divider(self, fold_divider_type, **kwargs):
        fd = dynamic_import(fold_divider_type)(**kwargs)
        return fd

    def load_filter(self, filter_type, **kwargs):
        filter_class = dynamic_import(filter_type)
        if issubclass(filter_class, filter.OsuTrainDataFilterGroup):
            data_filters = []
            for filter_arg in kwargs['filter_arg']:
                data_filters.append(self.load_filter(**filter_arg))
            # pass other parameters other than specifications for sub filters
            del kwargs['filter_arg']
            flt = dynamic_import(filter_type)(data_filters, **kwargs)
        else:
            flt = dynamic_import(filter_type)(**kwargs)
        return flt

    def prepare_train_data(self):
        if self.inference:
            raise ValueError(
                'DataPreparer initialized for inference data preparation can not be used to prepare train data'
            )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        database = db.OsuDB(self.db_path, connect=True)
        if self.do_preprocess:
            if self.from_db_path is None:
                database.gen_preprocessed(
                    self.preprocessor,
                    self.save_dir,
                    use_beatmap_list=False,
                    save_ext=self.save_ext,
                )
            else:
                database.gen_preprocessed_from_db(
                    self.preprocessor,
                    self.from_db_path,
                    self.save_dir,
                    self.from_dir,
                    save_ext=self.save_ext,
                )
        # records = database.get_table_records('MAIN')
        # print([record[4] for record in records])
        if self.do_filter:
            database.delete_view(view_name=None)
            # print(self.filter)
            # if issubclass(self.filter.__class__, filter.OsuTrainDataFilterGroup):
            #     print(self.filter.data_filters)
            database.filter_data(self.filter, self.save_dir)
        else:
            if 'FILTERED' not in database.all_view_names():
                database.create_view_from_rows('ID', database.get_column('ID', 'MAIN'), 'FILTERED')
        if self.do_fold_divide:
            all_views = database.all_view_names()
            for fold in range(1, 1 + self.fold_divider.folds):
                for view_name in ['TRAINFOLD%d' % fold, 'TESTFOLD%d' % fold]:
                    if view_name in all_views:
                        database.delete_view(view_name=view_name)
            database.split_folds(self.fold_divider)
        database.close()

    def prepare_inference_data(self,
                               from_audio_path_list,
                               beatmap_list,
                               start_id=None,
                               end_id=None):
        assert len(from_audio_path_list) == len(beatmap_list)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        database = db.OsuDB(self.db_path, connect=True)
        if self.do_preprocess:
            start_id, end_id = database.gen_preprocessed(
                self.preprocessor,
                self.save_dir,
                use_beatmap_list=True,
                beatmap_list=beatmap_list,
                from_audio_path_list=from_audio_path_list,
                clear_table=False,
                save_ext=self.save_ext
            )
        else:
            assert start_id is not None
            if end_id is None:
                end_id = start_id + 1
        print('current REFERENCE view: %d to %d' % (start_id, end_id))
        database.create_inference_view(list(range(start_id, end_id)))
        # records = database.get_table_records('MAIN')
        # print([record[4] for record in records])
        database.close()


def prepare_train_data_with_config(config_path):
    with open(os.path.join(config_path), 'r') as f:
        prepare_data_config = yaml.load(f, Loader=yaml.FullLoader)
    prepare = DataPreparer(**prepare_data_config)
    prepare.prepare_train_data()


def prepare_inference_data_with_config(config_path,
                                       from_audio_path_list,
                                       beatmap_list_save_path):
    with open(os.path.join(config_path), 'r') as f:
        prepare_data_config = yaml.load(f, Loader=yaml.FullLoader)
    prepare = DataPreparer(**prepare_data_config)
    prepare.prepare_inference_data(from_audio_path_list, beatmap_list_save_path)


def prepare_mel_train_data():
    config_path = r'resources/config/prepare_data/train/mel_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.MelPreprocessor',
        'beat_feature_frames': 16384,
        'snap_offset': 0,
        'n_fft': 1024,
        'win_length': None,
        'hop_length': 512,
        'n_mels': 128,
        'from_resampled': True
    }
    filter_arg = {
        'filter_type': 'preprocess.filter.OsuTrainDataFilterGroup',
        'filter_arg': [
            {
                'filter_type': 'preprocess.filter.HitObjectFilter',
            },
            {
                'filter_type': 'preprocess.filter.BeatmapsetSingleAudioFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleBMPFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleUninheritedTimingPointFilter',
            },
            {
                'filter_type': 'preprocess.filter.SnapDivisorFilter',
            },
        ]
    }
    fold_divider_arg = {
        'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBFoldDivider',
        # 'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBDebugFoldDivider',
        'folds': 5,
        'shuffle': False,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'filter_arg': filter_arg,
        'fold_divider_arg': fold_divider_arg,
        'do_preprocess': True,
        'do_filter': True,
        'do_fold_divide': True,
        'save_dir': r'./resources/data/mel',
        'from_dir': DEFAULT_TRAIN_AUDIO_DIR,
        'db_path': r'./resources/data/osu_train_mel.db',
        'from_db_path': DEFAULT_TRAIN_DB_PATH,
        'save_ext': '.pkl',
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_train_data_with_config(config_path)


def prepare_seg_multi_label_train_data():
    config_path = r'resources/config/prepare_data/train/seg_multi_label_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.ResamplePreprocessor',
        'beat_feature_frames': 16384,
    }
    filter_arg = {
        'filter_type': 'preprocess.filter.OsuTrainDataFilterGroup',
        'filter_arg': [
            {
                'filter_type': 'preprocess.filter.HitObjectFilter',
            },
            {
                'filter_type': 'preprocess.filter.BeatmapsetSingleAudioFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleBMPFilter',
            },
            {
                'filter_type': 'preprocess.filter.SingleUninheritedTimingPointFilter',
            },
        ]
    }
    fold_divider_arg = {
        'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBFoldDivider',
        # 'fold_divider_type': 'preprocess.fold_divider.OsuTrainDBDebugFoldDivider',
        'folds': 5,
        'shuffle': False,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'filter_arg': filter_arg,
        'fold_divider_arg': fold_divider_arg,
        'do_preprocess': False,
        'do_filter': False,
        'do_fold_divide': True,
        'save_dir': DEFAULT_TRAIN_AUDIO_DIR,
        'db_path': DEFAULT_TRAIN_DB_PATH,
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_train_data_with_config(config_path)


def prepare_seg_multi_label_inference_data(from_audio_path_list, bpm_list, snap_divisor=8):
    assert len(from_audio_path_list) == len(bpm_list)
    beatmap_list = [beatmap_util.get_empty_beatmap()
                    for _ in range(len(from_audio_path_list))]
    for beatmap, bpm in zip(beatmap_list, bpm_list):
        beatmap_util.set_bpm(beatmap, bpm, snap_divisor)
    with open(DEFAULT_BEATMAP_LIST_SAVE_PATH, 'wb') as f:
        pickle.dump(beatmap_list, f)
    config_path = r'resources/config/prepare_data/inference/seg_multi_label_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'preprocess.preprocessor.ResamplePreprocessor',
        'beat_feature_frames': 16384,
    }
    prepare_data_config = {
        'preprocessor_arg': preprocessor_arg,
        'do_preprocess': False,
        'save_dir': DEFAULT_INFERENCE_AUDIO_DIR,
        'db_path': DEFAULT_INFERENCE_DB_PATH,
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare_inference_data_with_config(
        config_path, from_audio_path_list, DEFAULT_BEATMAP_LIST_SAVE_PATH
    )


if __name__ == '__main__':
    prepare_mel_train_data()
    # database = db.OsuTrainDB(DEFAULT_DB_PATH)
    # # database.delete_view()
    # all_values = set(database.get_column('AUDIOFILENAME'))
    # print(all_values)
    # for value in set(database.get_column('AUDIOFILENAME')):
    #     if '_' in value:
    #         new_value = value.split('_')[0] + '.mp3'
    #         database.update_rows('AUDIOFILENAME', value, 'AUDIOFILENAME', new_value)
    # database.filter_data(filter.HitObjectFilter())
    # database.split_folds(fold_divider.OsuTrainDBFoldDivider(folds=5, shuffle=False))
    # EXTRA_COLUMNS = ['RESAMPLE_RATE']
    # new_columns = db.OsuTrainDB.DEFAULT_COLUMNS
    # new_columns.extend(EXTRA_COLUMNS)
    # database.delete_view(view_name=None)
    # database.alter_table_columns(new_columns=new_columns)
