import os
from datetime import timedelta

import yaml
from abc import abstractmethod

from preprocess import db, filter
from util.data import audio_osu_data
from util.general_util import dynamic_import

DEFAULT_TRAIN_AUDIO_DIR = r'./resources/data/audio'
DEFAULT_TRAIN_MEL_AUDIO_DIR = r'./resources/data/mel'
DEFAULT_INFERENCE_AUDIO_DIR = r'./resources/data/inference_audio'
DEFAULT_INFERENCE_MEL_AUDIO_DIR = r'./resources/data/inference_mel'

DEFAULT_TRAIN_DB_PATH = r'./resources/data/osu_train.db'
DEFAULT_TRAIN_MEL_DB_PATH = r'./resources/data/osu_train_mel.db'
DEFAULT_INFERENCE_DB_PATH = r'./resources/data/osu_inference.db'
DEFAULT_INFERENCE_MEL_DB_PATH = r'./resources/data/osu_inference_mel.db'

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

class SimpleDataPreparer:
    def __init__(self, dataset_type, dataset_arg=dict()):
        self.dataset = dynamic_import(dataset_type)(**dataset_arg)

    def prepare_train_data(self, **kwargs):
        self.dataset.prepare()
        self.dataset.div_folds(save_first=1)

    def prepare_inference_data(self, **kwargs):
        raise NotImplementedError


class DataPreparer:
    def __init__(self,
                 preprocessor_arg=dict(), filter_arg=dict(), fold_divider_arg=dict(),
                 do_preprocess=False, do_filter=False, do_fold_divide=False,
                 db_path=None,
                 from_db_path=None,
                 save_dir=None,
                 from_dir=None,
                 save_ext=None,
                 inference=False,
                 fit_dataset=False, fit_dataset_arg=dict(),
                 **kwargs):
        self.inference = inference
        self.fit_dataset = fit_dataset
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
        if self.fit_dataset:
            self.dataset = self.load_dataset(**fit_dataset_arg)

    def load_dataset(self, dataset_type, **kwargs):
        return dynamic_import(dataset_type)(**kwargs)

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
