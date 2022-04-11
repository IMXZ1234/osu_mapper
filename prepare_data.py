import os
from datetime import timedelta
import yaml

from util.data import audio_osu_data, db, filter, prepare_data_util, fold_divider
from util.general_util import dynamic_import

DEFAULT_AUDIO_DIR = r'./resources/data/audio'
DEFAULT_DB_PATH = r'./resources/data/osu_train.db'


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


class OsuTrainDBFactory:
    def __init__(self, preprocessor_arg, filter_arg, fold_divider_arg,
                 do_preprocess=True, do_filter=True, do_fold_divide=True,
                 audio_dir=DEFAULT_AUDIO_DIR,
                 db_path=DEFAULT_DB_PATH):
        self.audio_dir, self.db_path = audio_dir, db_path
        self.preprocessor = self.load_preprocessor(**preprocessor_arg)
        self.filter = self.load_filter(**filter_arg)
        self.fold_divider = self.load_fold_divider(**fold_divider_arg)
        self.do_preprocess = do_preprocess
        self.do_filter = do_filter
        self.do_fold_divide = do_fold_divide

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

    def prepare_data(self):
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        database = db.OsuTrainDB(self.db_path, connect=True)
        if self.do_preprocess:
            database.gen_preprocessed(self.preprocessor, self.audio_dir, )
        # records = database.get_table_records('MAIN')
        # print([record[4] for record in records])
        if self.do_filter:
            database.delete_view(view_name=None)
            print(self.filter)
            if issubclass(self.filter.__class__, filter.OsuTrainDataFilterGroup):
                print(self.filter.data_filters)
            database.filter_data(self.filter)
        if self.do_fold_divide:
            for fold in range(1, 1+self.fold_divider.folds):
                database.delete_view(view_name='TRAINFOLD%d' % fold)
                database.delete_view(view_name='TESTFOLD%d' % fold)
            database.split_folds(self.fold_divider)
        database.close()


def prepare_seg_multi_label_data():
    config_path = r'./resources/config/prepare_data/seg_multi_label_data.yaml'
    preprocessor_arg = {
        'preprocessor_type': 'util.data.preprocessor.ResamplePreprocessor',
        'beat_feature_frames': 16384,
    }
    filter_arg = {
        'filter_type': 'util.data.filter.OsuTrainDataFilterGroup',
        'filter_arg': [
            {
                'filter_type': 'util.data.filter.HitObjectFilter',
            },
            {
                'filter_type': 'util.data.filter.BeatmapsetSingleAudioFilter',
            },
            {
                'filter_type': 'util.data.filter.SingleBMPFilter',
            },
            {
                'filter_type': 'util.data.filter.SingleUninheritedTimingPointFilter',
            },
        ]
    }
    fold_divider_arg = {
        'fold_divider_type': 'util.data.fold_divider.OsuTrainDBFoldDivider',
        # 'fold_divider_type': 'util.data.fold_divider.OsuTrainDBDebugFoldDivider',
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
    }
    with open(os.path.join(config_path), 'w') as f:
        yaml.dump(prepare_data_config, f)
    prepare = OsuTrainDBFactory(**prepare_data_config)
    prepare.prepare_data()


if __name__ == '__main__':
    prepare_seg_multi_label_data()
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
