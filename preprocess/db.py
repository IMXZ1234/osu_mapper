import pickle
import sqlite3
import os
from typing import Union

import slider
import traceback

from preprocess import preprocessor, prepare_data_util, filter, fold_divider
from util import beatmap_util, general_util


class SQLite3DB:
    def __init__(self, db_path, detect_types=sqlite3.PARSE_COLNAMES, connect=True):
        self.db_path = db_path
        self.detect_types = detect_types
        if connect:
            self.con = sqlite3.connect(db_path, detect_types=detect_types)

    def connect(self):
        self.con = sqlite3.connect(self.db_path, detect_types=self.detect_types)

    def close(self):
        self.con.close()

    def create_table(self, columns, columns_type, table_name='MAIN', commit=True):
        sql = 'CREATE TABLE %s(%s %s PRIMARY KEY NOT NULL' % (table_name, columns[0], columns_type[0])
        for i in range(1, len(columns)):
            sql += ', %s %s NOT NULL' % (columns[i], columns_type[i])
        sql += ');'
        self.con.execute(sql)
        if commit:
            self.con.commit()

    def rename_table(self, table_name, new_table_name, commit=True):
        """
        Rename from table_name to new_table_name.
        """
        if new_table_name == table_name:
            print('new table name is same as old table name, no action performed.')
            return
        self.con.execute(r'ALTER TABLE %s RENAME TO %s;' % (table_name, new_table_name))
        if commit:
            self.con.commit()

    def delete_table(self, table_name='MAIN', commit=True):
        self.con.execute('DROP TABLE %s' % table_name)
        if commit:
            self.con.commit()

    def copy_table(self, new_table_name, table_name='MAIN', commit=True):
        self.con.execute('CREATE TABLE %s AS SELECT * FROM %s' % (new_table_name, table_name))
        if commit:
            self.con.commit()

    def alter_table_columns(self, new_columns=None, new_columns_type=None, table_name='MAIN', commit=True):
        if new_columns is None and new_columns_type is None:
            print('new_columns and new_columns_type are both None, no action performed.')
            return
        columns, columns_type = self.table_info(table_name)
        if new_columns is None:
            new_columns = columns
        if new_columns_type is None:
            new_columns_type = columns_type
        tmp_table_name = table_name + 'RENAMETMP'
        self.rename_table(table_name, tmp_table_name, commit=False)
        self.create_table(new_columns, new_columns_type, table_name, commit=False)
        self.con.execute('INSERT INTO %s SELECT * FROM %s;' % (table_name, tmp_table_name))
        self.delete_table(tmp_table_name, commit=False)
        if commit:
            self.con.commit()

    def create_view_from_rows(self, column, values, view_name, table_name='MAIN', commit=True):
        sql = 'CREATE VIEW %s AS SELECT * FROM %s WHERE %s IN (%s);' % \
              (view_name, table_name, column, ','.join([str(value) for value in values]))
        self.con.execute(sql)
        if commit:
            self.con.commit()

    def delete_view(self, view_name=None, commit=True):
        """
        If view_name is None, delete all views.
        """
        all_view_names = self.all_view_names()
        if view_name is None:
            for view_name in all_view_names:
                self.con.execute('DROP VIEW %s' % view_name)
        else:
            if view_name in all_view_names:
                self.con.execute('DROP VIEW %s' % view_name)
            else:
                print('view not found, no action performed.')
        if commit:
            self.con.commit()

    def insert_row(self, values, table_num='MAIN', commit=True):
        sql = 'INSERT INTO %s VALUES (%s);' % (table_num, ','.join(['?'] * len(values)))
        try:
            self.con.execute(sql, values)
        except sqlite3.IntegrityError:
            print('duplicate record!')
            return
        if commit:
            self.con.commit()

    def update_rows(self, column, values, column_alter, new_values, table_num='MAIN', commit=True):
        """
        Select rows whose value in `column` is in `values`, then
        alter these rows' values in `column_alter` to `new_values`.
        """
        assert isinstance(values, (list, tuple)) == isinstance(new_values, (list, tuple))
        if not isinstance(values, (list, tuple)):
            values = (values,)
            new_values = (new_values,)
        for value, new_value in zip(values, new_values):
            self.con.execute('UPDATE %s SET %s = ? WHERE %s = ?;'
                             % (table_num, column_alter, column),
                             (new_value, value))
        if commit:
            self.con.commit()

    def all_view_names(self):
        all_view_info = self.con.execute("SELECT name FROM sqlite_master WHERE type='view';").fetchall()
        return [view_info[0] for view_info in all_view_info]

    def all_table_names(self):
        all_table_info = self.con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return [table_info[0] for table_info in all_table_info]

    def table_info(self, table_name='MAIN'):
        """
        Returns [column names, column types]
        """
        cursor = self.con.execute('PRAGMA TABLE_INFO(%s)' % table_name)
        all_info = cursor.fetchall()
        return [column_info[1] for column_info in all_info], [column_info[2] for column_info in all_info]

    def record_num(self, table_name='MAIN'):
        cursor = self.con.execute('SELECT count(*) FROM %s;' % table_name)
        return cursor.fetchall()[0][0]

    def get_table_records(self, table_name):
        cursor = self.con.execute('SELECT * from %s;' % table_name)
        return cursor.fetchall()

    def get_row(self, column, values, table_num='MAIN'):
        """
        If values is a single value, return a row. else return a list of rows.
        """
        if isinstance(values, (tuple, list)):
            cursor = self.con.execute('SELECT * from %s WHERE %s IN (%s);'
                                      % (table_num, column, ','.join(['?'] * len(values))),
                                      values)
            return cursor.fetchall()
        else:
            cursor = self.con.execute('SELECT * from %s WHERE %s = %s;'
                                      % (table_num, column, values))
            return cursor.fetchall()[0]

    def get_column(self, column, table_name='MAIN'):
        """
        Return the column as a list.
        """
        cursor = self.con.execute('SELECT %s from %s;' % (column, table_name))
        return [record[0] for record in cursor.fetchall()]


class OsuDB(SQLite3DB):
    DEFAULT_COLUMNS = ['ID', 'BEATMAPID', 'BEATMAPSETID', 'BEATMAP [%s]' % slider.Beatmap.__name__, 'AUDIOFILENAME']
    DEFAULT_COLUMNS_TYPE = ['INT', 'INT', 'INT', 'BLOB', 'TEXT']
    DEFAULT_COLUMNS_DATA = ['ID', 'BEATMAPID', 'BEATMAPSETID', 'BEATMAP [%s]' % slider.Beatmap.__name__, 'DATA']
    DEFAULT_COLUMNS_DATA_TYPE = ['INT', 'INT', 'INT', 'BLOB', 'BLOB']
    DEFAULT_COLUMNS_NUM = 5

    @staticmethod
    def beatmap_sql_adaptor(beatmap: slider.Beatmap):
        return pickle.dumps(beatmap)

    @staticmethod
    def beatmap_sql_converter(data: bytes):
        return pickle.loads(data)

    def __init__(self, db_path, connect=True):
        # primary key should be at pos 0
        sqlite3.register_adapter(slider.Beatmap, OsuDB.beatmap_sql_adaptor)
        # # sadly this doesn't work
        # sqlite3.register_converter(slider.Beatmap.__name__, OsuTrainDB.beatmap_sql_converter)
        super(OsuDB, self).__init__(db_path, sqlite3.PARSE_COLNAMES, connect=connect)

    def filter_data(self, data_filter: filter.HitObjectFilter, save_dir):
        if 'FILTERED' in self.all_view_names():
            if input('FILTERED view already exist, deleted old view? y/n') in ('y', 'Y'):
                self.delete_view('FILTERED')
            else:
                return
        # ID is on pos 0, BEATMAP is on pos 3, AUDIOFILENAME is on pos 4
        keep_sample_idx = [sample[0] for sample in self.get_table_records('MAIN')
                           if data_filter.filter(pickle.loads(sample[3]), os.path.join(save_dir, sample[4]))]
        self.create_view_from_rows('ID', keep_sample_idx, 'FILTERED', 'MAIN')
        print('filter complete')

    def gen_preprocessed(self, data_preprocessor: Union[preprocessor.OsuAudioFilePreprocessor,
                                                        preprocessor.OsuAudioPreprocessor],
                         save_dir: str = None,
                         use_beatmap_list: bool = False,
                         osu_songs_dir: str = prepare_data_util.OsuSongsDir.DEFAULT_OSU_SONGS_DIR,
                         beatmap_list: list[slider.Beatmap] = None,
                         from_audio_path_list: str = None,
                         clear_table=True,
                         save_ext=None):
        """
        If use_beatmap_list, preprocess audio files using Beatmaps in beatmap_list.
        """
        columns = OsuDB.DEFAULT_COLUMNS + data_preprocessor.EXTRA_COLUMNS
        columns_type = OsuDB.DEFAULT_COLUMNS_TYPE + data_preprocessor.EXTRA_COLUMNS_TYPE
        self._try_create_main_table(columns, columns_type, clear_table)

        # used to record extra columns of preprocessed audio file
        # also help prevent preprocessing same audio again
        audio_extra_columns = dict()
        start_id = self.record_num('MAIN')
        current_id = start_id
        if use_beatmap_list:
            # most likely during inference data preparation
            assert from_audio_path_list is not None
            assert beatmap_list is not None
            for beatmap, audio_from_path in zip(beatmap_list, from_audio_path_list):
                if not os.path.exists(audio_from_path):
                    print('audio %s does not exist!' % audio_from_path)
                    continue
                if save_ext is not None:
                    save_audio_filename = general_util.change_ext(beatmap.audio_filename, save_ext)
                else:
                    save_audio_filename = beatmap.audio_filename
                beatmap.audio_filename = save_audio_filename
                audio_to_path = os.path.join(save_dir, beatmap.audio_filename)
                if audio_from_path in audio_extra_columns:
                    self.insert_row([current_id,
                                     beatmap.beatmap_id,
                                     beatmap.beatmap_set_id,
                                     beatmap,
                                     save_audio_filename,
                                     *audio_extra_columns[audio_from_path]])
                else:
                    try:
                        extra_columns = data_preprocessor.preprocess(beatmap, audio_from_path, audio_to_path)
                    except:
                        traceback.print_exc()
                        print('fail to preprocess, skipping: %d_%d' % (beatmap.beatmap_set_id, beatmap.beatmap_id))
                        print('audio_from_path: %s' % audio_from_path)
                        print('audio_to_path: %s' % audio_to_path)
                        continue
                    audio_extra_columns[audio_from_path] = extra_columns
                    self.insert_row([current_id,
                                     beatmap.beatmap_id,
                                     beatmap.beatmap_set_id,
                                     beatmap,
                                     save_audio_filename,
                                     *extra_columns])
                current_id += 1
        else:
            # most likely during train data preparation
            dir_obj = prepare_data_util.OsuSongsDir(osu_songs_dir)
            for beatmapset_dirname, beatmapset_dir_path, osu_filename_list, osu_file_path_list in dir_obj.beatmapsets():
                print('processing %s' % beatmapset_dirname)
                beatmaps = []
                # extract Beatmap from .osu files first, ensure that .osu and corresponding Beatmap is valid
                for osu_file_path in osu_file_path_list:
                    try:
                        beatmap = slider.Beatmap.from_path(osu_file_path)
                        if beatmap_util.check_essential_fields(beatmap):
                            beatmaps.append(beatmap)
                        else:
                            print('beatmap from %s is missing essential fields, skipped!' % osu_file_path)
                    except:
                        print('parsing .osu failed for %s' % osu_file_path)
                        continue
                beatmapset_audio_id = None
                for beatmap in beatmaps:
                    if beatmapset_audio_id is None or beatmapset_audio_id == 0:
                        audio_to_filename = '%d.mp3' % beatmap.beatmap_set_id
                    else:
                        audio_to_filename = '%d_%d.mp3' % (beatmap.beatmap_set_id, beatmapset_audio_id)
                    print('version %s' % beatmap.version)
                    audio_from_path = os.path.join(beatmapset_dir_path, beatmap.audio_filename)
                    # audio data is shared among beatmaps in a beatmapset and has been preprocessed
                    if audio_from_path in audio_extra_columns:
                        self.insert_row([current_id,
                                         beatmap.beatmap_id,
                                         beatmap.beatmap_set_id,
                                         beatmap,
                                         audio_to_filename,
                                         *(audio_extra_columns[audio_from_path])])
                    else:
                        if beatmapset_audio_id is None:
                            # the first audio for this beatmapset
                            beatmapset_audio_id = 0
                        else:
                            beatmapset_audio_id += 1
                        if beatmapset_audio_id != 0:
                            print('audio file No.%d found for beatmapset %d' % (beatmapset_audio_id, beatmap.beatmap_set_id))
                        audio_to_path = os.path.join(save_dir, audio_to_filename)
                        try:
                            extra_columns = data_preprocessor.preprocess(beatmap, audio_from_path, audio_to_path)
                        except:
                            traceback.print_exc()
                            print('fail to preprocess, skipping: %d_%d' % (beatmap.beatmap_set_id, beatmap.beatmap_id))
                            print('audio_from_path: %s' % audio_from_path)
                            print('audio_to_path: %s' % audio_to_path)
                            continue
                        audio_extra_columns[audio_from_path] = extra_columns
                        self.insert_row([current_id,
                                         beatmap.beatmap_id,
                                         beatmap.beatmap_set_id,
                                         beatmap,
                                         audio_to_filename,
                                         *extra_columns])
                    current_id += 1
        print('preprocess complete')
        return start_id, current_id

    def _try_create_main_table(self, columns, columns_type, clear_table):
        if clear_table or ('MAIN' not in self.all_table_names()):
            try:
                self.create_table(columns, columns_type, 'MAIN')
            except sqlite3.OperationalError:
                if input('MAIN table already exist, deleted old table? y/n') in ('y', 'Y'):
                    self.delete_table('MAIN')
                    self.create_table(columns, columns_type, 'MAIN')
                else:
                    return

    def _get_header_columns_by_preprocessor(self, data_preprocessor):
        data_in_db = isinstance(data_preprocessor, preprocessor.OsuAudioPreprocessor)
        if not data_in_db:
            columns = OsuDB.DEFAULT_COLUMNS
            columns_type = OsuDB.DEFAULT_COLUMNS_TYPE
        else:
            columns = OsuDB.DEFAULT_COLUMNS_DATA
            columns_type = OsuDB.DEFAULT_COLUMNS_DATA_TYPE
        columns += data_preprocessor.EXTRA_COLUMNS
        columns_type += data_preprocessor.EXTRA_COLUMNS_TYPE
        return data_in_db, columns, columns_type

    def gen_preprocessed_from_db(self, data_preprocessor: Union[preprocessor.OsuAudioFilePreprocessor,
                                                        preprocessor.OsuAudioPreprocessor],
                                 from_db_path: str,
                                 save_dir: str = None,
                                 from_dir: str = None,
                                 clear_table=True,
                                 save_ext=None):
        data_in_db, columns, columns_type = self._get_header_columns_by_preprocessor(data_preprocessor)
        self._try_create_main_table(columns, columns_type, clear_table)

        ref_db = OsuDB(from_db_path)
        total_samples = ref_db.record_num('MAIN')
        # ref_columns = ref_db.table_info('MAIN')[0]
        # ref_extra_columns = ref_columns[OsuDB.DEFAULT_COLUMNS_NUM:]

        start_id = self.record_num('MAIN')
        current_id = start_id
        audio_extra_columns = dict()
        for ref_current_id in range(total_samples):
            row = ref_db.get_row('ID', ref_current_id, 'MAIN')
            if not data_in_db:
                # pos 4 saves audio path
                audio_filename = row[4]
                if save_ext is not None:
                    save_audio_filename = general_util.change_ext(audio_filename, save_ext)
                else:
                    save_audio_filename = audio_filename
                # pos 3 saves pickled Beatmap
                beatmap = pickle.loads(row[3])
                audio_from_path = os.path.join(from_dir, audio_filename)
                if not os.path.exists(audio_from_path):
                    print('audio %s does not exist!' % audio_from_path)
                    continue
                audio_to_path = os.path.join(save_dir, save_audio_filename)
                beatmap.audio_filename = save_audio_filename
                if audio_from_path in audio_extra_columns:
                    print('%s already processed' % audio_filename)
                    self.insert_row([current_id,
                                     beatmap.beatmap_id,
                                     beatmap.beatmap_set_id,
                                     beatmap,
                                     save_audio_filename,
                                     *audio_extra_columns[audio_from_path]])
                else:
                    try:
                        print('processing %s' % audio_filename)
                        extra_columns = data_preprocessor.preprocess(beatmap, audio_from_path, audio_to_path)
                    except:
                        traceback.print_exc()
                        print('fail to preprocess, skipping: %d_%d' % (beatmap.beatmap_set_id, beatmap.beatmap_id))
                        print('audio_from_path: %s' % audio_from_path)
                        print('audio_to_path: %s' % audio_to_path)
                        continue
                    audio_extra_columns[audio_from_path] = extra_columns
                    self.insert_row([current_id,
                                     beatmap.beatmap_id,
                                     beatmap.beatmap_set_id,
                                     beatmap,
                                     save_audio_filename,
                                     *extra_columns])
            else:
                raise NotImplementedError
            current_id += 1
        print('preprocess complete')
        return start_id, current_id

    def split_folds(self, data_fold_divider: fold_divider.OsuTrainDBFoldDivider):
        """
        Only create views of database.
        """
        all_id = [sample[0] for sample in self.get_table_records('FILTERED')]
        for fold_idx, (train_idx, test_idx) in enumerate(data_fold_divider.div_folds(self.record_num('FILTERED'))):
            train_id = [all_id[idx] for idx in train_idx]
            test_id = [all_id[idx] for idx in test_idx]
            self.create_view_from_rows('ID', train_id, 'TRAINFOLD%d' % (fold_idx + 1), 'MAIN')
            self.create_view_from_rows('ID', test_id, 'TESTFOLD%d' % (fold_idx + 1), 'MAIN')
        print('split folds complete')

    def create_inference_view(self, id_list):
        if 'INFERENCE' in self.all_view_names():
            print('deleted old INFERENCE view')
            self.delete_view('INFERENCE')
        self.create_view_from_rows(
            'ID', id_list, 'INFERENCE', 'MAIN'
        )
