import os
import pickle
import zipfile

from preprocess.dataset import embedding_datasetv7
from gen.interpreter import remap
import slider
import sys
import traceback

from util import beatmap_util

if __name__ == '__main__':
    save_dir = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\preprocessed_v8'
    """
    generate preprocessed
    """
    mel_args = {
        'sample_rate': 22000,
        'mel_frame_per_snap': 8,
        'n_fft': 400,
        'n_mels': 40,
    }
    feature_args = {
        'beat_divisor': 8,
        'off_snap_threshold': 0.15,
    }
    ds = embedding_datasetv7.HeatmapDataset(
        mel_args, feature_args
    )
    
    # for osz_path in [
    #     r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\beatmapsets\1000078.osz',
    # ]:
    #     beatmapset_id = os.path.splitext(os.path.basename(osz_path))[0]
    #     osz_file = zipfile.ZipFile(osz_path, 'r')
    #     try:
    #         all_beatmaps = list(slider.Beatmap.from_osz_file(osz_file).values())
    #     except Exception:
    #         traceback.print_exc()
    #         print('beatmap parse failed %s' % beatmapset_id)
    #         sys.exit(-1)
    #     for beatmap in all_beatmaps:
    #         beatmap_id = str(beatmap.beatmap_id)
    #         ds.process_beatmap_meta_dict(
    #             {'beatmapset_id': beatmapset_id, 'beatmap_id': beatmap_id},
    #             save_dir=save_dir,
    #             osz_dir=os.path.dirname(osz_path),
    #             temp_dir=save_dir,
    #         )

    # sys.exit()
    """
    generate beatmap from preprocessed
    """
    osu_dir = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\generate\osu'
    osz_path = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\generate\osz\1000078.osz'
    audio_file_path = r'C:\Users\admin\Desktop\python_project\osu_mapper\resources\data\beatmapsets\1000078\audio.mp3'
    meta_dict = {
        'audio_filename': os.path.basename(audio_file_path),  # indispensable
        'artist_unicode': 'ai',
        'artist': 'ai',  # indispensable
        'title_unicode': 'ai',
        'title': 'ai',  # indispensable
        'version': '8',  # indispensable
        'creator': 'IMXZ123',
        'circle_size': 3,
        'approach_rate': 8,
        'slider_tick_rate': 2,
        'hp_drain_rate': 5,
        'overall_difficulty': 7,
        'star': 4.5,
    }
    mel_dir = os.path.join(save_dir, 'mel')
    meta_dir = os.path.join(save_dir, 'meta')
    label_dir = os.path.join(save_dir, 'label')
    info_dir = os.path.join(save_dir, 'info')
    label_idx_dir = os.path.join(save_dir, 'label_idx')

    osu_path_list = []
    beatmap_list = []
    for beatmapid in [
        '2092306',
        '2093042',
        '2093043',
        '2094354',
    ]:
        info_path = os.path.join(info_dir, '%s.pkl' % beatmapid)
        with open(info_path, 'rb') as f:
            info = pickle.load(f)

        meta_path = os.path.join(meta_dir, '%s.pkl' % beatmapid)
        label_path = os.path.join(label_dir, '%s.pkl' % beatmapid)
        with open(meta_path, 'rb') as f:
            try:
                meta = pickle.load(f)
            except Exception:
                print('unable to read %s' % meta_path)
        with open(label_path, 'rb') as f:
            # n_snaps, 3(hit_object_type, x, y)
            try:
                label = pickle.load(f)
            except Exception:
                print('unable to read %s' % label_path)

        total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap, snaps_before_first_tp, start_s = info
        star, cs, ar, od, hp, bpm = meta
        meta_dict['circle_size'] = round(cs)
        meta_dict['approach_rate'] = round(ar)
        meta_dict['overall_difficulty'] = round(od)
        meta_dict['hp_drain_rate'] = round(hp)
        meta_dict['version'] = beatmapid
        beatmap = beatmap_util.get_empty_beatmap()
        beatmap_util.set_bpm(beatmap, bpm, 8)
        beatmap_util.set_meta(
            beatmap,
            meta_dict
        )
        remap.ReMapInterpreter().gen_hitobjects(
            beatmap,
            label,
            bpm,
            start_ms=start_s * 1000,
        )

        beatmap_name = beatmap_util.osu_filename(beatmap)
        osu_path = os.path.join(osu_dir, beatmap_name)
        beatmap.write_path(osu_path)
        osu_path_list.append(osu_path)
        beatmap_list.append(beatmap)

    beatmap_util.pack_to_osz(audio_file_path, osu_path_list, osz_path, beatmap_list)
