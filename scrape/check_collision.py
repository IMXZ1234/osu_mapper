import json
import os


period_meta_dir = r'C:\Users\asus\coding\python\osu_mapper\resources\data\osz'
agg_meta_filepath = r'C:\Users\asus\coding\python\osu_mapper\resources\data\meta20230530.json'
agg_meta_dict = {}
all_beatmapsetids = set()
all_beatmapids = set()
for meta_filename in os.listdir(period_meta_dir):
    with open(os.path.join(period_meta_dir, meta_filename), 'r') as f:
        meta_dict_list = json.load(f)
    for beatmap_meta_dict in meta_dict_list:
        beatmap_id, beatmapset_id = beatmap_meta_dict['beatmap_id'], beatmap_meta_dict['beatmapset_id']
        if beatmap_id in agg_meta_dict and beatmapset_id not in all_beatmapsetids:
            print('collision %d %d' % (beatmap_id, beatmapset_id))
        all_beatmapids.add(beatmap_id)
        all_beatmapsetids.add(beatmapset_id)
        agg_meta_dict[beatmap_id] = beatmap_meta_dict

print(len(agg_meta_dict))
with open(agg_meta_filepath, 'w') as f:
    json.dump(agg_meta_dict, f)
