from datetime import timedelta

import slider


def hit_objects_to_label(beatmap, aligned_start_time, snap_per_microsecond, total_snap_num,
                         align_to_snaps=None, multi_label=False, include_holdnote=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    # print(audio_start_time_offset)
    if align_to_snaps is not None:
        if total_snap_num % align_to_snaps != 0:
            total_snap_num = (total_snap_num // align_to_snaps + 1) * align_to_snaps
    label = [0 for _ in range(total_snap_num)]
    for hit_obj in beatmap.hit_objects():
        snap_idx = (hit_obj.time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        snap_idx = round(snap_idx)
        if snap_idx >= total_snap_num:
            print('snap index %d out of bound! total snap num %d' % (snap_idx, total_snap_num))
            snap_idx = total_snap_num
        if isinstance(hit_obj, slider.beatmap.Circle):
            label[snap_idx] = 1
            continue
        end_snap_idx = (hit_obj.end_time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        if end_snap_idx >= total_snap_num:
            print('end snap index %d out of bound! total snap num %d' % (end_snap_idx, total_snap_num))
            end_snap_idx = total_snap_num
        if multi_label:
            if isinstance(hit_obj, slider.beatmap.Slider):
                label_value = 2
            elif isinstance(hit_obj, slider.beatmap.Spinner):
                label_value = 3
            else:
                # HoldNote
                if include_holdnote:
                    label_value = 4
                else:
                    label_value = 1
        else:
            label_value = 1
        end_snap_idx = round(end_snap_idx)
        for i in range(snap_idx, end_snap_idx + 1):
            label[i] = label_value
    return label
