from datetime import timedelta
from itertools import chain

from gen.position_generator import RandomWalkInRectangle
from util import beatmap_util


class BiLabelInterpreter:
    @staticmethod
    def gen_hitobjects(beatmap, label, start_time, snap_microseconds, snap_divisor=8):
        label = label.numpy().tolist()

        def periods():
            """
            yields (current position, nonzero period, trailing zero period length)
            """
            pos = 0
            next_pos = 0
            stop = False
            while True:
                # skip prepending zeros
                while label[next_pos] == 0:
                    next_pos += 1
                    if next_pos >= len(label):
                        stop = True
                        break
                if stop:
                    break
                zero_num = next_pos - pos
                pos = next_pos
                while label[next_pos] > 0:
                    next_pos += 1
                    if next_pos >= len(label):
                        stop = True
                        break
                if stop:
                    break
                yield pos, label[pos:next_pos], zero_num
                pos = next_pos

        # save some space about the border as circles have radius
        pos_gen = RandomWalkInRectangle(30, 482, 30, 354)
        pos_gen.move_to_random_pos()
        pos_gen.set_walk_dist_range(50, 150)
        # min_dist, max_dist = 0, -1
        # pin_dist = False

        ms_per_beat = 60000 / beatmap.bpm_min()
        last_slider_time = start_time - 2 * snap_microseconds / 1000
        for pos, period, trailing_zero_period_len in periods():
            time = (start_time + pos * snap_microseconds) / 1000
            if len(period) > 1:
                num_beats = len(period) / snap_divisor
                if round(num_beats) == int(num_beats) + 1:
                    # we only generate linear sliders for simplicity
                    pos_list = [
                        pos_gen.next_pos()
                        for _ in range(max(2, int(num_beats)))
                    ]

                    ho_slider = beatmap_util.add_slider(
                        beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                    )
                    last_slider_time = ho_slider.end_time // timedelta(microseconds=1)
                    print(beatmap._hit_objects[-1].curve.points)
                    print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
                # else:
                #     # combo circles
                #     pos_gen.set_walk_dist_range(0, 0)
                #     for i in range(len(period)):
                #         circle_time = time + i * snap_microseconds / 1000
                #         x, y = pos_gen.next_pos()
                #         beatmap_util.add_circle(beatmap, (x, y), circle_time)
                #         print(('add combo circle at (%.3f, (%d, %d))' % (circle_time, x, y)))
                #     pos_gen.set_walk_dist_range(50, 150)
            else:
                # avoid generating circles too close to the end_time of last slider
                # if (time - last_slider_time) >= snap_microseconds * 1.5 / 1000:
                x, y = pos_gen.next_pos()
                beatmap_util.add_circle(beatmap, (x, y), time)
                print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))


class MultiLabelInterpreter:
    @staticmethod
    def gen_hitobjects(beatmap, labels, start_time, snap_ms, snap_divisor=8):
        # save some space about the border as circles have radius
        pos_gen = RandomWalkInRectangle(30, 482, 30, 354)
        pos_gen.move_to_random_pos()
        pos_gen.set_walk_dist_range(50, 150)
        # min_dist, max_dist = 0, -1
        # pin_dist = False

        def periods():
            """
            yields (current position, nonzero period, trailing zero period length)
            """
            pos = 0
            next_pos = 0
            while True:
                pos = next_pos
                next_pos += 1
                if pos >= len(labels):
                    break
                # skip prepending zeros
                if labels[pos] == 2:
                    if next_pos >= len(labels):
                        break
                    while labels[next_pos] == 2:
                        next_pos += 1
                        if next_pos >= len(labels):
                            break
                    yield pos, next_pos - pos, 2
                elif labels[pos] == 1:
                    yield pos, 1, 1

        ms_per_beat = 60000 / beatmap.bpm_min()
        for pos, period, label in periods():
            # print('pos, period, label')
            # print(pos, period, label)
            time = start_time + pos * snap_ms
            if label == 1:
                x, y = pos_gen.next_pos()
                beatmap_util.add_circle(beatmap, (x, y), time)
                print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                num_beats = period // snap_divisor
                if num_beats == 0:
                    continue
                pos_list = [
                    pos_gen.next_pos()
                    for _ in range(num_beats + 1)
                ]

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))


class SwitchInterpreter:
    @staticmethod
    def gen_hitobjects(beatmap, labels, start_time, snap_ms, snap_divisor=8):
        # save some space about the border as circles have radius
        pos_gen = RandomWalkInRectangle(30, 482, 30, 354)
        pos_gen.move_to_random_pos()
        pos_gen.set_walk_dist_range(50, 150)
        # min_dist, max_dist = 0, -1
        # pin_dist = False

        def periods():
            """
            yields (current position, nonzero period, trailing zero period length)
            """
            pos = 0
            next_pos = 0
            while True:
                pos = next_pos
                next_pos += 1
                if pos >= len(labels):
                    break
                if labels[pos] == 2:
                    yield pos, next_pos - pos, labels[pos]
                elif labels[pos] >= 3:
                    if next_pos >= len(labels):
                        break
                    while labels[next_pos] == 0:
                        # keep state
                        next_pos += 1
                        if next_pos >= len(labels):
                            break
                    yield pos, next_pos - pos, labels[pos]

        ms_per_beat = 60000 / beatmap.bpm_min()
        end_time = 0
        for pos, period, label in periods():
            # print('pos, period, label')
            # print(pos, period, label)
            time = start_time + pos * snap_ms
            if time <= end_time:
                continue
            if label == 2:
                # circles, period == 1
                x, y = pos_gen.next_pos()
                beatmap_util.add_circle(beatmap, (x, y), time)
                # print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 3:
                # slider start
                num_beats = max(1, period // snap_divisor)
                # if num_beats == 0:
                #     continue
                pos_list = [
                    pos_gen.next_pos()
                    for _ in range(num_beats)
                ]

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                # print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
                end_time = ho_slider.end_time // timedelta(milliseconds=1)
