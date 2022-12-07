from datetime import timedelta
from itertools import chain

import slider

from gen.position_generator import RandomWalkInRectangle
from util import beatmap_util
import random


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


class AssembledLabelOutputInterpreter:
    @staticmethod
    def gen_hitobjects(beatmap: slider.Beatmap, labels, start_time, snap_ms, snap_divisor=8, stack_circle_prob=0.3):
        # save some space about the border as circles have radius
        pos_gen = RandomWalkInRectangle(30, 482, 30, 354)
        pos_gen.move_to_random_pos()
        pos_gen.set_walk_dist_range(50, 150)

        ms_per_beat = beatmap.timing_points[0].ms_per_beat
        print('ms_per_beat')
        print(ms_per_beat)
        print('ms_per_beat / snap_divisor')
        print(ms_per_beat / snap_divisor)
        print('snap_ms')
        print(snap_ms)
        snap_ms = ms_per_beat / snap_divisor
        print('snap_ms')
        print(snap_ms)
        pos = 0
        while pos < len(labels):
            label = labels[pos]
            # print('pos, period, label')
            # print(pos, period, label)
            time = start_time + pos * snap_ms
            if label == 1:
                # circles, period == 1
                x, y = pos_gen.next_pos()
                beatmap_util.add_circle(beatmap, (x, y), time)
                look_forward_pos = pos + 2
                trailing_circle_num = 0
                step_dist = random.choice([5, 10, 15, 20])
                while labels[look_forward_pos] == 1:
                    trailing_circle_num += 1
                    look_forward_pos += 2
                stack = False
                if 1 <= trailing_circle_num <= 3:
                    if random.uniform(0, 1) < stack_circle_prob:
                        stack = True
                elif trailing_circle_num > 3:
                    stack = True
                if stack:
                    pos_gen.set_walk_dist_range(step_dist, step_dist)
                for _ in range(trailing_circle_num):
                    time += snap_ms * 2
                    x, y = pos_gen.next_pos()
                    beatmap_util.add_circle(beatmap, (x, y), time)
                pos_gen.set_walk_dist_range(50, 150)
                pos = look_forward_pos
                # print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                start_pos = pos
                # slider start
                while pos < len(labels) and labels[pos] == 2:
                    pos += 1
                num_beats = max(0.5, (pos-start_pos) / snap_divisor)
                # if num_beats == 0:
                #     continue
                pos_list = [pos_gen.next_pos()]
                pos_gen.set_walk_dist_range(140*num_beats, 140*num_beats)
                pos_list.append(pos_gen.next_pos())
                pos_gen.set_walk_dist_range(50, 150)

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
                end_time = ho_slider.end_time // timedelta(milliseconds=1)
            else:
                pos += 1


class LabelWithPosInterpreter:
    @staticmethod
    def map_pos(x, y):
        x = int(x * (691 + 180) - 180)
        y = int(y * (407 + 82) - 82)
        return x, y

    @staticmethod
    def gen_hitobjects(beatmap: slider.Beatmap, labels, start_time, snap_ms, snap_divisor=8):
        ms_per_beat = beatmap.timing_points[0].ms_per_beat
        snap_ms = ms_per_beat / snap_divisor
        pos = 0
        while pos < len(labels):
            label_with_pos = labels[pos]
            label, x, y = int(label_with_pos[0]), label_with_pos[1], label_with_pos[2]
            x, y = LabelWithPosInterpreter.map_pos(x, y)
            time = start_time + pos * snap_ms
            if label == 1:
                # circles, period == 1
                beatmap_util.add_circle(beatmap, (x, y), time)
                pos += 1
                # print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                pos_list = [[x, y]]
                start_pos = pos + 1
                # slider start
                while pos < len(labels):
                    label_with_pos = labels[pos]
                    label, x, y = int(label_with_pos[0]), label_with_pos[1], label_with_pos[2]
                    x, y = LabelWithPosInterpreter.map_pos(x, y)
                    if label == 2:
                        pos_list.append([x, y])
                        pos += 1
                    else:
                        break
                if pos - start_pos < 2:
                    # bad slider
                    beatmap_util.add_circle(beatmap, (x, y), time)
                    continue
                num_beats = (pos-start_pos) / snap_divisor

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                # print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
            else:
                pos += 1


class DensityLabelWithPosInterpreter:
    @staticmethod
    def map_pos(x, y):
        x = int(x * (691 + 180) - 180)
        y = int(y * (407 + 82) - 82)
        return x, y

    @staticmethod
    def parse_step_label(step_label):
        circle_density, slider_density, x, y = step_label
        x, y = DensityLabelWithPosInterpreter.map_pos(x, y)
        if circle_density < 0.5 and slider_density < 0.5:
            return 0, x, y
        if circle_density > slider_density:
            return 1, x, y
        else:
            return 2, x, y

    @staticmethod
    def gen_hitobjects(beatmap: slider.Beatmap, labels, start_time, snap_ms, snap_divisor=8):
        ms_per_beat = beatmap.timing_points[0].ms_per_beat
        snap_ms = ms_per_beat / snap_divisor
        pos = 0
        while pos < len(labels):
            label, x, y = DensityLabelWithPosInterpreter.parse_step_label(labels[pos])
            time = start_time + pos * snap_ms
            if label == 1:
                # circles, period == 1
                beatmap_util.add_circle(beatmap, (x, y), time)
                pos += 1
                # print(('add circle at (%.3f, (%d, %d))' % (time, x, y)))
            elif label == 2:
                pos_list = [[x, y]]
                start_pos = pos + 1
                # slider start
                while pos < len(labels):
                    label, x, y = DensityLabelWithPosInterpreter.parse_step_label(labels[pos])
                    if label == 2:
                        pos_list.append([x, y])
                        pos += 1
                    else:
                        break
                if pos - start_pos < 2:
                    # bad slider
                    beatmap_util.add_circle(beatmap, (x, y), time)
                    continue
                num_beats = (pos-start_pos) / snap_divisor

                ho_slider = beatmap_util.add_slider(
                    beatmap, 'L', pos_list, time, num_beats, ms_per_beat
                )
                # print(('add slider at (%.3f, %s)' % (time, ' ({}, {})' * len(pos_list))).format(*chain(*pos_list)))
            else:
                pos += 1
