import random
import math
from bisect import bisect

from util.math_util import IntervalList


class RandomWalkInRectangle:
    def __init__(self, left, right, top, bottom, start_x=None, start_y=None):
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.x, self.y = start_x, start_y
        if start_x is None:
            start_x = (self.left + self.right) // 2
        if start_y is None:
            start_y = (self.top + self.bottom) // 2
        self.move_to_pos(start_x, start_y)
        self.min_dist, self.max_dist = -1, -1
        self.set_walk_dist_range()

    def move_to_random_pos(self):
        self.x = random.randint(self.left, self.right - 1)
        self.y = random.randint(self.top, self.bottom - 1)

    def set_walk_dist_range(self, min_dist=0, max_dist=-1):
        """
        [min_dist, max_dist),
        not strictly ensured since pos will be round to int
        """
        self.min_dist = max(0, min_dist)
        if max_dist < 0:
            dist_to_bound_list = [
                self.x - self.left,
                self.right - self.x,
                self.y - self.top,
                self.bottom - self.y,
            ]
            dist_to_bound_list.sort()
            self.max_dist = math.sqrt(
                dist_to_bound_list[-1]**2 + dist_to_bound_list[-2]**2
            )
        else:
            self.max_dist = max(self.min_dist, max_dist)

    def move_to_pos(self, x, y):
        assert self.left <= x <= self.right
        assert self.top <= y <= self.bottom
        self.x, self.y = x, y

    def next_pos(self):
        """
        Only generate next pos inside the rectangle(screen).
        Distance to last pos will be randomly chosen between [self.max_dist, self.min_dist).
        Walk direction will be randomly chosen among valid angle range to avoid walking out of rectangular bound.
        """
        dist = random.random() * (self.max_dist - self.min_dist) + self.min_dist
        # 0 ~ 2*pi
        dist_to_bound_list = [
            self.x - self.left,
            self.right - self.x,
            self.y - self.top,
            self.bottom - self.y,
        ]
        invalid_angle_center = [
            math.pi, 0, math.pi / 2, math.pi * 3 / 2
        ]
        valid_angle_range_list = IntervalList([(0, 2 * math.pi)])

        for idx, dist_to_bound in enumerate(dist_to_bound_list):
            if abs(dist_to_bound) < dist:
                angle = math.acos(abs(dist_to_bound) / dist)
                center = invalid_angle_center[idx]
                if idx == 1:
                    valid_angle_range_list.remove_interval((0, center + angle))
                    valid_angle_range_list.remove_interval((2*math.pi - angle, 2*math.pi))
                else:
                    valid_angle_range_list.remove_interval((center - angle, center + angle))
        #     print('valid_angle_range_list')
        #     print(valid_angle_range_list)
        # print('valid_angle_range_list')
        # print([(l * 180 / math.pi, r * 180 / math.pi) for l, r in valid_angle_range_list.interval_list()])
        # remove invalid range of angle
        total_len = random.random() * valid_angle_range_list.total_len()
        angle = valid_angle_range_list.map_len_to_value(total_len)
        # print('angle')
        # print(angle * 180 / math.pi)
        x_offset = math.cos(angle) * dist
        y_offset = -math.sin(angle) * dist
        # print('x_offset')
        # print(x_offset)
        # print('y_offset')
        # print(y_offset)
        self.x = min(self.right, max(self.left, round(self.x + x_offset)))
        self.y = min(self.bottom, max(self.top, round(self.y + y_offset)))
        return self.x, self.y


if __name__ == '__main__':
    gen = RandomWalkInRectangle(0, 512, 0, 384)
    gen.set_walk_dist_range(5, 5)
    gen.move_to_pos(0, 0)
    print(gen.next_pos())
    print(gen.next_pos())
    print(gen.next_pos())
    print(gen.next_pos())