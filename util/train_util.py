import collections


class MultiStepScheduler:
    def __init__(self, milestones, milestone_output):
        """
        milestones = [5, 10, 15]
        milestone_output = [0.1, 0.5, 0.7, 0.9]

        then:
        if step < 5, cur_milestone_output() = 0.1
        if 5 < step < 10, cur_milestone_output() = 0.5
        ...
        if 15 < step, cur_milestone_output() = 0.9
        """
        self.milestones = milestones
        self.milestone_output = milestone_output
        assert len(self.milestones) == len(self.milestone_output) - 1
        self.current_step = 0
        self.cur_milestone_idx = 0

    def step(self):
        self.current_step += 1
        if self.cur_milestone_idx < len(self.milestones):
            if self.current_step > self.milestones[self.cur_milestone_idx]:
                self.cur_milestone_idx += 1

    def reset(self):
        self.cur_milestone_idx = 0
        self.current_step = 0

    def cur_milestone_output(self):
        return self.milestone_output[self.cur_milestone_idx]


def idx_set_with_uniform_itv(length, num, offset=0):
    itv = length / num
    idx_offset = offset % round(itv)
    return set([round(itv * i) + idx_offset for i in range(num)])


class AvgLossLogger:
    def __init__(self, win_len=10):
        self.win_len = win_len
        self.loss_value_history = collections.deque()
        self.avg = 0

    def clear(self):
        self.avg = 0
        self.loss_value_history.clear()

    def append(self, value_in):
        if self.win_len < 0 or len(self.loss_value_history) < self.win_len:
            self.avg = self.avg * len(self.loss_value_history) / (len(self.loss_value_history) + 1) + value_in / (len(self.loss_value_history) + 1)
        else:
            value_out = self.loss_value_history.popleft()
            self.avg += (value_in - value_out) / self.win_len
        self.loss_value_history.append(value_in)

    def __str__(self):
        return str(self.avg)
