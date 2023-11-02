import collections
import math


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
            if self.current_step >= self.milestones[self.cur_milestone_idx]:
                self.cur_milestone_idx += 1

    def set_current_step(self, step):
        self.current_step = step
        self.cur_milestone_idx = 0
        while self.cur_milestone_idx < len(self.milestones) and self.current_step >= self.milestones[self.cur_milestone_idx]:
            self.cur_milestone_idx += 1

    def reset(self):
        self.cur_milestone_idx = 0
        self.current_step = 0

    def cur_milestone_output(self):
        return self.milestone_output[self.cur_milestone_idx]


class AbsCosineScheduler:
    def __init__(self, milestones, milestone_output, period):
        """
        milestones = [5, 10, 15]
        milestone_output = [0.1, 0.5, 0.7, 0.9]
        period: half period of underlying cos

        then:
        if step < 5, cur_milestone_output() = 0.1
        if 5 < step < 10, cur_milestone_output() = 0.5
        ...
        if 15 < step, cur_milestone_output() = 0.9
        """
        self.period = period
        self.milestones = milestones
        self.milestone_output = milestone_output
        assert len(self.milestones) == len(self.milestone_output) - 1
        self.current_step = 0
        self.cur_milestone_idx = 0

    def step(self):
        self.current_step += 1
        if self.cur_milestone_idx < len(self.milestones):
            if self.current_step >= self.milestones[self.cur_milestone_idx]:
                self.cur_milestone_idx += 1

    def set_current_step(self, step):
        self.current_step = step
        self.cur_milestone_idx = 0
        while self.cur_milestone_idx < len(self.milestones) and self.current_step >= self.milestones[self.cur_milestone_idx]:
            self.cur_milestone_idx += 1

    def reset(self):
        self.cur_milestone_idx = 0
        self.current_step = 0

    def cur_milestone_output(self):
        ampl = self.milestone_output[self.cur_milestone_idx]
        phase = (self.current_step % self.period) / self.period * math.pi
        return ampl * abs(math.cos(phase))


class BatchAbsCosineScheduler:
    def __init__(self, milestones, milestone_output, period=None):
        """
        milestones = [5, 10, 15]
        milestone_output = [0.1, 0.5, 0.7, 0.9]
        period: half period of underlying cos

        then:
        if step < 5, cur_milestone_output() = 0.1
        if 5 < step < 10, cur_milestone_output() = 0.5
        ...
        if 15 < step, cur_milestone_output() = 0.9
        """
        self.period = period
        self.milestones = milestones
        self.milestone_output = milestone_output
        assert len(self.milestones) == len(self.milestone_output) - 1
        self.current_step = 0
        self.current_batch_step = 0
        self.cur_milestone_idx = 0

    def step(self):
        self.current_step += 1
        self.current_batch_step = 0
        if self.cur_milestone_idx < len(self.milestones):
            if self.current_step >= self.milestones[self.cur_milestone_idx]:
                self.cur_milestone_idx += 1

    def set_current_step(self, step):
        self.current_step = step
        self.cur_milestone_idx = 0
        while self.cur_milestone_idx < len(self.milestones) and self.current_step >= self.milestones[self.cur_milestone_idx]:
            self.cur_milestone_idx += 1

    def step_batch(self):
        self.current_batch_step += 1

    def set_current_batch_step(self, batch_step):
        self.current_batch_step = batch_step

    def set_period(self, period):
        self.period = period

    def reset(self):
        self.cur_milestone_idx = 0
        self.current_step = 0

    def cur_milestone_output(self):
        ampl = self.milestone_output[self.cur_milestone_idx]
        if self.period is None:
            print('set period first! returned unmodulated ampl')
            return ampl
        phase = (self.current_batch_step % self.period) / self.period * math.pi
        return ampl * abs(math.cos(phase))


class BatchAbsCosineSchedulerMod:
    def __init__(self, y_coeff=1., x_coeff=0.1, period=None, low_bound=None):
        """
        e-x
        """
        self.period = period
        self.current_step = 0
        self.current_batch_step = 0
        self.y_coeff = y_coeff
        self.x_coeff = x_coeff
        self.low_bound = low_bound

    def step(self):
        self.current_step += 1
        self.current_batch_step = 0

    def set_current_step(self, step):
        self.current_step = step

    def step_batch(self):
        self.current_batch_step += 1

    def set_current_batch_step(self, batch_step):
        self.current_batch_step = batch_step

    def set_period(self, period):
        self.period = period

    def reset(self):
        self.current_step = 0

    def cur_milestone_output(self):
        ampl = self.y_coeff * math.exp(-self.x_coeff * self.current_step)
        if self.period is None:
            print('set period first! returned unmodulated ampl')
            return ampl
        phase = (self.current_batch_step % self.period) / self.period * math.pi
        output = ampl * abs(math.cos(phase))
        if self.low_bound is not None:
            output = max(self.low_bound, output)
        return output


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
