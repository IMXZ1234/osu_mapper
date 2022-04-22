from bisect import bisect


class IntervalList(list):
    """
    list of intervals, open/close is not considered
    """
    def __init__(self, interval_list=()):
        super().__init__()
        # left bounds, right bounds of intervals
        # self.lbs[i] < self.rbs[i] < self.lbs[i+1] < self.rbs[i+1] is always satisfied
        self.lbs, self.rbs = list(zip(*interval_list))
        self.lbs, self.rbs = list(self.lbs), list(self.rbs)

    def add_interval(self, interval: (tuple, list)):
        # adds at most one new interval
        # possibly removes multiple intervals
        lb_in, rb_in = interval
        # pos of lb_in in self.lbs (if inserted)
        ll_pos = bisect(self.lbs, lb_in)
        # pos of lb_in in self.rbs (if inserted)
        lr_pos = bisect(self.rbs, lb_in)
        # pos of rb_in in self.lbs (if inserted)
        rl_pos = bisect(self.lbs, rb_in)
        # pos of rb_in in self.rbs (if inserted)
        rr_pos = bisect(self.rbs, rb_in)
        if ll_pos == lr_pos:
            # lb_in not inside any interval in original interval list
            new_lb = lb_in
        else:
            # lb_in inside interval with idx ll_pos in original interval list
            # ll_pos = lr_pos + 1
            new_lb = self.lbs[lr_pos]
        rm_pos_start = lr_pos
        if rl_pos == rr_pos:
            # rb_in not inside any interval in original interval list
            new_rb = rb_in
            # not included in intervals to be popped
        else:
            # rb_in inside interval with idx ll_pos in original interval list
            # rl_pos = rr_pos + 1
            new_rb = self.rbs[rr_pos]
        rm_pos_end = rl_pos
        # print('new_lb')
        # print(new_lb)
        # print('new_rb')
        # print(new_rb)
        # print('rm_pos_start')
        # print(rm_pos_start)
        # print('rm_pos_end')
        # print(rm_pos_end)
        # print(self.lbs[rm_pos_start])
        # print(self.lbs)
        self.lbs = self.lbs[:rm_pos_start] + [new_lb] + self.lbs[rm_pos_end:]
        self.rbs = self.rbs[:rm_pos_start] + [new_rb] + self.rbs[rm_pos_end:]

    def remove_interval(self, interval: (tuple, list)):
        # adds at most two new intervals
        # possibly removes multiple intervals
        lb_in, rb_in = interval
        # pos of lb_in in self.lbs (if inserted)
        ll_pos = bisect(self.lbs, lb_in)
        # pos of lb_in in self.rbs (if inserted)
        lr_pos = bisect(self.rbs, lb_in)
        # pos of rb_in in self.lbs (if inserted)
        rl_pos = bisect(self.lbs, rb_in)
        # pos of rb_in in self.rbs (if inserted)
        rr_pos = bisect(self.rbs, rb_in)
        # does operation produces a new interval at lb_in/rb_in
        l_new, r_new = False, False
        if ll_pos != lr_pos:
            # lb_in inside interval with idx ll_pos in original interval list
            l_new = True
            # ll_pos = lr_pos + 1
            l_new_lb = self.lbs[lr_pos]
            l_new_rb = lb_in
        rm_pos_start = lr_pos
        if rl_pos != rr_pos:
            # rb_in inside interval with idx ll_pos in original interval list
            r_new = True
            # rl_pos = rr_pos + 1
            r_new_lb = rb_in
            r_new_rb = self.rbs[rr_pos]
        rm_pos_end = rl_pos
        new_lbs = self.lbs[:rm_pos_start]
        new_rbs = self.rbs[:rm_pos_start]
        if l_new:
            new_lbs += [l_new_lb]
            new_rbs += [l_new_rb]
        if r_new:
            new_lbs += [r_new_lb]
            new_rbs += [r_new_rb]
        new_lbs += self.lbs[rm_pos_end:]
        new_rbs += self.rbs[rm_pos_end:]
        self.lbs = new_lbs
        self.rbs = new_rbs

    def total_len(self):
        return sum(rb - lb for rb, lb in zip(self.rbs, self.lbs))

    def interval_list(self):
        return list(zip(self.lbs, self.rbs))

    def map_len_to_value(self, len_from_leftmost):
        for lb, rb in zip(self.lbs, self.rbs):
            itv_len = rb - lb
            if len_from_leftmost > itv_len:
                len_from_leftmost -= itv_len
            else:
                return lb + len_from_leftmost

    def __len__(self):
        return len(self.lbs)

    def __getitem__(self, item):
        return self.lbs[item], self.rbs[item]

    def __str__(self):
        return str(self.interval_list())


if __name__ == '__main__':
    itv_list = IntervalList([(3, 4), (5, 6), (7, 8)])
    print(itv_list)
    itv_list.add_interval((4.5, 7.5))
    print(itv_list)
    itv_list.remove_interval((4.25, 7.5))
    print(itv_list)
