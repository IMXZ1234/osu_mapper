import csv
import math
import os
from typing import Any
import matplotlib.pyplot as plt
import time
from functools import wraps
import random
import collections

import numpy as np

test_cost_time_run_times = 100


def cost_time(func):
    @wraps(func)
    def with_cost_time(*args, **kwargs):
        result = None
        t0 = time.perf_counter()
        for _ in range(test_cost_time_run_times):
            result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print('%s cost %f sec' % (func.__name__, t1 - t0))
        return result
    return with_cost_time


def dynamic_time_warping(c, q, f=None):
    if f is None:
        f = lambda x, y: abs(x - y)
    # d = np.zeros([len(c), len(q)])
    # for i in range(len(c)):
    #     for j in range(len(q)):
    #         d[i, j] = f(c[i], q[j])

    d = [[f(c[i], q[j]) for j in range(len(q))] for i in range(len(c))]
    d = np.reshape(d, [len(c), len(q)])

    def g(i, j):
        if i == 0:
            return np.sum(d[0, :j + 1]), [(0, t) for t in range(j + 1)]
        elif j == 0:
            return np.sum(d[:i + 1, 0]), [(t, 0) for t in range(i + 1)]
        else:
            r = [g(i - 1, j), g(i - 1, j - 1), g(i, j - 1)]
            dis_list, path_list = list(zip(*r))
            min_dis_i = np.argmin(dis_list)
            return dis_list[min_dis_i] + d[i, j], path_list[min_dis_i] + [(i, j)]

    # min_dis, path = g(len(c) - 1, len(q) - 1)
    # return min_dis, path
    path = [[[] for j in range(len(q))] for i in range(len(c))]
    path[0][0] = [(0, 0)]
    for i in range(1, len(c)):
        path[i][0] = path[i - 1][0] + [(i, 0)]
    for j in range(1, len(q)):
        path[0][j] = path[0][j] + [(0, j)]
    for i in range(1, len(c)):
        for j in range(1, len(q)):
            dis_list = [d[i - 1, j - 1], d[i - 1, j], d[i, j - 1]]
            path_list = [path[i - 1][j - 1], path[i - 1][j], path[i][j - 1]]
            min_dis_i = np.argmin(dis_list)
            d[i, j] = dis_list[min_dis_i] + d[i, j]
            path[i][j] = path_list[min_dis_i] + [(i, j)]
    return d[len(c) - 1, len(q) - 1], path[len(c) - 1][len(q) - 1]


def AMPD(x):
    win_num = math.ceil(len(x) / 2) - 1
    m = np.zeros([win_num, len(x)])
    # window grows in size
    for j in range(1, win_num + 1):
        m[j-1, :j] = 1 + np.random.randn(j)
        m[j-1, len(x)-j:] = 1 + np.random.randn(j)
        for t in range(j, len(x)-j):
            if x[t] <= x[t-j] or x[t] <= x[t+j]:
                m[j-1, t] = 1 + np.random.randn()
    # the row with maximum zeros
    # 2 * j_min_sum(which is approximately win_len) is close to period length
    j_min_sum = np.argmin(np.sum(m, axis=1)) + 1
    # print(j_min_sum)
    peaks = []
    for t in range(len(x)):
        # m[:j_min_sum, t] is zero vector
        if len(np.where(m[:j_min_sum, t] != 0)[0]) == 0:
            peaks.append(t)
    return peaks


def append_to_list_of_list(element, list_of_list):
    """
    Append element to the right of every list in list_of_list.

    :param element:
    :param list_of_list:
    :return:
    """
    appended = []
    for lst in list_of_list:
        new_lst = [element]
        new_lst.extend(lst)
        appended.append(new_lst)
    return appended


def combination(m, n):
    """
    Get choices in C(m,n)

    example:

    C(3, 2) => [[1, 1, 0], [0, 1, 1], [1, 0, 1]]

    :param m:
    :param n:
    :return:
    """
    if n == 1:
        if m == 1:
            choice_list = [[1]]
        else:
            choice_list = []
            choice_base = [0 for _ in range(m)]
            for i in range(m):
                choice = choice_base.copy()
                choice[i] = 1
                choice_list.append(choice)
    elif m > n:
        choice_list = append_to_list_of_list(1, combination(m - 1, n - 1))  # select first
        no_sel_first = append_to_list_of_list(0, combination(m - 1, n))
        choice_list.extend(no_sel_first)
    else:
        choice_list = [[1 for _ in range(m)]]
    return choice_list


def rearrange_list(list_in, index_list, in_place=False):
    """
    Rearrange list according to given permutation specified by index_list.

    :param list_in:
    :param index_list:
    :param in_place: do permutation in place
    :return:
    """
    assert len(list_in) == len(index_list)
    if in_place:
        # map i to current pos of element with index i in the original list
        pos_map = list(range(len(index_list)))
        # the permutation of current sequence
        pos_record = list(range(len(index_list)))
        for i in range(len(index_list) - 1):
            src_pos = index_list[i]
            map_pos = pos_map[src_pos]
            list_in[i], list_in[map_pos] = list_in[map_pos], list_in[i]
            # update mapping
            pos_map[pos_record[i]] = map_pos
            pos_map[pos_record[map_pos]] = i
            # update current permutation
            pos_record[i], pos_record[map_pos] = pos_record[map_pos], pos_record[i]
        return list_in
    else:
        list_out = list_in.copy()
        for i in range(len(list_in)):
            list_out[i] = list_in[index_list[i]]
        return list_out


def map_to_order(list_in, in_place=False, reverse=False):
    """
    [33, 44, 11, 22] -> [2, 3, 0, 1]
    [33, 44, 30, 22] -> [2, 3, 1, 0]

    :param reverse: if reverse, larger item will be assigned smaller order
    :param in_place:
    :param list_in:
    :return:
    """
    argsort_list_in = np.argsort(list_in).tolist()
    if reverse:
        argsort_list_in = argsort_list_in.reverse()
    if not in_place:
        list_out = [None for i in range(len(argsort_list_in))]
        for i, pos in enumerate(argsort_list_in):
            list_out[pos] = i
        return list_out
    else:
        for i, pos in enumerate(argsort_list_in):
            list_in[pos] = i
        return list_in


def binary_search(list_in, item, f_comp=None, f_equal=None, insert=False):
    """
    Binary search in ordered list_in for the position of item.

    :param list_in:
    :param item:
    :param f_comp: comparison function, if None use basic_type_np_smaller_equal
    :param f_equal: equal function, if None use basic_type_np_equal
    :param insert: whether the item should be inserted at the searched position,
    will replace the original item in list_in which is equal to item
    :return: pos: item should be inserted at pos, note it is possible that pos == len(list_in);
     is_equal: is item equal to current item at pos
    """
    if f_comp is None:
        f_comp = basic_type_np_smaller_equal
    if f_equal is None:
        f_equal = basic_type_np_equal
    if len(list_in) == 0:
        return 0, False
    l = 0
    if f_comp(item, list_in[l]):
        is_equal = f_equal(item, list_in[l])
        if insert:
            if is_equal:
                list_in[l] = item
            else:
                list_in.insert(l, item)
        return l, is_equal
    r = len(list_in) - 1
    if not f_comp(item, list_in[r]):
        return len(list_in), False
    pos = (r - l) // 2 + l
    while True:
        # print(pos, l, r)
        if f_comp(item, list_in[pos]):
            r = pos
        else:
            l = pos + 1
        if l == r:
            is_equal = f_equal(item, list_in[r])
            if insert:
                if is_equal:
                    list_in[r] = item
                else:
                    list_in.insert(r, item)
            return r, is_equal
        else:
            pos = (r - l) // 2 + l


def random_ints(random_min, random_max, num, random_inst=None):
    """
    Random choose num ints in the range of [random_min, random_max),
    returned chosen ints are arranged from small to large(order is kept).

    :param random_inst: should be random.Random instance, if None, use random default instance
    :param random_min:
    :param random_max:
    :param num:
    :return:
    """
    assert num <= random_max - random_min
    if random_inst is None:
        r = random.randint(random_min, random_max - 1)
    else:
        r = random_inst.randint(random_min, random_max - 1)
    int_set = set()
    for _ in range(num):
        offset = 1
        while r in int_set:
            if r - offset > random_min and (r - offset not in int_set):
                r = r - offset
                break
            if r + offset < random_max and (r + offset not in int_set):
                r = r + offset
                break
            # print(r, offset)
            offset += 1
        int_set.add(r)
        if random_inst is None:
            r = random.randint(random_min, random_max - 1)
        else:
            r = random_inst.randint(random_min, random_max - 1)
    ret_list = list(int_set)
    ret_list.sort()
    return ret_list
    # ret_list = [None for _ in range(random_max - random_min)]
    # for _ in range(num):
    #     r = random.randint(0, random_max - random_min - 1)
    #     _, is_equal = binary_search([pos for pos in ret_list if pos is not None], r)
    #     offset = 1
    #     while is_equal:
    #         if r - offset > 0 and ret_list[r - offset] is None:
    #             r = r - offset
    #             break
    #         if r + offset < len(ret_list) and ret_list[r + offset] is None:
    #             r = r + offset
    #             break
    #         offset += 1
    #     ret_list[r] = r
    # ret_list = [pos + random_min for pos in ret_list if pos is not None]
    # return ret_list


def rearrange_lists(list_of_list, index_list, in_place=False):
    """
    Rearrange according to given permutation specified by index_list
    for every list in list_of_list.

    :param list_of_list:
    :param index_list:
    :param in_place:
    :return:
    """
    for lst in list_of_list:
        assert len(lst) == len(index_list)
    if in_place:
        # map i to current pos of element with index i in the original list
        pos_map = list(range(len(index_list)))
        # the permutation of current sequence
        pos_record = list(range(len(index_list)))
        for i in range(len(index_list) - 1):
            src_pos = index_list[i]
            map_pos = pos_map[src_pos]
            for lst in list_of_list:
                lst[i], lst[map_pos] = lst[map_pos], lst[i]
            # update mapping
            pos_map[pos_record[i]] = map_pos
            pos_map[pos_record[map_pos]] = i
            # update current permutation
            pos_record[i], pos_record[map_pos] = pos_record[map_pos], pos_record[i]
        return list_of_list
    else:
        new_list_of_list = [[] for _ in range(len(list_of_list))]
        for i, lst in enumerate(list_of_list):
            new_list_of_list[i] = lst.copy()
        for i, lst in enumerate(list_of_list):
            for j in range(len(lst)):
                new_list_of_list[i][j] = lst[index_list[j]]
        return new_list_of_list


def rearrange_list_inplace(list_in, index_list, allow_alter_index_list=False):
    assert len(list_in) == len(index_list)
    if not allow_alter_index_list:
        index_list = index_list.copy()
    for pos in range(len(list_in)):
        target_pos = index_list[pos]
        new_swap_dest = index_list[target_pos]
        if pos < target_pos:
            list_in[pos], list_in[target_pos] = list_in[target_pos], list_in[pos]
        elif pos > target_pos:
            list_in[pos], list_in[new_swap_dest] = list_in[new_swap_dest], list_in[pos]
            index_list[pos] = new_swap_dest
    return list_in


# @cost_time
def rearrange_list_inplace_1(list_in, index_list):
    assert len(list_in) == len(index_list)
    for pos in range(len(list_in) - 1):
        target_pos = index_list[pos]
        while target_pos < pos:
            target_pos = index_list[target_pos]
        list_in[pos], list_in[target_pos] = list_in[target_pos], list_in[pos]
    return list_in


# @cost_time
def rearrange_list_inplace_2(list_in, index_list):
    """
    Not preferable as is of higher time complexity, and only works when items in list_in are hashable.
    Written to demonstrate a possible usage of list.sort()

    :param list_in:
    :param index_list:
    :return:
    """
    assert len(list_in) == len(index_list)
    item_to_index = dict(zip(list_in, index_list))
    list_in.sort(key=lambda x: item_to_index[x])
    return list_in


def sort_lists(list_of_list, get_key_func=None, list_pos_reference=0, in_place=False):
    """

    :param list_of_list:
    :param get_key_func: should take in an item from the reference list and output the key
    :param list_pos_reference: position of the reference list which contains the key for the sorting
    :param in_place:
    :return:
    """
    # item is key itself
    if get_key_func is None:
        get_key_func = lambda item:item
    index_list = list(range(len(list_of_list[0])))
    # append index_list to record the permutation of sorted lists
    zipped = list(zip(*list_of_list, index_list))
    # print(zipped)
    zipped.sort(key=lambda x: get_key_func(x[list_pos_reference]))
    new_list_of_list = list(zip(*zipped))
    for i in range(len(new_list_of_list) - 1):
        new_list_of_list[i] = list(new_list_of_list[i])
    sorted_permutation = new_list_of_list[-1]
    if not in_place:
        return new_list_of_list[:-1], sorted_permutation
    else:
        return rearrange_lists(list_of_list, sorted_permutation, in_place=True), sorted_permutation


def combine_lists(list_of_list, interleave=False, fill_with_none=True):
    """
    Aggregate items from several lists into a single list.

    :param list_of_list:
    :param interleave: If False, lists are concatenated, else in every round, one item is fetched from each list and put into combined list.
    :param fill_with_none: Only functions when interleave=True, will pad None if some lists are not long enough.
    :return:
    """
    combined_list = []
    if not interleave:
        for i in range(len(list_of_list)):
            combined_list.extend(list_of_list[i])
    else:
        list_len = [len(list_of_list[i]) for i in range(len(list_of_list))]
        max_list_len = max(list_len)
        for j in range(max_list_len):
            for i in range(len(list_of_list)):
                if j < len(list_of_list[i]):
                    combined_list.append(list_of_list[i][j])
                elif fill_with_none:
                    combined_list.append(None)
    return combined_list


def delete_from_lists(list_of_list, del_list, ref_list_pos=0, f=None, in_place=False, delete_once=False):
    """
    For items in del_list, find corresponding(f called with two items as param and returns True)
    items' index in list_of_list[ref_list_pos], then:


    if not in_place:

    delete items with same position in all lists in copy of list_of_list
    and return deleted lists as list_of_list_out. list_of_list itself is not changed.


    if in_place:


    delete items with same position in all lists in list_of_list and return list_of_list.

    :param delete_once: if True, delete all corresponding items, else only delete one item(first encountered from the beginning of list_of_list[ref_list_pos])
    :param in_place:
    :param f: if None, use basic_type_np_equal
    :param list_of_list:
    :param del_list:
    :param ref_list_pos:
    :return:
    """
    if f is None:
        f = basic_type_np_equal
    ref_list = list_of_list(ref_list_pos)
    del_index_list = []
    for del_item in del_list:
        for ref_index, ref_item in enumerate(ref_list):
            if f(del_item, ref_item):
                del_index_list.append(ref_index)
                if delete_once:
                    break
    # we want more flexibility, so realization below is abandoned, which might be more efficient if re_list is very long
    # ref_dict = {}
    # for i in range(len(ref_list)):
    #     ref_dict[ref_list[i]] = i
    # del_index_list = [ref_dict[item] for item in del_list]
    del_index_list.sort()
    if in_place:
        list_of_list_out = list_of_list
    else:
        list_of_list_out = [_list.copy() for _list in list_of_list]
    for i in range(len(del_index_list) - 1, -1, -1):
        # pop from largest index
        for _list in list_of_list_out:
            _list.pop(i)
    return list_of_list_out


def list_with_appended_item(list_in, item):
    list_out = list_in.copy()
    list_out.append(item)
    return list_out


def list_with_left_appended_item(list_in, item):
    list_out = [item]
    list_out.extend(list_in.copy())
    return list_out


def lists_union_sorted1(list_of_list):
    """
    Slower version.

    :param list_of_list:
    :return:
    """
    list_num = len(list_of_list)
    current_idx_in_category = [0 for _ in range(list_num)]
    all_appended_category = list_num
    kept_idx = []
    current_cmp_label = [list_of_list[label_idx][0] for label_idx in range(list_num)]
    while all_appended_category > 0:
        min_category = np.argmin(current_cmp_label)
        kept_idx.append(current_cmp_label[min_category])
        current_idx_in_category[min_category] += 1
        if current_idx_in_category[min_category] == len(list_of_list[min_category]):
            all_appended_category -= 1
            current_cmp_label[min_category] = np.inf
        else:
            current_cmp_label[min_category] = list_of_list[min_category][current_idx_in_category[min_category]]
    return kept_idx


def lists_union(list_of_list, sorted=True, allow_repeat=False):
    """

    :param list_of_list:
    :param sorted: if True, all lists in list_of_list should be sorted lists, small->big
    :param allow_repeat: reserve same elements in merged list
    :return:
    """
    # get rid of empty lists
    list_of_list = [list_of_list[i] for i in range(len(list_of_list)) if len(list_of_list[i]) != 0]
    list_num = len(list_of_list)
    if list_num == 0:
        return []
    if sorted:
        union_list = []
        union_source_index_list = []
        list_len_list = [len(list_of_list[i]) for i in range(list_num)]
        # one step multi list merge sort
        pos_list = [0 for _ in range(list_num)]
        heap = [list_of_list[i][0] for i in range(list_num)]
        heap_source_index = list(range(list_num))
        # create heap
        for i in range(list_num // 2 + 1):
            adjust_min_heap(heap, [heap_source_index], i)
        last_item = None
        while True:
            # fetch the min
            src_i = heap_source_index[0]
            if allow_repeat or heap[0] != last_item:
                union_list.append(heap[0])
                union_source_index_list.append(src_i)
                last_item = heap[0]
            if pos_list[src_i] >= list_len_list[src_i] - 1:
                sort_complete = True
                # find a list where there are remaining items
                i = 0
                while i < list_num:
                    if pos_list[i] < list_len_list[i] - 1:
                        src_i = i
                        sort_complete = False
                        break
                    i += 1
                if sort_complete:
                    # fetch the last few items in heap
                    while len(heap) > 1:
                        heap[0] = heap.pop()
                        heap_source_index[0] = heap_source_index.pop()
                        adjust_min_heap(heap, [heap_source_index], 0)
                        if allow_repeat or heap[0] != last_item:
                            union_list.append(heap[0])
                            union_source_index_list.append(heap_source_index[0])
                            last_item = heap[0]
                    return union_list
            pos_list[src_i] += 1
            heap[0] = list_of_list[src_i][pos_list[src_i]]
            heap_source_index[0] = src_i
            adjust_min_heap(heap, [heap_source_index], 0)
    else:
        if not allow_repeat:
            # use builtin set!
            combined_set = set(list_of_list[0])
            for i in range(1, len(list_of_list)):
                combined_set.update(list_of_list[i])
            return list(combined_set)
        else:
            return combine_lists(list_of_list)


def lists_intersection(list_of_list, sorted=True, allow_repeat=False):
    list_num = len(list_of_list)
    for list in list_of_list:
        if len(list) == 0:
            return []
    if sorted:
        intersection_list = []
        list_len_list = [len(list_of_list[i]) for i in range(list_num)]
        pos_list = [0 for _ in range(list_num)]
        last_item = None
        while True:
            current_item = list_of_list[0][pos_list[0]]
            if not allow_repeat and last_item == current_item:
                pos_list[0] += 1
                last_item = current_item
                continue
            in_all_lists = True
            for i in range(1, list_num):
                exam_list = list_of_list[i]
                while exam_list[pos_list[i]] < current_item:
                    pos_list[i] += 1
                    if pos_list[i] >= list_len_list[i]:
                        return intersection_list
                if exam_list[pos_list[i]] != current_item:
                    in_all_lists = False
                    break
            if in_all_lists:
                intersection_list.append(current_item)
            pos_list[0] += 1
            if pos_list[0] >= len(list_of_list[0]):
                return intersection_list
            last_item = current_item
        return intersection_list
    else:
        if not allow_repeat:
            # use builtin set!
            combined_set = set(list_of_list[0])
            for i in range(1, len(list_of_list)):
                combined_set.intersection(set(list_of_list[i]))
            return list(combined_set)
        else:
            # use builtin dict!
            # dict value: records repeat times
            intersection_dict = {}
            new_intersection_dict = {}
            for item in list_of_list[0]:
                if item not in intersection_dict:
                    # [last occurrence, current occurrence]
                    intersection_dict[item] = [1, 0]
                else:
                    intersection_dict[item][0] += 1
            for i in range(1, len(list_of_list)):
                exam_list = list_of_list[i]
                for item in exam_list:
                    if item in intersection_dict:
                        if item not in new_intersection_dict:
                            new_intersection_dict[item] = intersection_dict[item]
                            new_intersection_dict[item][1] += 1
                        else:
                            new_intersection_dict[item][1] += 1
                # keep the least occurrence
                for k, v in new_intersection_dict.items():
                    if v[0] > v[1]:
                        v[0] = v[1]
                    v[1] = 0
                intersection_dict.clear()
                intersection_dict, new_intersection_dict = new_intersection_dict, intersection_dict
            return list(intersection_dict.keys())


def get_list_diff(list1, list2):
    """
    Union(list1, list2) - Intersection(list1, list2)

    :param list1:
    :param list2:
    :return:
    """
    i1n2 = []
    i2n1 = []
    for item in list1:
        if item not in list2:
            i1n2.append(item)
    for item in list2:
        if item not in list1:
            i2n1.append(item)
    return i1n2, i2n1


def get_pos0_item_in_csv(csv_path, pos=0):
    pos_item_list = []
    with open(csv_path, 'r', newline='') as f:
        rdr = csv.reader(f)
        for row in rdr:
            pos_item_list.append(row[pos])
    return pos_item_list


def list_to_dict(list_of_list, list_pos_as_key=0, key_in_item=False):
    """
    Change a list into dict with list at position of list_pos_reference as the key list.
    For each item in key list, items from other lists form a tuple as the value of that key.

    :param list_of_list:
    :param list_pos_as_key:
    :param key_in_item:
    :return:
    """
    assert 0 <= list_pos_as_key < len(list_of_list)
    list_len = len(list_of_list[0])
    for lst in list_of_list:
        assert len(lst) == list_len
    dict_out = dict()
    if key_in_item:
        key_list = list_of_list[list_pos_as_key]
    else:
        key_list = list_of_list.pop(list_pos_as_key)
    zipped = tuple(zip(*list_of_list))
    for i, key in enumerate(key_list):
        dict_out[key] = zipped[i]
    return dict_out


def recursive_rmdir(target_dir):
    """
    Delete directory and everything inside.

    :param target_dir:
    :return:
    """
    file_list = os.listdir(target_dir)
    for file in file_list:
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            recursive_rmdir(file_path)
    os.rmdir(target_dir)


class im2file:
    """
    Provides imshow() and show() just like cv2.imshow() and plt.show(),
    but saves output to a specific dir besides showing them.
    """
    MODULES = ['cv2', 'plt']

    def __init__(self, target_temp_dir=r'C:\Users\admin\Desktop\graph\work_report\temp',
                 save_duplicates=True, delete_when_exit=True, wait_key_before_exit=True,
                 try_use_cv2_waitKey=True):
        """

        :param target_temp_dir:
        :param save_duplicates: if True, then when multiple images are shown on same win_name
        :param delete_when_exit:
        :param wait_key_before_exit:
        :param try_use_cv2_waitKey: if True, call cv2.waitKey() instead of stdin input() before exit
        if imshow() has been called(there is at least one cv2 window) within the context
        """
        self.target_temp_dir = target_temp_dir
        self.module_temp_dir_dict = {module: os.path.join(self.target_temp_dir, module) for module in im2file.MODULES}
        for temp_dir in self.module_temp_dir_dict.values():
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        # cache those filenames which have already appeared and save their max index_suffix
        self.module_duplicate_filename_dict = {module: dict() for module in im2file.MODULES}
        self.save_duplicates = save_duplicates
        self.delete_when_exit = delete_when_exit
        self.wait_key_before_exit = wait_key_before_exit
        self.try_use_cv2_waitKey = try_use_cv2_waitKey
        self.imshow_called = False

    def __enter__(self):
        self.imshow_called = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Delete anything within temp dir which saves output image files.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        if self.wait_key_before_exit:
            while input('input \'q\' to exit and clear img dir.') != 'q':
                pass
        recursive_rmdir(self.target_temp_dir)

    def plot(self, *args: Any,
             scalex: bool = True,
             scaley: bool = True,
             data: Any = None,
             **kwargs: Any):
        filename = plt.get('title') + '.jpg'
        if self.save_duplicates:
            filename = self.find_available_path(filename, 'plt')
        plt.plot(*args, scalex, scaley, data, **kwargs)

    def find_available_path(self, filename, module='cv2'):
        list_split = filename.split('.')
        filename_no_suffix, suffix = '.'.join(list_split[:-1]), list_split[:-1]
        if filename in self.module_duplicate_filename_dict[module]:
            index_suffix = self.module_duplicate_filename_dict[module][filename] + 1
            target_temp_dir = self.module_temp_dir_dict[module]
            if os.path.exists(os.path.join(target_temp_dir, filename)):
                new_filename = filename_no_suffix + '_' + str(index_suffix) + '.' + suffix
                while os.path.exists(os.path.join(target_temp_dir, new_filename)):
                    index_suffix += 1
                    new_filename = filename_no_suffix + '_' + str(index_suffix) + '.' + suffix
            else:
                new_file_name = filename
            self.module_duplicate_filename_dict[module][filename] = index_suffix
            return new_file_name
        else:
            self.module_duplicate_filename_dict[module][filename] = -1
            return filename


def cal_angle(data, angle_range=0):
    """
    angle between (x, y) and (1, 0)

    :param data: first dim is x, y
    :param angle_range: 0:-pi~pi 1:0~2pi
    :return:
    """
    angle_data = np.arctan(data[1] / data[0])
    is_x_neg = data[0] < 0
    is_angle_neg = angle_data < 0
    if angle_range == 0:
        # 2nd quadrant
        pos_x_neg_angle_neg = np.where(np.logical_and(is_x_neg, is_angle_neg))
        angle_data[pos_x_neg_angle_neg] += np.pi
        # 3rd quadrant
        pos_x_neg_angle_zero_pos = np.where(np.logical_and(is_x_neg, np.logical_not(is_angle_neg)))
        angle_data[pos_x_neg_angle_zero_pos] -= np.pi
    else:
        # 2nd quadrant
        pos_x_neg_angle_neg = np.where(np.logical_and(is_x_neg, is_angle_neg))
        angle_data[pos_x_neg_angle_neg] += np.pi
        # 3rd quadrant
        pos_x_neg_angle_zero_pos = np.where(np.logical_and(is_x_neg, np.logical_not(is_angle_neg)))
        angle_data[pos_x_neg_angle_zero_pos] += np.pi
        # 4th quadrant
        pos_x_neg_angle_neg = np.where(np.logical_and(np.logical_not(is_x_neg), is_angle_neg))
        angle_data[pos_x_neg_angle_neg] += 2 * np.pi
    return angle_data


def absolute_angle_diff_numerical(angle_1, angle_2):
    return min(abs(angle_1 - angle_2), abs(angle_1 + 2 * np.pi - angle_2), abs(angle_1 - 2 * np.pi - angle_2))


def to_absolute_angle_diff(angle_diff):
    """
    Restrict angle difference within [0, pi)

    :param angle_diff: should take value within (-2pi, 2pi)
    :return:
    """
    return np.min(np.abs(np.stack([angle_diff, angle_diff + 2 * np.pi, angle_diff - 2 * np.pi])), axis=0)


def is_angle_closer_to_pi(data, same_dist_true=True):
    """

    :param data: should take value within [0, pi)
    :param same_dist_true:
    :return:
    """
    if same_dist_true:
        return (np.pi - data) <= data
    else:
        return (np.pi - data) < data


def euclidean_dist(data, ref=None):
    """

    :param data: first dim is x, y, ...
    :param ref: if None, calculate distance to (0, 0)
    :return: first dim is squeezed
    """
    if ref is None:
        data = data - ref
    return np.sqrt(np.sum([np.square(data[i]) for i in range(data.shape[0])], axis=0))


def moving_window_mean(data, win_len, prefer_left=False, pad_with=0):
    """
    Calculate mean in moving window

    :param pad_with: 0:local mean, 1:global mean
    :param prefer_left:
    :param data: Windows are created in the first dim
    :param win_len:
    :return: output length is len(data)
    """
    win_mean_list = []
    global_mean = np.mean(data, axis=0)
    win_center_pos = 0
    if win_len % 2 == 0:
        if prefer_left:
            len_before_center = win_len // 2
            len_after_center = len_before_center - 1
        else:
            len_after_center = win_len // 2
            len_before_center = len_after_center - 1
    else:
        len_before_center = win_len // 2
        len_after_center = len_before_center
    while win_center_pos < len_before_center:
        if pad_with == 0:
            win_mean_list.append(np.mean(data[:win_center_pos + len_after_center + 1], axis=0))
        else:
            win_mean_list.append(((len_before_center - win_center_pos) * global_mean +
                                  np.sum(data[:win_center_pos + len_after_center + 1], axis=0))
                                 / win_len)
        win_center_pos += 1
    while win_center_pos < len(data) - len_after_center:
        win_mean_list.append(
            np.mean(data[win_center_pos - len_before_center:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data):
        if pad_with == 0:
            win_mean_list.append(np.mean(data[win_center_pos - len_before_center:], axis=0))
        else:
            win_mean_list.append((np.sum(data[win_center_pos - len_before_center:], axis=0) +
                                  (len_after_center + 1 + win_center_pos - len(data)) * global_mean)
                                 / win_len)
        win_center_pos += 1
    return np.stack(win_mean_list, axis=0)


def moving_window_max(data, win_len, prefer_left=False):
    """
    Calculate max in moving window

    :param prefer_left:
    :param data: Windows are created in the first dim
    :param win_len:
    :return: output length is len(data)
    """
    win_max_list = []
    win_center_pos = 0
    if win_len % 2 == 0:
        if prefer_left:
            len_before_center = win_len // 2
            len_after_center = len_before_center - 1
        else:
            len_after_center = win_len // 2
            len_before_center = len_after_center - 1
    else:
        len_before_center = win_len // 2
        len_after_center = len_before_center
    while win_center_pos < len_before_center:
        win_max_list.append(np.max(data[:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data) - len_after_center:
        win_max_list.append(
            np.max(data[win_center_pos - len_before_center:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data):
        win_max_list.append(np.max(data[win_center_pos - len_before_center:], axis=0))
        win_center_pos += 1
    return np.stack(win_max_list, axis=0)


def moving_window_min(data, win_len, prefer_left=False):
    """
    Calculate min in moving window

    :param prefer_left:
    :param data: Windows are created in the first dim
    :param win_len:
    :return: output length is len(data)
    """
    win_min_list = []
    win_center_pos = 0
    if win_len % 2 == 0:
        if prefer_left:
            len_before_center = win_len // 2
            len_after_center = len_before_center - 1
        else:
            len_after_center = win_len // 2
            len_before_center = len_after_center - 1
    else:
        len_before_center = win_len // 2
        len_after_center = len_before_center
    while win_center_pos < len_before_center:
        win_min_list.append(np.min(data[:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data) - len_after_center:
        win_min_list.append(
            np.min(data[win_center_pos - len_before_center:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data):
        win_min_list.append(np.min(data[win_center_pos - len_before_center:], axis=0))
        win_center_pos += 1
    return np.stack(win_min_list, axis=0)



def adjust_min_heap(key, value_list, i):
    """
    Same exchange process in heap adjust also performed on every list in value_list.
    Therefore there will be correspondence.

    :param key:
    :param value_list: list of lists
    :param i:
    :return:
    """
    cur_pos = i
    while True:
        l_child_pos = 2 * cur_pos + 1
        r_child_pos = l_child_pos + 1
        if r_child_pos < len(key):
            # sub tree on the right may be shorter
            xchg_child_pos = l_child_pos if key[l_child_pos] < key[r_child_pos] else r_child_pos
        elif l_child_pos < len(key):
            xchg_child_pos = l_child_pos
        else:
            break
        if key[cur_pos] > key[xchg_child_pos]:
            key[cur_pos], key[xchg_child_pos] = key[xchg_child_pos], key[cur_pos]
            for value in value_list:
                value[cur_pos], value[xchg_child_pos] = value[xchg_child_pos], value[cur_pos]
            cur_pos = xchg_child_pos
        else:
            break


def moving_window_std(data, win_len, prefer_left=False):
    """
    Calculate std in moving window

    :param prefer_left:
    :param data: Windows are created in the first dim
    :param win_len:
    :return: output length is len(data)
    """
    win_std_list = []
    win_center_pos = 0
    if win_len % 2 == 0:
        if prefer_left:
            len_before_center = win_len // 2
            len_after_center = len_before_center - 1
        else:
            len_after_center = win_len // 2
            len_before_center = len_after_center - 1
    else:
        len_before_center = win_len // 2
        len_after_center = len_before_center
    while win_center_pos < len_before_center:
        win_std_list.append(np.std(data[:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data) - len_after_center:
        win_std_list.append(
            np.std(data[win_center_pos - len_before_center:win_center_pos + len_after_center + 1], axis=0))
        win_center_pos += 1
    while win_center_pos < len(data):
        win_std_list.append(np.std(data[win_center_pos - len_before_center:], axis=0))
        win_center_pos += 1
    return np.stack(win_std_list, axis=0)


def delete_all_nan_from_list(list_in, in_place=False):
    """
    Items should be numbers. Delete those are np.nan.

    :param in_place:
    :param list_in: first dim is to be deleted from
    :return:
    """
    if in_place:
        i = 0
        while i < len(list_in):
            if np.isnan(list_in[i]):
                list_in.pop(i)
            else:
                i += 1
        return list_in
    else:
        list_out = []
        for item in list_in:
            if not np.isnan(item):
                list_out.append(item)
        return list_out


def delete_all_nan(data):
    """

    :param data: first dim is to be deleted from
    :return:
    """
    data_parts = []
    deleted_frames = []
    start_pos = 0
    nan_start_pos = 0
    first_not_nan = True
    first_nan = True
    for i in range(len(data)):
        if not np.isnan(data[i]).any():
            if first_not_nan:
                start_pos = i
                for j in range(nan_start_pos, i):
                    deleted_frames.append(j)
                first_not_nan = False
                first_nan = True
        else:
            if first_nan:
                nan_start_pos = i
                if start_pos < i:
                    data_parts.append(data[start_pos:i])
                first_not_nan = True
                first_nan = False
    if len(data_parts) == 0:
        return data.copy(), deleted_frames
    else:
        return np.concatenate(data_parts, axis=0), deleted_frames


def index_map_to_before_delete(index_list, deleted_index_list):
    """
    Suppose a few items is deleted from a continuously indexed list,
    and index_list is given referring to the list after deletion.
    Map indexes in index_list back to indexes in the list before deletion.

    :param index_list:
    :param deleted_index_list:
    :return: index_list_before_del
    """
    if len(index_list) == 0:
        return []
    index_list_before_del = []
    index_list_pos = 0
    before_del_pos = index_list[index_list_pos]
    deleted_index_list_pos = 0
    while index_list_pos < len(index_list):
        if deleted_index_list_pos >= len(deleted_index_list):
            break
        while before_del_pos < deleted_index_list[deleted_index_list_pos]:
            index_list_before_del.append(before_del_pos)
            index_list_pos += 1
            if index_list_pos >= len(index_list):
                return index_list_before_del
            before_del_pos = index_list[index_list_pos] + deleted_index_list_pos
        deleted_index_list_pos += 1
        before_del_pos += 1
    for i in range(index_list_pos, len(index_list)):
        before_del_pos = index_list[i] + deleted_index_list_pos
        index_list_before_del.append(before_del_pos)
    return index_list_before_del


def os_path_split_until(path, target_str, target_str_in_parts=True):
    """
    Take off one part separated by sep from the tail each time, until target_str is encountered.

    :param path:
    :param target_str:
    :param target_str_in_parts: Upon return, target_str either in path(True) or in split_parts(False).
    :return: path, split_parts
    """
    split_parts = []
    while True:
        path, filename = os.path.split(path)
        if filename == '':
            break
        if filename == target_str:
            if target_str_in_parts:
                split_parts.append(filename)
            else:
                path = os.path.join(path, filename)
            split_parts.reverse()
            break
        split_parts.append(filename)
    return path, split_parts


def basic_type_np_equal(x, y):
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        return np.array_equal(x, y)
    else:
        return x == y


def basic_type_np_smaller_equal(x, y):
    """
    Use Frobenius norm for ndarray comparison.

    :param x:
    :param y:
    :return: x <= y
    """
    if isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        return np.linalg.norm(x) <= np.linalg.norm(y)
    else:
        return x <= y


def is_sub_list(list1, list2, f=None):
    """
    Check if list2 is a sublist of list1, which means all items in list2 exist in list1
    in exactly the same order, although some items in list1 may not exist in list2,
    and positions of these missing items may be arbitrary.

    Examples
    --------
    list2 is a sublist of list1:

    list1 = [5, 1, 2, 6, 3, 4]

    list2 = [1, 2, 3, 4]

    list2 is NOT a sublist of list1:

    list1 = [1, 3, 2, 4, 5]

    list2 = [1, 2, 3, 4]

    :param list1:
    :param list2:
    :param f: should take two params and return True if they are considered the same.
    :return:
    """
    if len(list2) > len(list1):
        return False
    if f is None:
        f = basic_type_np_equal
    i = 0
    j = 0
    while True:
        while not f(list1[i], list2[j]):
            i += 1
            if i == len(list1):
                return False
        i += 1
        j += 1
        if j == len(list2):
            return True
        if i == len(list1):
            return False


def is_list_same(list1, list2, f=None):
    if f is None:
        f = basic_type_np_equal
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        # print(list1[i])
        # print(list2[i])
        # print(i)
        if not f(list1[i], list2[i]):
            print(list1[i])
            print(list2[i])
            return False
    return True


def fetch_list_of_list_items(list_of_list, fetch_idx_list):
    list_num = len(list_of_list)
    # print(list_of_list)
    list_of_list_out = [[] for _ in range(list_num)]
    for fetch_idx in fetch_idx_list:
        for list_idx in range(list_num):
            list_of_list_out[list_idx].append(list_of_list[list_idx][fetch_idx])
    return list_of_list_out


def fetch_list_items_with_step(list_in, step=2, start_pos=0):
    list_out, list_remainder = [], []
    remainder_start = 0
    for i in range(start_pos, len(list_in), step):
        list_out.append(list_in[i])
        list_remainder.extend(list_in[remainder_start: i])
        remainder_start = i + 1
    list_remainder.extend(list_in[remainder_start: len(list_in)])
    return list_out, list_remainder


def fetch_list_slices_with_step(list_in, slice_len=1, step=2, start_pos=0, drop_last=False):
    """

    :param list_in:
    :param slice_len:
    :param step:
    :param start_pos:
    :param drop_last: if True, put the group of last few items which has length smaller than slice_len to list_remainder
    :return:
    """
    list_out, list_remainder = [], []
    remainder_start = 0
    for i in range(start_pos, len(list_in), step):
        list_remainder.extend(list_in[remainder_start: i])
        remainder_start = i + slice_len
        if i + slice_len <= len(list_in) or not drop_last:
            list_out.extend(list_in[i: i + slice_len])
        else:
            list_remainder.extend(list_in[i: i + slice_len])
    list_remainder.extend(list_in[remainder_start: len(list_in)])
    return list_out, list_remainder


def join_as_str_with(list_in, sep='_'):
    return sep.join([str(item) for item in list_in])


def pearson_corr(x, y):
    # return np.cov(x, y)[0, 1] / np.std(x) / np.std(y)
    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) == len(y)
    centered_x, centered_y = x - np.mean(x), y - np.mean(y)
    return np.sum(centered_x * centered_y) / np.sqrt(np.sum(centered_x * centered_x)) / np.sqrt(np.sum(centered_y * centered_y))
    # np.cov()


def remove_list_of_list_items(list_of_list, index_list, in_place=False):
    list_len = len(list_of_list[0])
    if not in_place:
        keep_index_list = []
        j = 0
        for i in range(list_len):
            if i == index_list[j]:
                j += 1
                if j >= len(index_list):
                    for t in range(i + 1, list_len):
                        keep_index_list.append(t)
                    break
            else:
                keep_index_list.append(i)
        return fetch_list_of_list_items(list_of_list, keep_index_list)
    else:
        # in_place
        target_list_len = list_len - len(index_list)
        remap_index = []
        j = 0
        i = 0
        while i < list_len:
            if i == index_list[j]:
                j += 1
                i += 1
                if j >= len(index_list):
                    for t in range(i, list_len):
                        remap_index.append(t)
                    break
            remap_index.append(i)
            i += 1
        for i in range(target_list_len):
            for lst in list_of_list:
                lst[i] = lst[remap_index[i]]
        for i in range(len(index_list)):
            for lst in list_of_list:
                lst.pop()
        return list_of_list

def count(list_in, min_value=None, max_value=None, bin_num=None, bin_len=None, value_on_interval=True, display=True):
    if min_value is None:
        min_value = np.min(list_in)
    if max_value is None:
        max_value = np.max(list_in)
    if value_on_interval:
        if bin_num is None:
            assert bin_len is not None
            bin_num = int((max_value - min_value) / bin_len)
        counts, bins, _ = plt.hist(list_in,
                                   bins=np.linspace(min_value, max_value, bin_num + 1, True))
    else:
        if bin_len is None:
            assert bin_num is not None
            bin_len = (max_value - min_value) / (bin_num - 1)
        if bin_num is None:
            assert bin_len is not None
            bin_num = int((max_value - min_value) / bin_len + 1)
        counts, bins, _ = plt.hist(list_in,
                                   bins=np.linspace(min_value - bin_len / 2,
                                                    max_value + bin_len / 2,
                                                    bin_num + 1, True))
    if display:
        plt.show()
        plt.clf()
    return counts, bins


if __name__ == '__main__':
    # print(binary_search(list(range(15)), 1))
    # print(random_ints(0, 15, 6))
    # a = [[12, 45, 78], [34, 35, 36], [46, 97, 100]]
    # print(lists_union_sorted1(a))
    # print(lists_union(a))
    # a = [1, 2, 3]
    # b = [1.5, 2, 2.5]
    x = list(range(15))
    y = list(range(15, 30))
    index_list = [1, 5, 8, 13]
    print(remove_list_of_list_items([x, y], index_list, True))
    print([x, y])
    # print(np.array(zip(a, b)))
    # print(pearson_corr(a, b))
    # print(np.corrcoef(a, b))
