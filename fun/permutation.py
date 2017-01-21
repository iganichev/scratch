#!/usr/bin/env python

def perm(lst):
    """ Returns a list of all permutations of lst."""
    if len(lst) == 1:
        return [lst]
    result = []
    for e in lst:
        without_e = [x for x in lst if x != e]
        sub_answer = perm(without_e)
        result.extend([[e] + sub for sub in sub_answer])
    return result

print "\n".join([str(lst) for lst in perm([1, 2])])

print "\n".join([str(lst) for lst in perm([3, 1, 2])])

