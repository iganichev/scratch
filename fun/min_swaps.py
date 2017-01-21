#!/usr/bin/env python

# http://qa.geeksforgeeks.org/1666/google-interview-question-minimum-no-of-swaps

def next_pair(p):
    for i, v in enumerate(p):
        if not v:
            yield i

def min_swaps(lst, pairs):
    """ lst - list of elements
        pairs - list of pairs of these elements"""

    elem2idx = {e: idx for (idx, e) in enumerate(lst)}
    elem2elem = {e1 : e2 for (ep1, ep2) in pairs for (e1, e2) in [(ep1, ep2),
                                                                  (ep2, ep1)]}
    print "elem2elem:", elem2elem
    done_pairs = [False for x in xrange(len(lst)/2)] 
    result = 0
    for pair_idx in next_pair(done_pairs):
        print "starting pair_idx:", pair_idx
        counter = 0
        while not done_pairs[pair_idx]:
            done_pairs[pair_idx] = True
            counter += 1

            e1 = lst[2 * pair_idx]
            e2 = elem2elem[e1]
            pair_idx = elem2idx[e2] / 2
        result += counter - 1
    return result




print min_swaps([1, 3, 5, 2, 4, 6], [(1,2), (3,4), (5,6)])
print min_swaps([1, 3, 5, 2, 10, 20, 21, 11, 4, 6],
                [(1,2), (3,4), (5,6), (10, 11), (20, 21)])
