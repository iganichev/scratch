#!/usr/bin/env python

def reduce_spaces(line):
    res = []
    seen_space = False
    for c in line:
        if c != ' ':
            res.append(c)
        else:
            if not seen_space:
                seen_space = True
                res.append(c)
    return ''.join(res)

print reduce_spaces("Hello    World!    ")
assert "Hello World!" == reduce_spaces("Hello    World!    ")

