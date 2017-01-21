#!/usr/bin/env python

class Stack(object):
    def __init__(self):
        self.stack = []

    def push(self, val):
        self.stack.append(val)
        return self

    def pop(self):
        return self.stack.pop()

    @property
    def top(self):
        return self.stack[-1]

    def __str__(self):
        return str(self.stack)

    def __nonzero__(self):
        return len(self.stack) != 0


class Node(object):
    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value

    def insert(self, val):
        if val > self.value:
            if self.right:
                self.right.insert(val)
            else:
                self.right = Node(value=val)
        elif val < self.value:
            if self.left:
                self.left.insert(val)
            else:
                self.left = Node(value=val)

    def traverse(self):
        def t(node):
            if not node:
                return []
            return t(node.left) + [node.value] + t(node.right)
        return t(self)

    def traverse_iter(self):
        ENTER = 1
        LEFT_DONE = 2
        SELF_DONE = 3

        # a stack of nodes to traverse
        stack = Stack().push((ENTER, self))
        vals = []
        while stack:
            print stack
            place, node = stack.pop()
            if place == ENTER:
                stack.push((LEFT_DONE, node))
                if node.left:
                    stack.push((ENTER, node.left))
            elif place == LEFT_DONE:
                stack.push((SELF_DONE, node))
                vals.append(node.value)
            elif place == SELF_DONE:
                if node.right:
                    stack.push((ENTER, node.right))

        return vals

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


b = Node(value=5)
b.insert(2)
b.insert(4)
b.insert(40)
b.insert(10)
b.insert(20)
b.insert(60)
print b.traverse_iter()
