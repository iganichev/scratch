#!/usr/bin/env python

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        carry = 0
        c1 = l1
        c2 = l2
        r = None
        head = None
        while True:
            if c1 is None and c2 is None and carry == 0:
                return head
            elif c1 is None and c2 is None:
                r.next = ListNode(1)
                return head

            v1 = c1.val if c1 else 0
            c1 = c1.next if c1 else None
            v2 = c2.val if c2 else 0
            c2 = c2.next if c2 else None

            n = ListNode((v1 + v2 + carry) % 10)
            carry = (carry + v1 + v2) / 10
            head = head or n
            if r is not None:
                r.next = n
            r = n
