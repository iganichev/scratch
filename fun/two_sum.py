#!/usr/bin/env python

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        a = {}
        for i, n in enumerate(nums):
            if (target - n) in a:
                return [a[target - n], i]
            a[n] = i

s = Solution()
print s.twoSum([2, 3, 8], 11) 
