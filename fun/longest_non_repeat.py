#!/usr/bin/env python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        r = 0  # result
        class nl:
            i = 0  # first idx in substring
            j = 1  # 1+last idx in substring
            sub = set(s[0])  # set of chars in substring
        def expand():
            while nl.j < len(s) and s[nl.j] not in nl.sub:
                nl.sub.add(s[nl.j])
                nl.j += 1
            return nl.j - nl.i

        def shrink():
            dup = s[nl.j]
            idx = s[nl.i:nl.j].index(dup) + nl.i
            nl.sub.difference_update(s[nl.i:idx+1])
            nl.i = idx + 1

        while True:
            r = max(r, expand())
            if nl.j == len(s):
                break
            shrink()

        return r


s = Solution()
print s.lengthOfLongestSubstring("abcabcbb")
print s.lengthOfLongestSubstring("bbbbb")
print s.lengthOfLongestSubstring("pwwkew")

        
