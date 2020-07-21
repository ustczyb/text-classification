from typing import List
#from collections import defaultdict
class Solution:
    def maxNumOfSubstrings(self, s: str) -> List[str]:
        # 1.转换数组
        index_dict = {}
        for i in range(len(s)):
            index_dict[s[i]] = i
        arr = [0] * len(s)
        for i in range(len(s)):
            arr[i] = index_dict[s[i]]
        print(arr)
        # 2.求极值
        i = 0
        min_end = len(s)
        min_start = 0
        res = []
        taotai_set = set()
        while i < len(s):
            if i == min_end and arr[i] not in taotai_set:
                print(i)
                print(arr[i])
                print(taotai_set)
                if i < min_end:
                    taotai_set.add(min_end)
                    min_end = i
                    min_start = i
                res.append(s[min_start: min_end + 1])
                min_end = len(s)
                min_start = 0
                i += 1
                continue
            if arr[i] < min_end:
                taotai_set.add(min_end)
                if arr[i] not in taotai_set:
                    min_start = i
                    min_end = arr[i]
            elif arr[i] == min_end:
                i += 1
                continue
            else:
                taotai_set.add(min_end)
                min_end = arr[i]
                taotai_set.add(arr[i])
            i += 1
        if not res:
            res.append(s)
        return res

if __name__ == '__main__':
    print(Solution().maxNumOfSubstrings("abaabbcaaabbbccd"))
