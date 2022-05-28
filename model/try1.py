
# array = [1,2,3,4,5]
# def permutations(array,start,end):
#     if start==end:
#         print(array)
#         return
#     for i in range(start,end):
#         array[start],array[i] = array[i],array[start]
#         permutations(array,start+1,end)
#         array[start],array[i] = array[i],array[start]
#
# permutations(array,0,5)
#

#递归计算阶乘
# def alg(n):
#     if n<=0:
#         return 1
#
#     return n*alg(n-1)
# print(alg(4))

# for alist in li:
#     for _,g in groupby(alist):
#         print(g)
# for i in range(5,0,-1):
#     print(i)

# arr = [1,2,3,4,5]
# arr[1:]=0
# print(arr)

# for i in range(1):
#     print(i)
# from functools import reduce
# a=[[["abc","123"],["def","456"],["ghi","789"]]]
# b=reduce(lambda x,y:x+y , a )
# print(b)

# for j in reversed(range(10)):
#     print(j)
# import pandas as pd
#
# d1 = pd.DataFrame(df[df.columns[categorical_features[0]]].describe())
# det = pd.DataFrame()
# for i in range(11):
#     det = pd.concat([])

#Bubble Sort冒泡排序
# class Solution:
#     def bubbleSort(self, arr):
#         #排序趟数(最差n-1趟)
#         for i in range(len(arr)):
#             print('i:',i)
#             #可变动位置
#             for j in range(len(arr)-i-1):
#                 print('j:',j)
#                 if arr[j]>arr[j+1]:
#                     arr[j],arr[j+1] = arr[j+1],arr[j]
#         return arr
#
#     def sortArray(self, nums):
#         return self.bubbleSort(nums)
#
# solution = Solution()
#
# print(solution.sortArray([5,4,3,2,1]))

# nums = [[0]*(n+1) for n in range(4)]
# print(nums)
# for i in range(1,2):
#     print(i)
# def jj(rowIndex):
#     nums = [[1] * (n + 1) for n in range(rowIndex+1)]
#     print(nums)
#     if rowIndex < 2:
#         return nums[rowIndex]
#     else:
#         # 遍历行
#         for i in range(2, rowIndex + 1):
#             for j in range(1, rowIndex):
#                 nums[i][j] = nums[i - 1][j - 1] + nums[i - 1][j]
#         return nums[rowIndex]
# rowIndex = 4
# print(jj(rowIndex))

#众所周知，第n行第m个数是C(n,m)，所以利用Python math库的组合数就生成了答案
# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         if not matrix or not matrix[0]:
#             return list()
#
#         rows, columns = len(matrix), len(matrix[0])
#         visited = [[False] * columns for _ in range(rows)]
#         total = rows * columns
#         order = [0] * total
#
#         directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
#         row, column = 0, 0
#         directionIndex = 0
#         for i in range(total):
#             order[i] = matrix[row][column]
#             visited[row][column] = True
#             nextRow, nextColumn = row + directions[directionIndex][0], column + directions[directionIndex][1]
#             if not (0 <= nextRow < rows and 0 <= nextColumn < columns and not visited[nextRow][nextColumn]):
#                 directionIndex = (directionIndex + 1) % 4
#             row += directions[directionIndex][0]
#             column += directions[directionIndex][1]
#         return order
# matrix = [[1,2,3],[4,5,6],[7,8,9]]
# res = []
# count = 0
# while matrix:
#     count+=1
#     print(count)
#     res += matrix.pop(0)
#     matrix = list(zip(*matrix))[::-1]
#
# print(res)

#选择排序
# def selectionsort(arr):
#     #对"趟"遍历
#     for i in range(len(arr)-1):
#         min_index = i
#         for j in range(i+1,len(arr)):
#             if arr[j]<arr[min_index]:
#                 min_index = j
#         if min_index!=i:
#             arr[i],arr[min_index] = arr[min_index],arr[i]
#     return arr
#
# arr = [1,3,4,2,7,6]
#
# print(selectionsort(arr))

#插入排序

# def insertionSort(arr):
#     #遍历每一"趟"
#     for i in range(1, len(arr)):
#         #将第i个元素暂存
#         temp = arr[i]
#         j = i
#         #如果前面有大于暂存元素的元素，前面的元素则后移一个单位，如果有多个，则依次后移
#         while j > 0 and arr[j - 1] > temp:
#             arr[j] = arr[j - 1]
#             j -= 1
#         #把前面所有大于暂存元素的元素移动完毕，则这个时候的j就是暂存元素所处的index
#         arr[j] = temp
#
#     return arr



#fork
# def insertionSort(arr):
#     for i in range(1,len(arr)):
#         temp = arr[i]
#         j = i
#         while j>0 and arr[j-1]>temp:
#             arr[j] = arr[j-1]
#             j-=1
#         arr[j] = temp
#     return arr
#
# arr = [2,4,3,0,1,5]
# print(insertionSort((arr)))

#希尔排序
# def shellSort(arr):
#     size = len(arr)
#     #间隔
#     gap = size // 2
#
#     while gap > 0:
#         #从每个子序列的第二个元素开始遍历，到最后一个元素(每个子序列走n-1趟，n是子序列的长度)，同时对所有子序列做选择排序
#         for i in range(gap, size):
#             temp = arr[i]
#             j = i
#
#             while j >= gap and arr[j - gap] > temp:
#                 arr[j] = arr[j - gap]
#                 j -= gap
#             arr[j] = temp
#         gap = gap // 2
#     return arr

#arr = [2,4,3,0,1,5]
#print(shellSort(arr))

#fork

# def shellSort(arr):
#     size = len(arr)
#     gap = size//2
#
#     while gap>0:
#         for i in range(gap,size):
#             temp = arr[i]
#             j = i
#             while j>=gap and arr[j-gap] > temp:
#                 arr[j] = arr[j-gap]
#                 j -= gap
#             arr[j] = temp
#         gap = gap//2
#     return arr
#
# arr = [2,4,3,0,1,5]
# print(shellSort(arr))

#归并排序（MergeSort）
# class Solution:
#     def merge(self,left_arr,right_arr):
#         arr = []
#         while left_arr and right_arr:
#             if left_arr[0]<=right_arr[0]:
#                 arr.append(left_arr.pop(0))
#             else:
#                 arr.append(right_arr.pop(0))
#         while left_arr:
#             arr.append(left_arr.pop(0))
#         while right_arr:
#             arr.append(right_arr.pop(0))
#         return arr
#
#     def mergeSort(self,arr):
#         size = len(arr)
#         if size<2:
#             return arr
#         mid = len(arr)//2
#         left_arr,right_arr = arr[0:mid],arr[mid:]
#         return self.merge(self.mergeSort(left_arr),self.mergeSort(right_arr))
#
#     def sortArray(self,nums):
#         return self.mergeSort(nums)



# arr = [2,4,3,0,1,5]
# sorted_function = Solution()
# print(sorted_function.sortArray(arr))




#fork
# class Solution:
#     def merge(self,left_arr,right_arr):
#         arr = []
#         while left_arr and right_arr:
#             if left_arr[0]<=right_arr[0]:
#                 arr.append(left_arr.pop(0))
#             else:
#                 arr.append(right_arr.pop(0))
#         while left_arr:
#             arr.append(left_arr.pop(0))
#         while right_arr:
#             arr.append(right_arr.pop(0))
#         return arr
#
#     def mergeSort(self,arr):
#         size = len(arr)
#         if size<2:
#             return arr
#         mid = len(arr)//2
#
#         left_arr,right_arr = arr[0:mid],arr[mid:]
#         return self.merge(self.mergeSort(left_arr),self.mergeSort(right_arr))
#
#     def sortArray(self,arr):
#         return self.mergeSort(arr)
#
#
#
# arr = [2,4,3,0,1,5]
# sorted_function = Solution()
# print(sorted_function.sortArray(arr))


#《剑指offer第51题》
# self.cnt = 0
#
#
# def merge(nums, start, mid, end, temp):
#     i, j = start, mid + 1
#     while i <= mid and j <= end:
#         if nums[i] <= nums[j]:
#             temp.append(nums[i])
#             i += 1
#         else:
#             self.cnt += mid - i + 1
#             temp.append(nums[j])
#             j += 1
#     while i <= mid:
#         temp.append(nums[i])
#         i += 1
#     while j <= end:
#         temp.append(nums[j])
#         j += 1
#
#     for i in range(len(temp)):
#         nums[start + i] = temp[i]
#     temp.clear()
#
#
# def mergeSort(nums, start, end, temp):
#     if start >= end: return
#     mid = (start + end) >> 1
#     mergeSort(nums, start, mid, temp)
#     mergeSort(nums, mid + 1, end, temp)
#     merge(nums, start, mid, end, temp)
#
#
# mergeSort(nums, 0, len(nums) - 1, [])
# return self.cnt


# class Solution:
#     def mergeSort(self, nums, tmp, l, r):
#         if l >= r:
#             return 0
#
#         mid = (l + r) // 2
#         inv_count = self.mergeSort(nums, tmp, l, mid) + self.mergeSort(nums, tmp, mid + 1, r)
#         i, j, pos = l, mid + 1, l
#         while i <= mid and j <= r:
#             if nums[i] <= nums[j]:
#                 tmp[pos] = nums[i]
#                 i += 1
#                 inv_count += (j - (mid + 1))
#             else:
#                 tmp[pos] = nums[j]
#                 j += 1
#             pos += 1
#         for k in range(i, mid + 1):
#             tmp[pos] = nums[k]
#             inv_count += (j - (mid + 1))
#             pos += 1
#         for k in range(j, r + 1):
#             tmp[pos] = nums[k]
#             pos += 1
#         nums[l:r+1] = tmp[l:r+1]
#         return inv_count
#
#     def reversePairs(self, nums: List[int]) -> int:
#         n = len(nums)
#         tmp = [0] * n
#         return self.mergeSort(nums, tmp, 0, n - 1)


# def countSmaller(nums):
#     counts = [0] * len(nums)
#     def merge(nums, start, mid, end, temp):
#         i, j = start, mid + 1
#         while i <= mid and j <= end:
#             if nums[i] > nums[j]:
#                 counts[i] += end - j + 1
#                 temp.append(nums[i])
#                 i += 1
#             elif nums[i] == nums[j]:
#                 counts[i] += end - j
#                 temp.append(nums[i])
#                 i += 1
#             else:
#                 temp.append(nums[j])
#                 j += 1
#         while i <= mid:
#             temp.append(nums[i])
#             i += 1
#         while j <= end:
#             temp.append(nums[j])
#             j += 1
#         for k in range(len(temp)):
#             nums[start + k] = temp[k]
#         temp.clear()
#
#     def mergeSort(nums, start, end, temp):
#         if start >= end: return
#         mid = (start + end) // 2
#
#         mergeSort(nums, start, mid, temp)
#         mergeSort(nums, mid + 1, end, temp)
#         merge(nums, start, mid, end, temp)
#
#     mergeSort(nums, 0, len(nums) - 1, [])
#     return counts
#
# nums = [0,2,1]
# print(countSmaller(nums))


import random
#
#
# class Solution:
#     #low:0,   high:len(nums)-1
#     def randomPartition(self, arr: [int], low: int, high: int):
#         #从[low,high]里面查找
#         i = random.randint(low, high)
#         #第i个值和最后一个互换
#         arr[i], arr[high] = arr[high], arr[i]
#         #
#         return self.partition(arr, low, high)
#
#     def partition(self, arr: [int], low: int, high: int):
#         #赋值为-1
#         i = low - 1
#         #把最后一个数设为基准
#         pivot = arr[high]
#         #遍历第一个到倒数第二个数
#         for j in range(low, high):
#             #如果比基准小
#             if arr[j] <= pivot:
#                 #将它们放到一起，把比基准大的统一放到右边
#                 i += 1
#                 arr[i], arr[j] = arr[j], arr[i]
#         #把基准放到i+1索引处,也就是第一个比基准大的地方
#         arr[i + 1], arr[high] = arr[high], arr[i + 1]
#         #返回索引
#         return i + 1
#
#     def quickSort(self, arr, low, high):
#         if low < high:
#             #找到pivot应该在的位置
#             pi = self.randomPartition(arr, low, high)
#             #为左子序列递归做快排
#             self.quickSort(arr, low, pi - 1)
#             #为右子序列递归做快排
#             self.quickSort(arr, pi + 1, high)
#
#         return arr
#
#     def sortArray(self, nums):
#         return self.quickSort(nums, 0, len(nums) - 1)
#
# solution = Solution()
# nums = [1]
# print(solution.sortArray(nums))

#fork
# import random
# class Solution:
#     def randomPartition(self,arr,low,high):
#         i = random.randint(low,high)
#         #互换
#         arr[i],arr[high] = arr[high],arr[i]
#         return self.partition(arr,low,high)
#
#     def partition(self,arr,low,high):
#         i = low-1
#         pivot = arr[high]
#         for j in range(low,high):
#             if arr[j]<=pivot:
#                 i+=1
#                 arr[i],arr[j] = arr[j],arr[i]
#         arr[i+1],arr[high] = arr[high],arr[i+1]
#         return i+1
#
#     def quickSort(self,arr,low,high):
#         #确定递归停止规则
#         if low<high:
#             pi = self.randomPartition(arr,low,high)
#             self.quickSort(arr,low,pi-1)
#             self.quickSort(arr,pi+1,high)
#         return arr
#
#
#     def sortArray(self,nums):
#         return self.quickSort(nums,0,len(nums)-1)
#
# solution = Solution()
# nums = [3,2,45,6,7,4]
# print(solution.sortArray(nums))


import math
#向上取整：math.ceil()

# class Solution:
#     def majorityElement(self, nums) -> int:
#         self.lot_num = 0
#
#         def randomPartition(arr: [int], low: int, high: int):
#             i = random.randint(low, high)
#             arr[i], arr[high] = arr[high], arr[i]
#             return partition(arr, low, high)
#
#         def partition(arr: [int], low: int, high: int):
#             i = low - 1
#             pivot = arr[high]
#
#             for j in range(low, high):
#                 if arr[j] == pivot:
#                     i += 1
#                     arr[i], arr[j] = arr[j], arr[i]
#             arr[i + 1], arr[high] = arr[high], arr[i + 1]
#
#             if (i + 1) - low + 1 > int(len(arr) / 2):
#                 self.lot_num = arr[i + 1]
#             return i + 1
#
#         def quickSort(arr, low, high):
#             if len(arr) == 1:
#                 self.lot_num = arr[0]
#             if low < high:
#                 pi = randomPartition(arr, low, high)
#                 quickSort(arr, low, pi - 1)
#                 quickSort(arr, pi + 1, high)
#             return arr
#
#         return quickSort(nums, 0, len(nums) - 1)
#
# arr = [1,3,4,1,14,4,3,11,1]
# solution = Solution()
# print(solution.majorityElement(arr))

# class Solution:
#     def insertionSort(self, arr):
#         for i in range(1, len(arr)):
#             temp = arr[i]
#             j = i
#             while j > 0 and arr[j - 1] > temp:
#                 arr[j] = arr[j - 1]
#                 j -= 1
#             arr[j] = temp
#
#         return arr
#
#     def bucketSort(self, arr, bucket_size=5):
#         arr_min, arr_max = min(arr), max(arr)
#         bucket_count = (arr_max - arr_min) // bucket_size + 1
#         buckets = [[] for _ in range(bucket_count)]
#
#         for num in arr:
#             buckets[(num - arr_min) // bucket_size].append(num)
#
#         res = []
#         for bucket in buckets:
#             self.insertionSort(bucket)
#             res.extend(bucket)
#
#         return res
#
#     def sortArray(self, nums):
#         return self.bucketSort(nums)
# arr = [1,3,4,1,14,4,3,11,1]
#
# print((max(arr)-min(arr))//5+1)

# solution = Solution()
# print(solution.sortArray(arr))


#桶排序
# def bucket_sort(array,batch_size=3):
#     min_num, max_num = min(array), max(array)
#     #3表示一个桶含有3个数
#     bucket_num = (max_num-min_num)//batch_size + 1
#
#     buckets = [[] for _ in range(int(bucket_num))]
#     for num in array:
#         buckets[int((num-min_num)//batch_size)].append(num)
#     #存放新的排序结果
#     new_array = list()
#     for i in buckets:
#         for j in sorted(i):
#             new_array.append(j)
#     return new_array
#
# arr = [1,3,4,1,14,4,3,11,1]
# print(bucket_sort(arr))

#fork
# class Solution:
#
#     def bucketSort(self,arr):
#         #插入排序API
#         def insertionSort(arr):
#             for i in range(1,len(arr)):
#                 temp = arr[i]
#                 j = i
#                 while j>0 and arr[j-1]>temp:
#                     arr[j] = arr[j-1]
#                     j-=1
#                 arr[j] = temp
#             return arr
#         #正式分桶排序
#         def bucket_Sort(arr,batch_size=5):
#             min_num = min(arr)
#             max_num = max(arr)
#             #确定分几个桶
#             bucket_count = (max_num-min_num)//batch_size+1
#             #给桶分配数组
#             buckets = [[] for _ in range(bucket_count)]
#             #把原数组中的元素分配到桶中
#             for num in arr:
#                 buckets[int(num-min_num)//batch_size].append(num)
#             new_arr = []
#             for i in buckets:
#                 for j in insertionSort(i):
#                     new_arr.append(j)
#             return new_arr
#         return bucket_Sort(arr)

# solution = Solution()
# arr = [1,3,4,1,14,4,3,11,1]
# print(solution.bucketSort(arr))

#堆排序
# class Solution:
#     # 调整为大顶堆
#     def heapify(self, arr: [int], index: int, end: int):
#         left = index * 2 + 1
#         right = left + 1
#         while left <= end:
#             # 当前节点为非叶子结点
#             max_index = index
#             if arr[left] > arr[max_index]:
#                 max_index = left
#             if right <= end and arr[right] > arr[max_index]:
#                 max_index = right
#             if index == max_index:
#                 # 如果不用交换，则说明已经交换结束
#                 break
#             arr[index], arr[max_index] = arr[max_index], arr[index]
#             # 继续调整子树
#             index = max_index
#             left = index * 2 + 1
#             right = left + 1
#
#     # 初始化大顶堆
#     def buildMaxHeap(self, arr: [int]):
#         size = len(arr)
#         # (size-2) // 2 是最后一个非叶节点，叶节点不用调整
#         for i in range((size - 2) // 2, -1, -1):
#             self.heapify(arr, i, size - 1)
#         return arr
#
#     # 升序堆排序，思路如下：
#     # 1. 先建立大顶堆
#     # 2. 让堆顶最大元素与最后一个交换，然后调整第一个元素到倒数第二个元素，这一步获取最大值
#     # 3. 再交换堆顶元素与倒数第二个元素，然后调整第一个元素到倒数第三个元素，这一步获取第二大值
#     # 4. 以此类推，直到最后一个元素交换之后完毕。
#     def maxHeapSort(self, arr: [int]):
#         self.buildMaxHeap(arr)
#         size = len(arr)
#         for i in range(size):
#             arr[0], arr[size - i - 1] = arr[size - i - 1], arr[0]
#             self.heapify(arr, 0, size - i - 2)
#         return arr
#
#     def sortArray(self, nums):
#         return self.maxHeapSort(nums)


# class Solution:
#     def countingSort(self,arr):
#         arr_min,arr_max = min(arr),max(arr)
#         size = arr_max-arr_min+1
#         #创建计数数组
#         counts = [0 for _ in range(size)]
#         #填充计数数组
#         for num in arr:
#             counts[num-arr_min] += 1
#         #累加计数数组
#         for j in range(1,size):
#             counts[j] += counts[j-1]
#         #创建结果数组
#         res = [0 for _ in range(len(arr))]
#         #填充结果数组
#         for i in range(len(arr)):
#             res[counts[arr[i]-arr_min]-1] = arr[i]
#             #为了解决元素多次出现的问题
#             counts[arr[i]-arr_min] -= 1
#         return res



#fork
# class Solution:
#     def countingSort(self,arr):
#         min_arr,max_arr = min(arr),max(arr)
#         size = len(arr)
#         counts = [0 for _ in range(max_arr-min_arr+1)]
#
#         for num in arr:
#             counts[num-min_arr] += 1
#         for j in range(1,len(counts)):
#             counts[j] += counts[j-1]
#
#         res = [0 for _ in range(size)]
#         for i in range(size):
#             res[counts[arr[i]-min_arr]-1] = arr[i]
#             counts[arr[i]-min_arr] -= 1
#         return res
#
# solution = Solution()
# arr = [4,3,2,1,0,5,6,8,3,2]
# print(solution.countingSort(arr))

#基数排序
# class Solution:
#     def radixSort(self, arr):
#         #数组中最大值是几位
#         size = len(str(max(arr)))
#         #遍历位数
#         for i in range(size):
#             buckets = [[] for _ in range(10)]
#             for num in arr:
#                 #把i对应的位数提取出来，放到相应的buckets里
#                 buckets[num // (10 ** i) % 10].append(num)
#             arr.clear()
#             #放回arr
#             for bucket in buckets:
#                 for num in bucket:
#                     arr.append(num)
#         return arr
#
#     def sortArray(self, nums):
#         return self.radixSort(nums)
#
# solution = Solution()
# arr = [-1,-3,100]
# print(solution.radixSort(arr))

#fork
# class Solution:
#     def radixSort(self,arr):
#         size = len(str(max(arr)))
#         for i in range(size):
#             buckets = [[] for _ in range(10)]
#             #填充buckets
#             for num in arr:
#                 buckets[num//(10**i)%10].append(num)
#             arr.clear()
#             #填充回arr
#             for bucket in buckets:
#                 for num in bucket:
#                     arr.append(num)
#         return arr
#     def sortArray(self,nums):
#         return self.radixSort(nums)
#
# solution = Solution()
# arr = [-1,-3,100]
# print(solution.radixSort(arr))

# print(-11//(10**0)%10)


# merged = []
# if not merged:
#     print('merged is')

#二分查找
# class Solution:
#     def search(self, nums, target):
#         left = 0
#         right = len(nums) - 1
#         # 在区间 [left, right] 内查找 target
#         while left <= right:
#             # 取区间中间节点
#             mid = (left + right) // 2
#             # 如果找到目标值，则直接返回中心位置
#             if nums[mid] == target:
#                 return mid
#             # 如果 nums[mid] 小于目标值，则在 [mid + 1, right] 中继续搜索
#             elif nums[mid] < target:
#                 left = mid + 1
#             # 如果 nums[mid] 大于目标值，则在 [left, mid - 1] 中继续搜索
#             else:
#                 right = mid - 1
#         # 未搜索到元素，返回 -1
#         return -1
#
# solution = Solution()
# print(solution.search([0,1,2,3],3))
#
# #fork
#
# class Solution:
#     def search(self, nums, target):
#         left = 0
#         right = len(nums)-1
#
#         while left<=right:
#             mid = (left+right)//2
#             if target == nums[mid]:
#                 return mid
#             elif target > nums[mid]:
#                 left = mid + 1
#             else:
#                 right = mid - 1
#         return -1

# mid_num = []
# print(mid_num is None)
import torch
import pandas


# class Solution:
#     def findMin(self, nums: List[int]) -> int:
#         left, right = 0, len(nums) - 1
#         while left < right:
#             mid = (left + right) >> 1
#             if nums[mid] > nums[right]:
#                 left = mid + 1
#             else:
#                 right = mid
#         return nums[left]

# list1 = [3,3,4,4,5,5,1,2]
# print(list(set(list1)))

# nums = []
# #表示nums为空
# print(not nums)

# print(False==0)

# print(True==1)

# class Solution:
#     def findPeakElement(self, nums) -> int:
#         n = len(nums)
#         lo, hi = 0, n - 1
#         # 我们这里使用的是 <= 所以后面的边界判定是 +1和-1
#         while lo <= hi:
#             mid = lo + (hi - lo) // 2
#             # 处理边界情况，按照题意 nums[-1] = nums[n] = -∞
#             if mid == 0:
#                 left = float("-inf")
#             else:
#                 left = nums[mid - 1]
#             if mid == n - 1:
#                 right = float("-inf")
#             else:
#                 right = nums[mid + 1]
#             # 判断当前mid下标是否符合峰值要求
#             if left < nums[mid] and nums[mid] > right:
#                 return mid
#             # 若单调递增，则查找后半段
#             elif left < nums[mid]:
#                 lo = mid + 1
#             # 若递减，则查找前半段
#             else:
#                 hi = mid - 1

# print("a"<"b")

# class Solution:
#     def findMedianSortedArrays(self, nums1, nums2) -> float:
#         for num2 in nums2:
#             left,right = 0,len(nums1)-1
#             while left<=right:
#                 mid = (left+right)//2
#                 #定义某点的左右值
#                 if mid == 0:
#                     l = float("-inf")
#                 else:
#                     l = nums1(mid-1)
#
#                 if mid == len(nums1)-1:
#                     r = float("+inf")
#                 else:
#                     r = nums1[mid+1]
#                 #判断
#                 if l<=num2<=r:
#                     nums1.insert(mid+1,num2)
#                     break
#                 elif num2>l and num2>r:
#                     right = mid-1
#                 else:
#                     left = mid+1
#         return nums1
#         # if len(nums1)//2==0:
#         #     return (nums1[len(nums1)//2]+nums1[len(nums1)//2-1])/2
#         # else:
#         #     return nums1[len(nums1)//2]
# solution = Solution()
# nums1 = [1,2]
# nums2 = [3,4]
# print(solution.findMedianSortedArrays(nums1,nums2))

# class Solution:
#     def findDuplicate(self, nums) -> int:
#         left = 1
#         right = len(nums) - 1
#         while left < right:
#             mid = left + (right - left) // 2
#             cnt = 0
#             for num in nums:
#                 if num <= mid:
#                     cnt += 1
#             if cnt > mid:
#                 right = mid
#             else:
#                 left = mid + 1
#         return left

# n = 5
# while n:
#     print(n)
#     n-=1

# class Solution(object):
#     def myPow(self, x: float, n: int) -> float:
#         if n == 0:
#             return 1
#         if n < 0:
#             x = 1 / x
#             n = -n
#         if n % 2:
#             return x * self.myPow(x, n - 1)
#         return self.myPow(x * x, n / 2)

# class Solution:
#     def findBestValue(self, arr, target) -> int:
#         arr.sort()
#         n = len(arr)
#         prefix = [0]
#         #对arr做累加
#         for num in arr:
#             prefix.append(prefix[-1] + num)
#         #右边界为max(arr),answer初值为0，target赋给diff
#         r, ans, diff = max(arr), 0, target
#         #枚举所有的answer
#         for i in range(1, r + 1):
#             #
#             it = bisect.bisect_left(arr, i)
#             #
#             cur = prefix[it] + (n - it) * i
#             if abs(cur - target) < diff:
#                 ans, diff = i, abs(cur - target)
#         return ans



# class Solution:
#     def findBestValue(self, arr, target: int) -> int:
#         #求value==diff时的数组总和
#         def difference_value(diff):
#             sum0 = 0
#             for num in arr:
#                 sum0 += min(num, diff)
#             return sum0
#
#
#         left, right = 0, max(arr)
#         while left < right:
#             mid = left + (right - left)//2
#             if difference_value(mid) < target:
#                 left = mid + 1
#             else:
#                 right = mid
#
#         sum1, sum2 = difference_value(left), difference_value(left - 1)
#         #如果
#         return left if sum1 - target < target - sum2 else left - 1

# class Solution:
#     def minSubArrayLen(self, s: int, nums: List[int]) -> int:
#         #由于是求窗口大小所以这里right=len(nums)
#         left, right, res = 0, len(nums), 0
#         #判断窗口大小是否符合条件
#         def helper(size):
#             sum_size = 0
#             for i in range(len(nums)):
#                 sum_size += nums[i]
#                 #size:滑动窗口大小；如果下标大于等于窗口大小
#                 if i >= size:
#                     #减掉前面导致窗口过大的元素
#                     sum_size -= nums[i-size]
#                 #sun>=target
#                 if sum_size >= s:
#                     return True
#             return False
#         #二分查找（窗口大小）
#         while left<=right:
#             mid = (left+right)//2  # 滑动窗口大小
#             if helper(mid):  # 如果这个大小的窗口可以那么就缩小
#                 res = mid
#                 #缩小窗口以找到最小窗口
#                 right = mid-1
#             else:  # 否则就增大窗口
#                 left = mid+1
#         return res

# my_dict = {1:1,2:2,3:3}
#
# for i in my_dict.keys():
#     print(i)
#
# output = [11,23,4,3,1,54]
# print(output[:3])


# class Solution:
#     def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
#         # 二分法，始终对长度为k的连续子数组进行操作，最终确定起点位置即可，即左端点
#         n = len(arr)
#         # 最大的起点为n-k，这样才能保证选取长度为k的连续子数组
#         left, right = 0, n - k
#         while left < right:
#             mid = (left + right) // 2
#             # mid与mid+k分别为当前的左右端点
#             if x - arr[mid] <= arr[mid+k] - x:
#                 right = mid
#             else:
#                 left = mid + 1
#         return arr[left:left+k]

#mydataset是元组嵌套列表的形式
#每张图片张量维度为[3,224,224]，标签张量为[1]
#共4072个图片、标签对
#以下是Dataset类
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self,data):
        self.imgs = data

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_tensor =  self.imgs[idx][0]
        label = self.imgs[idx][1]
        return image_tensor,label




import sys, gc
from torch.utils.data import random_split, SubsetRandomSampler
from matplotlib import pyplot as plt

fold_num = 0
batch_size = 128
num_epochs = 10

df_data = CustomDataset(mydataset)

for train_index, test_index in fold.split(mydataset):
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)
    print(f'第{fold_num + 1}折模型')
    print(f'训练集大小：{len(train_index)}\n', f'测试集大小:{len(test_index)}')

    train_iter = DataLoader(df_data, batch_size, sampler=train_sampler)
    test_iter = DataLoader(df_data, batch_size, sampler=test_sampler)
    X, y = next(iter(train_iter))

    training_history = {"train_loss": [], "train_accuracy": [], "epoch": [], "test_accuracy": []}
    valid_loss_min = np.Inf
    best_loss = np.inf

    # 初始化模型、loss、优化器
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    metric = [0, 0, 0]

    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch, num_epochs - 1))
        #             metric = [0, 0, 0]
        model.train()

        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            metric[0] += l.item() * len(y)
            metric[1] += (y_hat.argmax(dim=1) == y).sum().item()
            metric[2] += len(y)

        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy(test_iter, model, device)

        training_history["train_loss"].append(train_l)
        training_history["train_accuracy"].append(train_acc)
        training_history["epoch"].append(epoch)
        training_history["test_accuracy"].append(test_acc)
        print('epoch %d, train_loss: %.4f, train_accuracy: %.4f, test_accuracy: %.4f'
              % (epoch, train_l, train_acc, test_acc))

    plt.plot(training_history["epoch"], training_history["train_loss"])
    plt.plot(training_history["epoch"], training_history["train_accuracy"])
    plt.plot(training_history["epoch"], training_history["test_accuracy"])
    plt.xlabel("number of epochs")
    plt.ylabel("objective function")
    plt.legend(["train_loss", "train_accuracy", "test_accuracy"])
    plt.show()
    plt.clf()

    torch.cuda.empty_cache()
    gc.collect()
    fold_num += 1