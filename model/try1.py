
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

#Bubble Sort
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
matrix = [[1,2,3],[4,5,6],[7,8,9]]
res = []
count = 0
while matrix:
    count+=1
    print(count)
    res += matrix.pop(0)
    matrix = list(zip(*matrix))[::-1]

print(res)

