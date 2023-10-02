package functions

import "sort"

// 215. 数组中的第K个最大元素
func findKthLargest(nums []int, k int) int {
	start, end := 0, len(nums)-1
	for {
		if start >= end {
			return nums[end]
		}
		p := quickSortPartition(nums, start, end)
		if p+1 == k { // 第K大   0 1 2 对应：第一大 第二大 第三大
			return nums[p]
		} else if p+1 < k { // 对p的右边数组进行分治, 即对 [p+1,right]进行分治
			start = p + 1
		} else { // 对p的左边数组进行分治, 即对 [left,p-1]进行分治
			end = p - 1
		}
	}
}

func quickSortPartition(nums []int, start, end int) int {
	// 从大到小排序
	pivot := nums[end]
	for i := start; i < end; i++ {
		if nums[i] > pivot { // 大的放左边
			nums[start], nums[i] = nums[i], nums[start]
			start++
		}
	}
	// for循环完毕, nums[start]左边的值, 均大于nums[start]右边的值
	nums[start], nums[end] = nums[end], nums[start] // 此时nums[end]是nums[start]右边最大的值，需要交换一下
	return start                                    // 确定了nums[start]的位置
}

// 912. 排序数组
func sortArray(nums []int) []int {
	var quick func(left, right int)
	quick = func(left, right int) {
		// 递归终止条件
		if left >= right {
			return
		}
		pivot := nums[right] // 左右指针及主元
		start, end := left, right
		for i := start; i < end; i++ { // start前面的都是小于pivot的
			if nums[i] < pivot {
				nums[start], nums[i] = nums[i], nums[start]
				start++
			}
		}
		nums[start], nums[end] = nums[end], nums[start] // 确定了start的位置
		quick(left, start-1)
		quick(start+1, right)
	}
	quick(0, len(nums)-1)
	return nums
}

// 56. 合并区间
func mergeIntervals(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	res := [][]int{}
	prev := intervals[0]
	// 合并区间
	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		// 前一个区间的右边界和当前区间的左边界进行比较，判断有无重合
		if prev[1] < cur[0] { // 没有重合
			res = append(res, prev) // 前一个区间合并完毕，加入结果集
			prev = cur
		} else { // 有重合
			prev[1] = max(prev[1], cur[1]) // 合并后的区间右边界为较大的那个
		}
	}
	// 当考察完最后一个区间，后面没区间了，遇不到不重合区间，最后的 prev 没推入 res。 要单独补上
	res = append(res, prev)
	return res
}
