package functions

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
