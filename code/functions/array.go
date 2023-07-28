package functions

// 704.二分查找
func search(nums []int, target int) int {
	left := 0
	right := len(nums)
	for left < right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid // 因为是左闭右开区间，所以这里不能是high = mid - 1，mid-1有可能是答案
		}
	}
	return -1
}
