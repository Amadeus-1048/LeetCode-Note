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

// 27.移除元素
func removeElement(nums []int, val int) int {
	length := len(nums)
	slow, fast := 0, 0
	for fast < length {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
			fast++
		} else {
			fast++
		}
	}
	return slow
}

// 977.有序数组的平方
func sortedSquares(nums []int) []int {
	n := len(nums)
	i, j, k := 0, n-1, n-1
	ans := make([]int, n)
	for i <= j {
		left, right := nums[i]*nums[i], nums[j]*nums[j]
		if left < right {
			ans[k] = right
			j--
		} else {
			ans[k] = left
			i++
		}
		k--
	}
	return ans
}
