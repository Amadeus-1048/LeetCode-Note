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

// 209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	i, sum := 0, 0
	length := len(nums)
	res := length + 1
	for j := 0; j < length; j++ {
		sum += nums[j]
		for sum >= target {
			tmp := j - i + 1
			if tmp < res {
				res = tmp
			}
			sum -= nums[i]
			i++
		}
	}
	if res == length+1 {
		return 0
	}
	return res
}

// 59. 螺旋矩阵II
func generateMatrix(n int) [][]int {
	top, bottom := 0, n-1
	left, right := 0, n-1
	num := 1 // 给矩阵赋值
	tar := n * n
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	for num <= tar { //
		for i := left; i <= right; i++ { // 左上到右
			matrix[top][i] = num
			num++
		}
		top++                            // 右上角往下一格
		for i := top; i <= bottom; i++ { // 右上到下
			matrix[i][right] = num
			num++
		}
		right--                          // 右下角往左一格
		for i := right; i >= left; i-- { // 右下到左
			matrix[bottom][i] = num
			num++
		}
		bottom--                         // 左下角往上一格
		for i := bottom; i >= top; i-- { // 左下到上
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}
