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

// 3. 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	ans, left, right := 1, 0, 1 // 滑动窗口
	hash := make(map[byte]bool)
	n := len(s)
	if n <= 1 {
		return n
	}
	hash[s[0]] = true
	for right < n {
		if !hash[s[right]] { // 没有遇到重复的字符，则s[right]存入map,计算长度，right推进
			hash[s[right]] = true
			ans = max(ans, right-left+1)
			right++
		} else { // 遇到重复字符，需要在map中去掉s[left]，left推进
			for hash[s[right]] { // 不断推进left，直到遇到和s[right]相同的字符
				delete(hash, s[left])
				left++
			}
			hash[s[right]] = true // 因为上面把s[right]给delete了，需要重新赋值
			right++
		}
	}
	return ans
}

// 33. 搜索旋转排序数组
func searchTwisted(nums []int, target int) int {
	n := len(nums)
	if n == 1 {
		if nums[0] == target {
			return 0
		} else {
			return -1
		}
	}
	left, right, mid := 0, n-1, 0
	for left <= right {
		mid = (left + right) / 2 // 二分法
		if nums[mid] == target { // 判断是否找到target
			return mid
		}
		if nums[0] <= nums[mid] { // 0~mid是有序的	这里必须加=号
			if nums[0] <= target && target < nums[mid] { // target在有序的0~mid范围内，进行查找
				right = mid - 1
			} else { // target不在0~mid范围内，在无序的mid+1~n-1范围内重新查找
				left = mid + 1
			}
		} else { // 0~mid是无序的，mid~n是有序的
			if nums[mid] < target && target <= nums[n-1] { // target在有序的mid~n-1范围内，进行查找
				left = mid + 1
			} else { // target不在mid~n-1范围内，在无序的0~mid-1范围内重新查找
				right = mid - 1
			}
		}
	}
	return -1
}

// 88. 合并两个有序数组
func merge(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1       // nums1 的初始长度为 m + n
	tail := m + n - 1      // 从后往前放置元素
	for i >= 0 || j >= 0 { // 只要有一个大于等于0，就表示还没合并完
		if i < 0 { // nums1全部用完，直接用nums2的
			nums1[tail] = nums2[j]
			j--
		} else if j < 0 { // nums2全部用完，直接用nums1的
			nums1[tail] = nums1[i]
			i--
		} else if nums1[i] <= nums2[j] {
			nums1[tail] = nums2[j]
			j--
		} else {
			nums1[tail] = nums1[i]
			i--
		}
		tail--
	}
}
