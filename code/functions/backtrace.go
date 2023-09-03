package functions

import (
	"fmt"
	"sort"
	"strconv"
)

// 77. 组合
func combine(n int, k int) [][]int {
	res := [][]int{}
	var backtrace func(start int, trace []int)
	backtrace = func(start int, trace []int) {
		if len(trace) == k {
			tmp := make([]int, k)
			copy(tmp, trace)
			res = append(res, tmp)
		}
		if len(trace)+n-start+1 < k { // 剪枝优化
			return
		}
		for i := start; i <= n; i++ { // 选择本层集合中元素，控制树的横向遍历
			trace = append(trace, i)     // 处理节点
			backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
			trace = trace[:len(trace)-1] // 回溯，撤销处理结果
		}
	}
	backtrace(1, []int{})
	return res
}

// 216. 组合总和 III
func combinationSum3(k int, n int) [][]int {
	res := [][]int{}
	var backtrace func(start int, trace []int)
	backtrace = func(start int, trace []int) {
		if len(trace) == k {
			sum := 0
			tmp := make([]int, k)
			for i, v := range trace {
				sum += v
				tmp[i] = v
			}
			if sum == n {
				res = append(res, tmp)
			}
			return
		}
		if start > n { // 剪枝优化
			return
		}
		for i := start; i <= 9; i++ { // 选择本层集合中元素，控制树的横向遍历
			trace = append(trace, i)     // 处理节点
			backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
			trace = trace[:len(trace)-1] // 回溯，撤销处理结果
		}
	}
	backtrace(1, []int{})
	return res
}

// 17. 电话号码的字母组合
func letterCombinations(digits string) []string {
	digitsMap := [10]string{
		"",     // 0
		"",     // 1
		"abc",  // 2
		"def",  // 3
		"ghi",  // 4
		"jkl",  // 5
		"mno",  // 6
		"pqrs", // 7
		"tuv",  // 8
		"wxyz", // 9
	}

	res := []string{}
	length := len(digits)
	if length <= 0 || length > 4 {
		return res
	}
	var backtrace func(s string, index int)
	backtrace = func(s string, index int) {
		if len(s) == length {
			res = append(res, s)
			return
		}
		num := digits[index] - '0' // 将index指向的数字转为int
		letter := digitsMap[num]   // 取数字对应的字符集
		for i := 0; i < len(letter); i++ {
			s += string(letter[i])
			backtrace(s, index+1)
			s = s[:len(s)-1]
		}
	}
	backtrace("", 0)
	return res
}

// 39. 组合总和
func combinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	trace := []int{}
	var backtrace func(start, sum int)
	backtrace = func(start, sum int) {
		// 本题没有组合数量要求，仅仅是总和的限制，所以递归没有层数的限制，只要选取的元素总和超过target，就返回
		if sum == target {
			tmp := make([]int, len(trace)) // 注意这里  测试案例输出：[[2,2,3],[7]]
			copy(tmp, trace)               // 必须创建一个用来拷贝的，使用copy函数
			res = append(res, tmp)
			return
		}
		if sum > target {
			return
		}
		// start用来控制for循环的起始位置
		for i := start; i < len(candidates); i++ {
			trace = append(trace, candidates[i])
			sum += candidates[i]
			backtrace(i, sum) // 本题元素为可重复选取的，所以关键点:不用i+1了，表示可以重复读取当前的数
			trace = trace[:len(trace)-1]
			sum -= candidates[i]
		}
	}
	backtrace(0, 0)
	return res
}

// 40. 组合总和 II
func combinationSum2(candidates []int, target int) [][]int {
	res := [][]int{}      // 存放组合集合
	trace := []int{}      // 符合条件的组合
	sort.Ints(candidates) // 首先把给candidates排序，让其相同的元素都挨在一起。
	var backtrace func(start, sum int)
	backtrace = func(start, sum int) {
		if sum == target {
			tmp := make([]int, len(trace))
			copy(tmp, trace)
			res = append(res, tmp)
			return
		}
		if sum > target {
			return
		}

		for i := start; i < len(candidates); i++ {
			// 前一个树枝，使用了candidates[i - 1]，也就是说同一树层使用过candidates[i - 1]。
			if i > start && candidates[i] == candidates[i-1] {
				continue
			}
			trace = append(trace, candidates[i])
			sum += candidates[i]
			backtrace(i+1, sum) // 和39.组合总和的区别1，这里是i+1，每个数字在每个组合中只能使用一次
			trace = trace[:len(trace)-1]
			sum -= candidates[i]
		}
	}
	backtrace(0, 0)
	return res
}

// 131. 分割回文串
func partition(s string) [][]string {
	res := [][]string{}
	trace := []string{}
	var backtrace func(start int)
	backtrace = func(start int) {
		if start == len(s) {
			tmp := make([]string, len(trace))
			copy(tmp, trace)
			res = append(res, tmp)
			return
		}

		for i := start; i < len(s); i++ { // 横向遍历：找切割线  切割到字符串的结尾位置
			if isPartition(s, start, i) { // 是回文子串
				trace = append(trace, s[start:i+1]) // 左开右闭，所以i+1
			} else {
				continue
			}
			backtrace(i + 1) // i+1 表示下一轮递归遍历的起始位置
			trace = trace[:len(trace)-1]
		}
	}
	backtrace(0)
	return res
}

// 判断是否为回文
func isPartition(s string, startIndex, end int) bool {
	for startIndex < end {
		if s[startIndex] != s[end] {
			return false
		}
		//移动左右指针
		startIndex++
		end--
	}
	return true
}

// 93. 复原 IP 地址
func restoreIpAddresses(s string) []string {
	var res, path []string
	var backtrace func(start int)
	backtrace = func(start int) {
		if start == len(s) && len(path) == 4 {
			tmpString := path[0] + "." + path[1] + "." + path[2] + "." + path[3]
			fmt.Println("tmpString:", tmpString)
			res = append(res, tmpString)
		}
		for i := start; i < len(s); i++ {
			path = append(path, s[start:i+1])
			fmt.Println("path:", path)
			if i-start+1 <= 3 && len(path) <= 4 && isIP(s, start, i) {
				backtrace(i + 1)
			} else {
				path = path[:len(path)-1]
				return // 直接返回
			}
			path = path[:len(path)-1]
		}
	}
	backtrace(0)
	return res
}

// 判断字符串s在左闭右闭区间[start, end]所组成的数字是否合法
func isIP(s string, start int, end int) bool {
	check, err := strconv.Atoi(s[start : end+1])
	if err != nil { // 遇到非数字字符不合法
		return false
	}
	if end-start+1 > 1 && s[start] == '0' { // 0开头的数字不合法
		return false
	}
	if check > 255 { // 大于255了不合法
		return false
	}
	return true
}

// 78. 子集
func subsets(nums []int) [][]int {
	res := [][]int{}
	trace := []int{}
	sort.Ints(nums)
	var backtrace func(start int)
	backtrace = func(start int) {
		tmp := make([]int, len(trace))
		copy(tmp, trace)
		res = append(res, tmp)
		for i := start; i < len(nums); i++ {
			trace = append(trace, nums[i])
			backtrace(i + 1) // 取过的元素不会重复取
			trace = trace[:len(trace)-1]
		}
	}
	backtrace(0)
	return res
}

// 90. 子集 II
func subsetsWithDup(nums []int) [][]int {
	res := [][]int{}
	trace := []int{}
	sort.Ints(nums) // 去重需要排序
	var backtrace func(start int)
	backtrace = func(start int) {
		tmp := make([]int, len(trace))
		copy(tmp, trace)
		res = append(res, tmp)
		for i := start; i < len(nums); i++ {
			if i > start && nums[i] == nums[i-1] { // 对同一树层使用过的元素进行跳过
				continue
			}
			trace = append(trace, nums[i])
			backtrace(i + 1)
			trace = trace[:len(trace)-1]
		}
	}
	backtrace(0)
	return res
}
