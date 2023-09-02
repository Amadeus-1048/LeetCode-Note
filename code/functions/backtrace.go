package functions

import "sort"

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
