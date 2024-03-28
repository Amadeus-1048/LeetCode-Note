package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

func main() {
	n := 0
	_, _ = fmt.Scan(&n)
	inputs := bufio.NewScanner(os.Stdin)
	inputs.Scan()
	data := strings.Split(inputs.Text(), " ")
	nums := make([]int, len(data))
	for i, v := range data {
		val, _ := strconv.Atoi(v) //将字符串转换为int
		nums[i] = val
	}
	fmt.Println(solution(nums))
}

func solution(nums []int) int {
	/*
		nums = [1, 2, 3, 3, 3, 7]
		duplicate = [3, 3]
		newNums 的长度为 max(6,7)+1 = 8
		newNums = [0 1 2 3 0 0 0 7]
		在for遍历中，找到 newNums[i] != i 的情况。开始时 res = 0, index = 0
			找到newNums[4] != 4	，res += i - duplicate[index] -> res += 4-3 , res=1, index=1
			找到newNums[5] != 5	，res += i - duplicate[index] -> res += 5-3 , res=3, index=2
			index = 2 = len(duplicate) , 跳出循环
	*/

	if len(nums) < 2 {
		return 0
	}
	sort.Ints(nums)
	maxLen := max(len(nums), nums[len(nums)-1]) + 1
	duplicate := make([]int, 0)
	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[i-1] {
			duplicate = append(duplicate, nums[i])
		}
	}
	newNums := make([]int, maxLen+len(duplicate)) // 用于记录调整后的值，确保所有元素都是唯一的
	for i := 0; i < len(nums); i++ {
		// 遍历原数组，将元素的值用作新数组的索引，并在该位置存储相同的值。这一步确保了newArray中已有元素的位置不会被重复使用。
		newNums[nums[i]] = nums[i]
	}
	res := 0
	index := 0 // 用于跟踪dup列表中下一个需要处理的重复元素
	for i := 0; i < len(nums); i++ {
		if index == len(duplicate) {
			break
		}
		// 当找到newArray中的空位时，如果重复元素的值小于当前的索引i，则递增该重复元素，并增加计数器cnt以记录操作次数。
		if newNums[i] != i { // 找到空位
			if duplicate[index] < i {
				res += i - duplicate[index]
				index++
			}
		}
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
