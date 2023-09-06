package functions

import "sort"

// 455. 分发饼干
func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	res := 0
	child := len(g) - 1
	for i := len(s) - 1; i >= 0 && child >= 0; child-- {
		if s[i] >= g[child] {
			res++
			i--
		}
	}
	return res
}

// 376. 摆动序列
func wiggleMaxLength(nums []int) int {
	count, preDiff, curDiff := 1, 0, 0 // 序列默认序列最右边有一个峰值
	if len(nums) < 2 {
		return 1
	}
	for i := 0; i < len(nums)-1; i++ {
		curDiff = nums[i+1] - nums[i]
		// //如果有正有负则更新下标值||或者只有前一个元素为0（针对两个不等元素的序列也视作摆动序列，且摆动长度为2）
		if (curDiff > 0 && preDiff <= 0) || (preDiff >= 0 && curDiff < 0) {
			preDiff = curDiff
			count++ // 统计数组的峰值数量	相当于是删除单一坡度上的节点，然后统计长度
		}
	}
	return count
}

// 53. 最大子数组和
func maxSubArray(nums []int) int {
	length := len(nums)
	if length == 1 {
		return nums[0]
	}
	res, sum := nums[0], nums[0]
	for i := 1; i < length; i++ {
		if sum < 0 { // 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
			sum = nums[i]
		} else {
			sum += nums[i] // 取区间累计的最大值（相当于不断确定最大子序终止位置）
		}
		res = max(res, sum)
	}
	return res
}

// 122. 买卖股票的最佳时机 II
func maxProfit(prices []int) int {
	sum := 0
	for i := 1; i < len(prices); i++ { // 第一天没有利润，至少要第二天才会有利润
		if prices[i]-prices[i-1] > 0 {
			sum += prices[i] - prices[i-1]
		}
	}
	return sum
}

// 55. 跳跃游戏
func canJump(nums []int) bool {
	mx := 0
	for i, num := range nums {
		if i > mx {
			return false
		}
		mx = max(mx, i+num)
	}
	return true
}

// 45. 跳跃游戏 II
func jump(nums []int) int {
	curDistance := 0                   // 当前覆盖的最远距离下标
	ans := 0                           // 记录走的最大步数
	nextDistance := 0                  // 下一步覆盖的最远距离下标
	for i := 0; i < len(nums)-1; i++ { // 注意这里是小于nums.size() - 1，这是关键所在
		if nums[i]+i > nextDistance {
			nextDistance = nums[i] + i // 更新下一步覆盖的最远距离下标
		}
		if i == curDistance { // 遇到当前覆盖的最远距离下标，直接步数加一
			curDistance = nextDistance // 更新当前覆盖的最远距离下标
			ans++
		}
	}
	return ans
}
