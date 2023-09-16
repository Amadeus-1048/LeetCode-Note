package functions

import "math"

// 509. 斐波那契数
func fib(n int) int {
	if n < 2 {
		return n
	}
	a, b, c := 0, 1, 1
	for i := 2; i <= n; i++ {
		c = a + b
		a, b = b, c
	}
	return c
}

// 70. 爬楼梯
func climbStairs(n int) int {
	if n < 3 {
		return n
	}
	a, b, c := 1, 2, 3
	for i := 3; i <= n; i++ { // 如果从i=2开始遍历，那么判断条件要去掉=
		c = a + b
		a, b = b, c
	}
	return c
}

// 746. 使用最小花费爬楼梯
func minCostClimbingStairs(cost []int) int {
	a, b, c := 0, 0, 0
	for i := 2; i <= len(cost); i++ {
		c = min(a+cost[i-2], b+cost[i-1])
		a, b = b, c
	}
	return c
}

// 62. 不同路径
func uniquePaths(m int, n int) int {
	a := 1     // 分子  从m+n-2开始累乘m-1个数
	b := m - 1 // 分母 从 1 累乘到 m-1
	t := m + n - 2
	for i := m - 1; i > 0; i-- {
		a *= t
		t--
		for b != 0 && a%b == 0 {
			a /= b
			b--
		}
	}
	return a
}

// 63. 不同路径 II
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m && obstacleGrid[i][0] == 0; i++ { // 障碍之后的dp还是初始值0
		dp[i][0] = 1
	}
	for j := 0; j < n && obstacleGrid[0][j] == 0; j++ { // 障碍之后的dp还是初始值0
		dp[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if obstacleGrid[i][j] != 1 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

// 343. 整数拆分
func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i := 3; i <= n; i++ {
		for j := 1; j <= i-2; j++ {
			dp[i] = max(dp[i], max(j*(i-j), dp[i-j]*j))
		}
	}
	return dp[n]
}

// 96. 不同的二叉搜索树
func numTrees(n int) int {
	dp := make([]int, n+1)
	// dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]
	dp[0] = 1
	for i := 1; i <= n; i++ {
		for j := 1; j <= i; j++ {
			// j-1 为 以j为头结点左子树节点数量
			// i-j 为以j为头结点右子树节点数量
			dp[i] += dp[j-1] * dp[i-j]
		}
	}
	return dp[n]
}

// 416. 分割等和子集
func canPartition(nums []int) bool {
	n := len(nums)
	sum := 0
	for i := 0; i < n; i++ {
		sum += nums[i]
	}
	if sum%2 == 1 {
		return false
	} else {
		sum /= 2
	}

	dp := make([]int, sum+1)
	for i := 0; i < n; i++ {
		for j := sum; j >= nums[i]; j-- { // 每一个元素一定是不可重复放入，所以从大到小遍历
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[sum] == sum // 集合中的元素正好可以凑成总和target
}

// 1049. 最后一块石头的重量 II
func lastStoneWeightII(stones []int) int {
	n := len(stones)
	sum := 0
	target := 0
	for i := 0; i < n; i++ {
		sum += stones[i]
	}
	target = sum / 2 // target总是较小的，且dp[target]<=target

	dp := make([]int, sum+1)
	for i := 0; i < n; i++ {
		for j := target; j >= stones[i]; j-- { // 每一个元素一定是不可重复放入，所以从大到小遍历
			dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
		}
	}
	return sum - dp[target] - dp[target]
}

// 494. 目标和
func findTargetSumWays(nums []int, target int) int {
	n := len(nums)
	sum := 0
	for i := 0; i < n; i++ {
		sum += nums[i]
	}
	x := (sum + target) / 2
	if (sum+target)%2 == 1 || abs(target) > sum {
		return 0
	}
	dp := make([]int, x+1)
	dp[0] = 1
	for i := 0; i < n; i++ {
		for j := x; j >= nums[i]; j-- {
			dp[j] += dp[j-nums[i]]
		}
	}
	return dp[x]
}

// 474. 一和零
func findMaxForm(strs []string, m int, n int) int {
	one, zero := 0, 0
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i < len(strs); i++ { // 遍历物品
		zero, one = 0, 0 // 物品i中0和1的数量
		for _, v := range strs[i] {
			if v == '0' {
				zero++
			}
		}
		one = len(strs[i]) - zero
		for j := m; j >= zero; j-- { // 遍历背包容量且从后向前遍历！
			for k := n; k >= one; k-- {
				dp[j][k] = max(dp[j][k], dp[j-zero][k-one]+1)
			}
		}
	}
	return dp[m][n]
}

// 518. 零钱兑换 II
func change(amount int, coins []int) int {
	n := len(coins)
	dp := make([]int, amount+1)
	dp[0] = 1
	for i := 0; i < n; i++ { // 遍历物品
		for j := coins[i]; j <= amount; j++ { // 遍历背包
			dp[j] += dp[j-coins[i]]
		}
	}
	return dp[amount]
}

// 377. 组合总和 Ⅳ
func combinationSum4(nums []int, target int) int {
	n := len(nums)
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 0; i <= target; i++ { // 遍历背包
		for j := 0; j < n; j++ { // 遍历物品
			if i >= nums[j] {
				dp[i] += dp[i-nums[j]]
			}
		}
	}
	return dp[target]
}

// 322. 零钱兑换
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for j := 1; j <= amount; j++ { // 遍历背包
		dp[j] = math.MaxInt32
		for i := 0; i < len(coins); i++ { // 遍历物品
			if j >= coins[i] {
				dp[j] = min(dp[j], dp[j-coins[i]]+1)
			}
		}
	}
	if dp[amount] == math.MaxInt32 {
		return -1
	} else {
		return dp[amount]
	}
}

// 279. 完全平方数
func numSquares(n int) int {
	dp := make([]int, n+1)
	for j := 1; j <= n; j++ { // 遍历背包
		dp[j] = math.MaxInt32
		for i := 1; i*i <= j; i++ { // 遍历物品
			if j >= i*i {
				dp[j] = min(dp[j], dp[j-i*i]+1)
			}
		}
	}
	return dp[n]
}

// 198. 打家劫舍
func rob(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	if n == 2 {
		return max(nums[0], nums[1])
	}
	dp := make([]int, n)
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < n; i++ {
		dp[i] = max(dp[i-2]+nums[i], dp[i-1])
	}
	return dp[n-1]
}

// 213. 打家劫舍 II
func rob2(nums []int) int {
	n := len(nums)
	if n == 1 {
		return nums[0]
	}
	if n == 2 {
		return max(nums[0], nums[1])
	}
	res1 := robRange(nums, 0, n-1) // 考虑包含首元素，不包含尾元素
	res2 := robRange(nums, 1, n)   // 考虑包含尾元素，不包含首元素
	return max(res1, res2)
}

func robRange(nums []int, start, end int) int {
	dp := make([]int, len(nums))
	dp[start] = nums[start]
	dp[start+1] = max(nums[start], nums[start+1])
	for i := start + 2; i < end; i++ {
		dp[i] = max(dp[i-2]+nums[i], dp[i-1])
	}
	return dp[end-1]
}

// 337. 打家劫舍 III
func rob3(root *TreeNode) int {
	var search func(node *TreeNode) []int
	search = func(node *TreeNode) []int {
		if node == nil {
			return []int{0, 0}
		}
		// 后续遍历
		left := search(node.Left)
		right := search(node.Right)
		sum1 := node.Val + left[0] + right[0]                   // 偷当前结点
		sum0 := max(left[0], left[1]) + max(right[1], right[0]) // 不偷当前结点，所以可以考虑偷或不偷子节点
		return []int{sum0, sum1}                                // {不偷，偷}
	}
	res := search(root)
	return max(res[0], res[1])
}

// 121. 买卖股票的最佳时机
func maxProfit1(prices []int) int {
	dp := [2]int{}
	dp[0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[0] = max(dp[0], -prices[i])      // 持有股票所得最多现金
		dp[1] = max(dp[1], dp[0]+prices[i]) // 不持有股票所得最多现金
	}
	return dp[1]
}
