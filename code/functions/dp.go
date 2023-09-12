package functions

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
