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
