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

// 122. 买卖股票的最佳时机 II
func maxProfit2(prices []int) int {
	dp := [2]int{}
	dp[0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[0] = max(dp[0], dp[1]-prices[i]) // 今天持有股票=max(昨天持有，昨天未持有且今天买入）
		dp[1] = max(dp[1], dp[0]+prices[i]) // 今天不持有股票=max(昨天未持有，昨天持有且今天卖出）
	}
	return dp[1]
}

// 123. 买卖股票的最佳时机 III
func maxProfit3(prices []int) int {
	dp := [5]int{}
	dp[1] = -prices[0] // 第一次持有股票，即买入第一天
	dp[3] = -prices[0] // 第二次持有股票，即买入第一天
	for i := 1; i < len(prices); i++ {
		dp[1] = max(dp[1], dp[0]-prices[i]) // 第一次持有股票=max(昨天持有，昨天未持有且今天买入)
		dp[2] = max(dp[2], dp[1]+prices[i]) // 第一次不持有股票=max(昨天不持有，昨天持有且今天卖出)
		dp[3] = max(dp[3], dp[2]-prices[i]) // 第二次持有股票=max(昨天持有，昨天未持有且今天买入)
		dp[4] = max(dp[4], dp[3]+prices[i]) // 第二次不持有股票=max(昨天不持有，昨天持有且今天卖出)
	}
	return dp[4]
}

// 188. 买卖股票的最佳时机 IV
func maxProfit4(k int, prices []int) int {
	if k == 0 || len(prices) == 0 {
		return 0
	}
	n := 2*k + 1 // 买k次，卖k次，加上什么操作都没有的0，所以n=2k+1
	dp := make([]int, n)
	for i := 1; i < n-1; i += 2 {
		dp[i] = -prices[0] // 第i(奇数次)持有股票，即买入第一天
	}
	for i := 1; i < len(prices); i++ {
		for j := 0; j < n-1; j += 2 { // 除了0以外，偶数就是卖出，奇数就是买入
			dp[j+1] = max(dp[j+1], dp[j]-prices[i])   // 持有股票=max(昨天持有，昨天未持有且今天买入)
			dp[j+2] = max(dp[j+2], dp[j+1]+prices[i]) // 不持有股票=max(昨天不持有，昨天持有且今天卖出)
		}
	}
	return dp[n-1]
}

// 309. 买卖股票的最佳时机含冷冻期
func maxProfit5(prices []int) int {
	n := len(prices)
	if n < 2 {
		return 0
	}

	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]int, 4)
	}
	dp[0][0] = -prices[0]

	for i := 1; i < n; i++ {
		// 达到买入股票状态：前一天就是持有股票状态、前一天是保持卖出股票的状态且今天买入、前一天是冷冻期且今天买入
		dp[i][0] = max(dp[i-1][0], max(dp[i-1][1]-prices[i], dp[i-1][3]-prices[i]))
		// 达到保持卖出股票状态：前一天是保持卖出股票的状态、前一天是冷冻期
		dp[i][1] = max(dp[i-1][1], dp[i-1][3])
		// 达到今天就卖出股票状态：昨天一定是持有股票状态且今天卖出
		dp[i][2] = dp[i-1][0] + prices[i]
		// 达到冷冻期状态：昨天卖出了股票
		dp[i][3] = dp[i-1][2]
	}

	return max(dp[n-1][1], max(dp[n-1][2], dp[n-1][3]))
}

// 714. 买卖股票的最佳时机含手续费
func maxProfit6(prices []int, fee int) int {
	dp := [2]int{}
	dp[0] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[0] = max(dp[0], dp[1]-prices[i])     // 买入
		dp[1] = max(dp[1], dp[0]+prices[i]-fee) // 卖出  区别就是这里多了一个减去手续费的操作
	}
	return dp[1]
}

// 300. 最长递增子序列
func lengthOfLIS(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	ans := 1
	dp := make([]int, len(nums)) // dp[i] 为以第 i 个数字结尾的最长上升子序列的长度，nums[i] 必须被选取
	dp[0] = 1
	for i := 1; i < len(nums); i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] { // if成立时，位置i的最长升序子序列 可以等于 位置j的最长升序子序列 + 1
				dp[i] = max(dp[i], dp[j]+1) // dp[i] 从 dp[j] 这个状态转移过来
			}
		}
		ans = max(ans, dp[i])
	}
	return ans
}

// 674. 最长连续递增序列
func findLengthOfLCIS(nums []int) int {
	n := len(nums)
	if n < 2 { // 不能漏这一步，否则有样例过不了
		return 1
	}
	dp := make([]int, n)
	res := 0
	for i := 0; i < n; i++ {
		dp[i] = 1 // dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	}
	for i := 0; i < n-1; i++ {
		if nums[i+1] > nums[i] {
			dp[i+1] = dp[i] + 1
		}
		res = max(res, dp[i+1])
	}
	return res
}

// 718. 最长重复子数组

func findLength(A []int, B []int) int {
	m, n := len(A), len(B)
	dp := make([][]int, m+1) // dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]
	res := 0
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if A[i-1] == B[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			}
			res = max(res, dp[i][j])
		}

	}
	return res
}

// 1143. 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	// dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				// 状态转移方程 : 主要就是两大情况： text1[i - 1] 与 text2[j - 1]相同，text1[i - 1] 与 text2[j - 1]不相同
				// 如果text1[i - 1] 与 text2[j - 1]相同，那么找到了一个公共元素，所以dp[i][j] = dp[i - 1][j - 1] + 1;
				// 如果text1[i - 1] 与 text2[j - 1]不相同，即：dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}

	}
	return dp[m][n]
}

// 53. 最大子数组和
func maxSubArrayDP(nums []int) int {
	// dp[i]：包括下标i之前的最大连续子序列和为dp[i]
	n := len(nums)
	dp := make([]int, n)
	// dp[i]的初始化	由于dp 状态转移方程依赖dp[0]
	dp[0] = nums[0]
	// 初始化最大的和
	res := nums[0]
	for i := 1; i < n; i++ {
		// dp[i]只有两个方向可以推出来：nums[i]加入当前连续子序列和	从头开始计算当前连续子序列和
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		res = max(res, dp[i])
	}
	return res
}

// 5. 最长回文子串
func longestPalindrome(s string) string {
	n := len(s)
	if n == 1 {
		return s
	}
	start, maxLen := 0, 1
	// dp[i][j] 表示 s[i..j] 是否是回文串
	dp := make([][]bool, n)
	// 初始化：所有长度为 1 的子串都是回文串
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
		dp[i][i] = true
	}
	for Len := 2; Len <= n; Len++ { // 先枚举子串长度
		for i := 0; i < n; i++ { // 枚举左边界，左边界的上限设置可以宽松一些
			j := i + Len - 1 // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
			if j >= n {      // 如果右边界越界，就可以退出当前循环
				break
			}
			if s[i] != s[j] {
				dp[i][j] = false
			} else {
				if j-i < 2 {
					dp[i][j] = true // 下标i 与 j相同（同一个字符例如a） 或 相差为1（例如aa），都是回文子串
				} else {
					dp[i][j] = dp[i+1][j-1] // 看dp[i + 1][j - 1]是否为true
				}
			}
			// 只要 dp[i][j] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
			if dp[i][j] && j-i+1 > maxLen {
				maxLen = j - i + 1
				start = i
			}
		}
	}
	return s[start : start+maxLen]
}
