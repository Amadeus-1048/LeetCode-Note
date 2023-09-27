package functions

const INF = 0x3f3f3f3f

// 迪杰斯特拉函数
func dijkstra(graph [][]int, start int) []int {
	n := len(graph)         // 图中顶点个数
	visit := make([]int, n) // 标记已经作为中间结点完成访问的顶点
	dist := make([]int, n)  // 存储 起始点 到 其他顶点 的最短路径

	for i := 0; i < n; i++ {
		dist[i] = graph[start][i] // 初始化遍历起点
	}
	visit[start] = 1 // 标记初始顶点

	var minDist, midNode int // 找出并记录未访问且路径最短的结点

	// 更新其他顶点最短路径，循环n次
	for i := 0; i < n; i++ {
		minDist = INF // 存储从起点到其他未被访问的结点中的最短路径
		midNode = 0   // 存储最短路径的结点编号

		// 遍历n个顶点，寻找未被访问且距离为起始位置到该点距离最小的顶点
		for j := 0; j < n; j++ {
			if visit[j] == 0 && minDist > dist[j] {
				minDist = dist[j] // 更新未被访问结点的最短路径
				midNode = j       // 更新顶点编号
			}
		}

		// 以midNode为中间结点，再循环遍历其他节点更新最短路径
		for j := 0; j < n; j++ {
			// 若该节点未被访问且找到更短路径即更新最短路径
			if visit[j] == 0 && dist[j] > dist[midNode]+graph[midNode][j] {
				dist[j] = dist[midNode] + graph[midNode][j]
			}
		}

		visit[midNode] = 1 // 标记已访问

	}
	return dist
}

// 200. 岛屿数量
func numIslands(grid [][]byte) int {
	ans := 0
	m, n := len(grid), len(grid[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		grid[i][j] = '0' // '1'（陆地）  '0'（水）
		// 每遍历到一块陆地，就把这块陆地和与之相连的陆地全部变成水
		if i-1 >= 0 && grid[i-1][j] == '1' {
			dfs(i-1, j)
		}
		if i+1 < m && grid[i+1][j] == '1' {
			dfs(i+1, j)
		}
		if j-1 >= 0 && grid[i][j-1] == '1' {
			dfs(i, j-1)
		}
		if j+1 < n && grid[i][j+1] == '1' {
			dfs(i, j+1)
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == '1' {
				ans++
				dfs(i, j)
			}
		}
	}
	return ans
}

// 695. 岛屿的最大面积
func maxAreaOfIsland(grid [][]int) int {
	ans := 0
	m, n := len(grid), len(grid[0])
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		count := 1
		grid[i][j] = 0 // '1'（陆地）  '0'（水）
		// 每遍历到一块陆地，就把这块陆地和与之相连的陆地全部变成水
		if i-1 >= 0 && grid[i-1][j] == 1 {
			count += dfs(i-1, j)
		}
		if i+1 < m && grid[i+1][j] == 1 {
			count += dfs(i+1, j)
		}
		if j-1 >= 0 && grid[i][j-1] == 1 {
			count += dfs(i, j-1)
		}
		if j+1 < n && grid[i][j+1] == 1 {
			count += dfs(i, j+1)
		}
		return count
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				count := dfs(i, j)
				ans = max(ans, count)
			}
		}
	}
	return ans
}
