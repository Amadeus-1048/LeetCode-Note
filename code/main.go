package main

import (
	"fmt"
	"math"
)

const INF = 100000

func main() {
	// 带权值邻接矩阵
	var gp = [][]int{
		// a b c d e f g s
		{0, 1, INF, INF, 5, INF, INF, 7},   // a
		{1, 0, INF, 1, 3, INF, INF, INF},   // b
		{2, INF, 0, 1, 1, 9, 6, INF},       // c
		{INF, 1, INF, 0, 1, INF, 9, INF},   // d
		{5, 3, 1, INF, 0, INF, INF, INF},   // e
		{INF, INF, 9, INF, INF, 0, 3, 3},   // f
		{INF, INF, 6, 9, INF, 3, 0, INF},   // g
		{7, INF, INF, INF, INF, 3, INF, 0}, // s
	}
	dist := dijkstra(gp, 1)
	fmt.Println(dist)
}

func dijkstra(graph [][]int, start int) []int {
	n := len(graph)         // 图中顶点个数
	visit := make([]int, n) // 标记已经作为中间结点完成访问的顶点
	dist := make([]int, n)  // 存储从起点到其他顶点的最短路径

	for i := 0; i < n; i++ {
		dist[i] = graph[start][i] // 初始化遍历起点
	}
	visit[start] = 1 // 标记初始顶点

	// 更新其他顶点最短路径，循环n次
	for i := 0; i < n; i++ {
		minDist := math.MaxInt // 存储从起点到其他未被访问的结点中的最短路径
		midNode := 0           // 中间结点

		// 遍历n个顶点，寻找未被访问且起始位置到该点距离最小的顶点
		for j := 0; j < n; j++ {
			if visit[j] == 0 && minDist > dist[j] {
				minDist = dist[j] // 更新未被访问结点的最短路径
				midNode = j       // 更新中间结点
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
