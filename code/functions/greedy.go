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
