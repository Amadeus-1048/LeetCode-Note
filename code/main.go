package main

import (
	"fmt"
	"strings"
)

func main() {
	s := "/1/2"
	a := strings.Split(s, "/")
	fmt.Println("length is :", len(a))
	for _, ss := range a {
		fmt.Println(ss)
	}
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func mySqrt(x int) int {
	res := 0 // x 平方根的整数部分 ans 是满足 k^2 ≤x 的最大 k 值
	sqrt := 0
	left, right := 0, x
	if x <= 1 {
		return x
	} else {
		for left <= right { // 对 k 进行二分查找
			sqrt = left + (right-left)/2
			if sqrt*sqrt <= x { // 比较中间元素的平方与 x 的大小关系
				res = sqrt
				left = sqrt + 1
			} else {
				right = sqrt - 1
			}
		}
	}
	return res
}
