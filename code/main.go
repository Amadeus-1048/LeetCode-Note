package main

import (
	"fmt"
)

// 哈希函数
func hash(r []bool, x []bool) []bool {
	// 执行题目中的哈希操作
	// 使用 != 作为XOR操作
	y0 := x[0] != r[1] != x[3] != r[0] != r[1]
	y1 := x[2] != x[4] != r[3] != r[2] != x[1]
	y2 := true != x[1] != x[0] != x[4] != false
	y3 := x[4] != true != x[2] != false != x[3]
	return []bool{y0, y1, y2, y3}
}

// 检查哈希值是否等于目标
func checkHash(r []bool, x []bool, target []bool) bool {
	result := hash(r, x)
	for i := range result {
		if result[i] != target[i] {
			return false
		}
	}
	return true
}

func main() {
	// puzzle-ID为1010, 转换为bool数组
	puzzleID := []bool{true, false, true, false}
	// 目标集合Y为{0101}, 同样转换为bool数组
	target := []bool{false, true, false, true}

	// 搜索所有6位输入x的组合
	for i := 0; i < 64; i++ {
		// 生成6位输入x
		x := make([]bool, 6)
		for j := 0; j < 6; j++ {
			// 第j位是1还是0
			x[j] = (i & (1 << j)) > 0
		}

		// 检查当前组合的哈希值是否等于目标
		if checkHash(puzzleID, x, target) {
			// 找到了一组解，打印出来
			fmt.Print("找到解：x = ")
			for _, bit := range x {
				if bit {
					fmt.Print("1")
				} else {
					fmt.Print("0")
				}
			}
			fmt.Println()
		}
	}
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func getMostGold(nodes *TreeNode) []int {
	res := make([]int, 0)
	maxSum := 0
	tmpSum := 0
	tmpSlice := make([]int, 0)
	// 回溯
	var backtrace func(node *TreeNode)
	backtrace = func(node *TreeNode) {
		tmpSum += node.Val
		tmpSlice = append(tmpSlice, node.Val)
		// 找到叶子节点
		if node.Left == nil && node.Right == nil {
			if tmpSum > maxSum {
				fmt.Println("res before copy = ", res)
				fmt.Println("tmpSlice = ", tmpSlice)
				maxSum = tmpSum
				res = make([]int, len(tmpSlice)) // 目标切片必须分配过空间且足够承载复制的元素个数
				copy(res, tmpSlice)
				fmt.Println("res after copy = ", res)
			}
			return
		}
		// 没有找到叶子节点
		if node.Left != nil {
			backtrace(node.Left)
			tmpSlice = tmpSlice[:len(tmpSlice)-1]
			tmpSum -= node.Left.Val
		}
		if node.Right != nil {
			backtrace(node.Right)
			tmpSlice = tmpSlice[:len(tmpSlice)-1]
			tmpSum -= node.Right.Val
		}

	}

	backtrace(nodes)
	return res
}
