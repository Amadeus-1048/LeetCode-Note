package functions

import (
	"math"
	"strconv"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 144. 二叉树的前序遍历
func preorderTraversal(root *TreeNode) (res []int) {
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		res = append(res, node.Val)
		traversal(node.Left)
		traversal(node.Right)
	}
	traversal(root)
	return res
}

// 145. 二叉树的后序遍历
func postorderTraversal(root *TreeNode) (res []int) {
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		traversal(node.Left)
		traversal(node.Right)
		res = append(res, node.Val)
	}
	traversal(root)
	return res
}

// 94. 二叉树的中序遍历
func inorderTraversal(root *TreeNode) (res []int) {
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		traversal(node.Left)
		res = append(res, node.Val)
		traversal(node.Right)
	}
	traversal(root)
	return res
}

// 102. 二叉树的层序遍历
func levelOrder(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		tmp := []int{}
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			tmp = append(tmp, node.Val) //将值加入本层切片中
		}
		ans = append(ans, tmp) //放入结果集
	}
	return ans
}

// 107. 二叉树的层序遍历 II
func levelOrderBottom(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		tmp := []int{}
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			tmp = append(tmp, node.Val) //将值加入本层切片中
		}
		ans = append(ans, tmp) //放入结果集
	}
	for i := 0; i < len(ans)/2; i++ {
		ans[i], ans[len(ans)-i-1] = ans[len(ans)-i-1], ans[i]
	}
	return ans
}

// 199. 二叉树的右视图
func rightSideView(root *TreeNode) []int {
	ans := []int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		tmp := []int{}
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			tmp = append(tmp, node.Val) //将值加入本层切片中
		}
		ans = append(ans, tmp[len(tmp)-1]) //放入结果集
	}
	return ans
}

// 637. 二叉树的层平均值
func averageOfLevels(root *TreeNode) []float64 {
	ans := []float64{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		sum := 0
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			sum += node.Val
		}
		ans = append(ans, float64(sum)/float64(length)) //放入结果集
	}
	return ans
}

// 429. N 叉树的层序遍历
type Node struct {
	Val      int
	Children []*Node
}

func NlevelOrder(root *Node) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*Node{root}
	for len(queue) > 0 {
		tmp := []int{}
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			for j := 0; j < len(node.Children); j++ {
				queue = append(queue, node.Children[j])
			}
			tmp = append(tmp, node.Val) //将值加入本层切片中
		}
		ans = append(ans, tmp) //放入结果集
	}
	return ans
}

// 515.在每个树行中找最大值
func largestValues(root *TreeNode) []int {
	ans := []int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		maxNumber := int(math.Inf(-1)) //负无穷   因为节点的值会有负数
		length := len(queue)           //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			maxNumber = max(maxNumber, node.Val)
		}
		ans = append(ans, maxNumber) //放入结果集
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 116. 填充每个节点的下一个右侧节点指针
type PerfectNode struct {
	Val   int
	Left  *PerfectNode
	Right *PerfectNode
	Next  *PerfectNode
}

func connect(root *PerfectNode) *PerfectNode {
	if root == nil {
		return root
	}
	queue := []*PerfectNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		pre := queue[0]      // 不用担心越界，因为上面的for循环已经判断过了
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			if i > 0 {
				pre.Next = node
				pre = node
			}
		}
	}
	return root
}

// 117. 填充每个节点的下一个右侧节点指针 II
func connect2(root *PerfectNode) *PerfectNode {
	if root == nil {
		return root
	}
	queue := []*PerfectNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		pre := queue[0]      // 不用担心越界，因为上面的for循环已经判断过了
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			if i > 0 {
				pre.Next = node
				pre = node
			}
		}
	}
	return root
}

// 104. 二叉树的最大深度
func maxDepth(root *TreeNode) int {
	ans := 0
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans++
	}
	return ans
}

// 111. 二叉树的最小深度
func minDepth(root *TreeNode) int {
	ans := 0
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left == nil && node.Right == nil { // 当前节点没有左右节点，则代表此层是最小层
				return ans + 1 // 返回当前层 ans代表是上一层
			}
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans++
	}
	return ans
}

// 226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			node.Left, node.Right = node.Right, node.Left
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	return root
}

// 101. 对称二叉树
func isSymmetric(root *TreeNode) bool {
	var queue []*TreeNode
	if root != nil {
		queue = append(queue, root.Left, root.Right)
	}
	for len(queue) > 0 {
		left := queue[0]  // 将左子树头结点加入队列
		right := queue[1] // 将右子树头结点加入队列
		queue = queue[2:]
		if left == nil && right == nil { // 左节点为空、右节点为空，此时说明是对称的
			continue
		}
		// 左右一个节点不为空，或者都不为空但数值不相同，返回false
		if left == nil || right == nil || left.Val != right.Val {
			return false
		}
		// 依次加入：左节点左孩子、右节点右孩子、左节点右孩子、右节点左孩子
		queue = append(queue, left.Left, right.Right, left.Right, right.Left)
	}
	return true
}

// 222. 完全二叉树的节点个数
func countNodes(root *TreeNode) int {
	ans := 0
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue) //保存当前层的长度，然后处理当前层
		for i := 0; i < length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			ans++
		}
	}
	return ans
}

// 110. 平衡二叉树
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if !isBalanced(root.Left) || !isBalanced(root.Right) {
		return false
	}
	// 分别求出其左右子树的高度
	LeftH := maxHeight(root.Left) + 1 // 以当前节点为根节点的树的最大高度
	RightH := maxHeight(root.Right) + 1
	// 如果差值大于1，表示已经不是二叉平衡树
	if abs(LeftH-RightH) > 1 {
		return false
	}
	return true
}

func maxHeight(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(maxHeight(root.Left), maxHeight(root.Right)) + 1
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// 257. 二叉树的所有路径
func binaryTreePaths(root *TreeNode) []string {
	res := make([]string, 0)
	var travel func(node *TreeNode, s string)
	travel = func(node *TreeNode, s string) {
		if node.Left == nil && node.Right == nil { // 找到了叶子节点
			v := s + strconv.Itoa(node.Val) // 找到一条路径
			res = append(res, v)            // 添加到结果集
			return
		}
		s += strconv.Itoa(node.Val) + "->" // 非叶子节点，后面多加符号
		if node.Left != nil {
			travel(node.Left, s)
		}
		if node.Right != nil {
			travel(node.Right, s)
		}
	}
	travel(root, "")
	return res
}
