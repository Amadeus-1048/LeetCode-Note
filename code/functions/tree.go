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

// 404. 左叶子之和
func sumOfLeftLeaves(root *TreeNode) int {
	res := 0
	var findLeft func(node *TreeNode)
	findLeft = func(node *TreeNode) {
		if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
			res += node.Left.Val
		}
		if node.Left != nil {
			findLeft(node.Left)
		}
		if node.Right != nil {
			findLeft(node.Right)
		}
	}
	findLeft(root)
	return res
}

// 513. 找树左下角的值
func findBottomLeftValue(root *TreeNode) int {
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
			if i == 0 { // 记录每一行第一个元素
				ans = node.Val
			}
		}
	}
	return ans
}

// 112. 路径总和
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	targetSum -= root.Val
	if root.Left == nil && root.Right == nil && targetSum == 0 { // 遇到叶子节点，并且计数为0
		return true
	}
	return hasPathSum(root.Left, targetSum) || hasPathSum(root.Right, targetSum)
}

// 113. 路径总和 II
func pathSum(root *TreeNode, targetSum int) [][]int {
	res := make([][]int, 0)
	curPath := make([]int, 0)
	var traverse func(node *TreeNode, targetSum int)
	traverse = func(node *TreeNode, targetSum int) {
		if node == nil {
			return
		}
		targetSum -= node.Val                                        // 将targetSum在遍历每层的时候都减去本层节点的值
		curPath = append(curPath, node.Val)                          // 把当前节点放到路径记录里
		if node.Left == nil && node.Right == nil && targetSum == 0 { // 遇到叶子节点，并且计数为0
			// 不能直接将currPath放到result里面, 因为currPath是共享的, 每次遍历子树时都会被修改
			pathCopy := make([]int, len(curPath))
			copy(pathCopy, curPath)
			res = append(res, pathCopy) // 将副本放到结果集里
		}
		traverse(node.Left, targetSum)
		traverse(node.Right, targetSum)
		curPath = curPath[:len(curPath)-1] // 当前节点遍历完成, 从路径记录里删除掉
	}
	traverse(root, targetSum)
	return res
}

// 106. 从中序与后序遍历序列构造二叉树
func buildTree(inorder []int, postorder []int) *TreeNode {
	if len(inorder) < 1 || len(postorder) < 1 {
		return nil
	}
	// 先找到根节点（后续遍历的最后一个就是根节点）
	nodeValue := postorder[len(postorder)-1]
	// 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
	left := findRootIndex(inorder, nodeValue)
	// 构造root
	root := &TreeNode{
		Val:   nodeValue,
		Left:  buildTree(inorder[:left], postorder[:left]), // 一棵树的中序遍历和后序遍历的长度相等
		Right: buildTree(inorder[left+1:], postorder[left:len(postorder)-1]),
	}
	return root
}

func findRootIndex(inorder []int, target int) (index int) {
	for i := 0; i < len(inorder); i++ {
		if target == inorder[i] {
			return i
		}
	}
	return -1
}

// 105. 从前序与中序遍历序列构造二叉树
func buildTree2(preorder []int, inorder []int) *TreeNode {
	if len(preorder) < 1 || len(inorder) < 1 {
		return nil
	}
	// 先找到根节点（先序遍历的第一个就是根节点）
	// 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
	left := findRootIndex(inorder, preorder[0])
	// 构造root
	root := &TreeNode{
		Val:   preorder[0],
		Left:  buildTree2(preorder[1:left+1], inorder[:left]), // 将先序遍历一分为二，左边为左子树，右边为右子树
		Right: buildTree2(preorder[left+1:], inorder[left+1:]),
	}
	return root
}

// 654. 最大二叉树
func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) < 1 {
		return nil
	}
	// 找到最大值
	index := findMax(nums)
	// 构造二叉树
	root := &TreeNode{
		Val:   nums[index],
		Left:  constructMaximumBinaryTree(nums[:index]),
		Right: constructMaximumBinaryTree(nums[index+1:]),
	}
	return root
}

func findMax(nums []int) (index int) {
	for i := 0; i < len(nums); i++ {
		if nums[i] > nums[index] {
			index = i
		}
	}
	return index
}

// 617. 合并二叉树
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
	if root1 == nil {
		return root2
	}
	if root2 == nil {
		return root1
	}
	root1.Val += root2.Val
	root1.Left = mergeTrees(root1.Left, root2.Left)
	root1.Right = mergeTrees(root1.Right, root2.Right)
	return root1
}

// 700. 二叉搜索树中的搜索
func searchBST(root *TreeNode, val int) *TreeNode {
	for root != nil {
		if root.Val > val {
			root = root.Left
		} else if root.Val < val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}

// 98. 验证二叉搜索树
func isValidBST(root *TreeNode) bool {
	var pre *TreeNode // 用来记录前一个节点
	var check func(node *TreeNode) bool
	check = func(node *TreeNode) bool {
		if node == nil {
			return true
		}
		// 中序遍历，验证遍历的元素是不是从小到大
		left := check(node.Left)
		if pre != nil && pre.Val >= node.Val {
			return false
		}
		pre = node
		right := check(node.Right)

		return left && right // 分别对左子树和右子树递归判断，如果左子树和右子树都符合则返回true
	}
	return check(root)
}

// 530. 二叉搜索树的最小绝对差
func getMinimumDifference(root *TreeNode) int {
	var pre *TreeNode // 用来记录前一个节点
	minDelta := math.MaxInt64
	var calc func(node *TreeNode)
	calc = func(node *TreeNode) {
		if node == nil {
			return
		}
		// 中序遍历
		calc(node.Left)
		if pre != nil {
			minDelta = min(minDelta, node.Val-pre.Val)
		}
		pre = node
		calc(node.Right)
	}
	calc(root)
	return minDelta
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 501.二叉搜索树中的众数
func findMode(root *TreeNode) []int {
	res := make([]int, 0)
	count := 1
	max := 1
	var prev *TreeNode
	var travel func(node *TreeNode)
	travel = func(node *TreeNode) { // 中序遍历
		if node == nil {
			return
		}
		travel(node.Left)
		if prev != nil && prev.Val == node.Val { // 遇到相同的值，计数+1
			count++
		} else {
			count = 1 // 遇到新的值，重新开始计数
		}
		if count >= max {
			if count > max && len(res) > 0 { // 遇到出现次数更多的值，重置res
				res = []int{node.Val}
			} else {
				res = append(res, node.Val) // 遇到出现次数相同的值，res多加一个值
			}
			max = count
		}
		prev = node
		travel(node.Right)
	}
	travel(root)
	return res
}

// 236. 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	parent := map[int]*TreeNode{} // 题目给出：所有 Node.val 互不相同，所以可以用int存储
	visited := map[int]bool{}
	var dfs func(node *TreeNode)
	// 从根节点开始遍历整棵二叉树，用哈希表记录每个节点的父节点指针
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil {
			parent[node.Left.Val] = node
			dfs(node.Left)
		}
		if node.Right != nil {
			parent[node.Right.Val] = node
			dfs(node.Right)
		}
	}
	dfs(root)
	// 从 p 节点开始不断往它的祖先移动，并记录已经访问过的祖先节点
	for p != nil {
		visited[p.Val] = true
		p = parent[p.Val]
	}
	// 再从 q 节点开始不断往它的祖先移动，如果有祖先已经被访问过，即意味着这是 p 和 q 的深度最深的公共祖先，即 LCA 节点
	for q != nil {
		if visited[q.Val] {
			return q
		}
		q = parent[q.Val]
	}
	return nil
}

// 235. 二叉搜索树的最近公共祖先
func lowestCommonAncestor2(root, p, q *TreeNode) *TreeNode {
	// 如果找到了 节点p或者q，或者遇到空节点，就返回。
	if root == nil {
		return nil
	}
	if root.Val >= p.Val && root.Val <= q.Val { // 当前节点的值在给定值的中间（或者等于），即为最深的祖先
		return root
	}
	if root.Val > p.Val && root.Val > q.Val { // 当前节点的值大于给定的值，则说明满足条件的在左边
		return lowestCommonAncestor(root.Left, p, q)
	}
	if root.Val < p.Val && root.Val < q.Val { // 当前节点的值小于各点的值，则说明满足条件的在右边
		return lowestCommonAncestor(root.Right, p, q)
	}
	return root
}

// 701. 二叉搜索树中的插入操作
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	// 找到遍历的节点为null的时候，就是要插入节点的位置了，并把插入的节点返回
	if root == nil {
		root = &TreeNode{Val: val}
		return root
	}
	if root.Val > val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}

// 450.删除二叉搜索树中的节点
func deleteNode(root *TreeNode, key int) *TreeNode {
	switch {
	// 没找到删除的节点，遍历到空节点直接返回了
	case root == nil:
		return nil
	// 说明要删除的节点在左子树	左递归
	case root.Val > key:
		root.Left = deleteNode(root.Left, key)
	// 说明要删除的节点在右子树	右递归
	case root.Val < key:
		root.Right = deleteNode(root.Right, key)
	// 找到了节点，且左儿子或右儿子有空的
	case root.Left == nil || root.Right == nil:
		// 右儿子为空，删除节点后左儿子补位
		if root.Left != nil {
			return root.Left
		}
		// 左儿子为空，删除节点后右儿子补位
		return root.Right
	// 找到了节点，且左右儿子都不为空
	// 则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
	// 并返回删除节点右孩子为新的根节点
	default:
		successor := root.Right
		for successor.Left != nil {
			successor = successor.Left
		}
		// 删除节点的右子树的最左面节点变为新的根节点
		// 所以新的根节点的右子树应该是删除节点的右子树去掉该新根节点
		// 新的根节点的左子树应该是删除节点的左子树
		successor.Right = deleteNode(root.Right, successor.Val)
		successor.Left = root.Left
		return successor
	}
	return root
}
