# 数组

## 704.二分查找

答案

```go
func search(nums []int, target int) int {
	left := 0
	right := len(nums)
	for left < right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid // 因为是左闭右开区间，所以这里不能是high = mid - 1，mid-1有可能是答案
		}
	}
	return -1
}
```



分析

```go
数组为有序数组，同时题目还强调数组中无重复元素，则可以用二分法

第一种二分法：[left, right]
for left <= right，因为left在区间成立时可能等于right
取左半边时，right = mid-1， 因为已经确定了nums[mid]不会是target

第二种二分法：[left, right)
for left < right，因为left在区间成立时不可能等于right
取左半边时，right = mid， 因为已经确定了nums[mid]不会是target，又因为右区间是开区间，所以不会取到mid
```



## 27.移除元素

答案

```go
func removeElement(nums []int, val int) int {
	length := len(nums)
	slow, fast := 0, 0
	for fast < length {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
			fast++
		} else {
			fast++
		}
	}
	return slow
}
```



分析

```go
双指针法（快慢指针法）： 通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。

定义快慢指针：
	快指针：寻找新数组的元素 ，新数组就是不含有目标元素的数组
	慢指针：指向更新 新数组下标的位置
```



## 977.有序数组的平方

答案

```go
func sortedSquares(nums []int) []int {
	n := len(nums)
	i, j, k := 0, n-1, n-1
	ans := make([]int, n)
	for i <= j {
		left, right := nums[i]*nums[i], nums[j]*nums[j]
		if left < right {
			ans[k] = right
			j--
		} else {
			ans[k] = left
			i++
		}
		k--
	}
	return ans
}
```



分析

```go
数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。

此时可以考虑双指针法了，i指向起始位置，j指向终止位置

定义一个新数组result，和A数组一样的大小，让k指向result数组终止位置
```





## 209. 长度最小的子数组

答案

```go
func minSubArrayLen(target int, nums []int) int {
	i, sum := 0, 0
	length := len(nums)
	res := length + 1
	for j := 0; j < length; j++ {
		sum += nums[j]
		for sum >= target {
			tmp := j - i + 1
			if tmp < res {
				res = tmp
			}
			sum -= nums[i]
			i++
		}
	}
	if res == length+1 {
		return 0
	}
	return res
}
```



分析

```go
滑动窗口 : 不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果

在本题中实现滑动窗口，主要确定如下三点：
窗口内是什么？
如何移动窗口的起始位置？
如何移动窗口的结束位置？

只用一个for循环，那么这个循环的索引，一定是表示滑动窗口的终止位置
```





# 链表

## 203 移除链表元素

答案

```go
func removeElements(head *ListNode, val int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	cur := dummy
	for cur.Next!=nil {
		if cur.Next.Val == val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}
```



分析

```go
在单链表中移除头结点 和 移除其他节点的操作方式是不一样的
设置一个虚拟头结点，这样原链表的所有节点就都可以按照统一的方式进行移除了
```



## 707.设计链表

答案

```go
type SingleNode struct {
	Val  int         // 节点的值
	Next *SingleNode // 下一个节点的指针
}

type MyLinkedList struct {
	dummyHead *SingleNode // 虚拟头节点
	Size      int         // 链表大小
}

func Constructor() MyLinkedList {
	newNode := &SingleNode{ // 创建新节点
		Next: nil,
	}
	return MyLinkedList{ // 返回链表
		dummyHead: newNode,
		Size:      0,
	}

}

func (this *MyLinkedList) Get(index int) int {
	if this == nil || index < 0 || index >= this.Size { // 如果索引无效则返回-1
		return -1
	}
	cur := this.dummyHead.Next   // 设置当前节点为真实头节点
	for i := 0; i < index; i++ { // 遍历到索引所在的节点
		cur = cur.Next
	}
	return cur.Val // 返回节点值
}

func (this *MyLinkedList) AddAtHead(val int) {
	newNode := &SingleNode{Val: val}   // 创建新节点
	newNode.Next = this.dummyHead.Next // 新节点指向当前头节点
	this.dummyHead.Next = newNode      // 新节点变为头节点
	this.Size++                        // 链表大小增加1
}

func (this *MyLinkedList) AddAtTail(val int) {
	newNode := &SingleNode{Val: val} // 创建新节点
	cur := this.dummyHead            // 设置当前节点为虚拟头节点
	for cur.Next != nil {            // 遍历到最后一个节点
		cur = cur.Next
	}
	cur.Next = newNode // 在尾部添加新节点
	this.Size++        // 链表大小增加1
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
	if index < 0 { // 如果索引小于0，设置为0
		index = 0
	} else if index > this.Size { // 如果索引大于链表长度，直接返回
		return
	}

	newNode := &SingleNode{Val: val} // 创建新节点
	cur := this.dummyHead            // 设置当前节点为虚拟头节点
	for i := 0; i < index; i++ {     // 遍历到指定索引的前一个节点
		cur = cur.Next
	}
	newNode.Next = cur.Next // 新节点指向原索引节点
	cur.Next = newNode      // 原索引的前一个节点指向新节点
	this.Size++             // 链表大小增加1
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
	if index < 0 || index >= this.Size { // 如果索引无效则直接返回
		return
	}
	cur := this.dummyHead        // 设置当前节点为虚拟头节点
	for i := 0; i < index; i++ { // 遍历到要删除节点的前一个节点
		cur = cur.Next
	}
	if cur.Next != nil {
		cur.Next = cur.Next.Next // 当前节点直接指向下下个节点，即删除了下一个节点
	}
	this.Size-- // 注意删除节点后应将链表大小减一
}
```



分析

```go

```





## 206 反转链表

答案

```go
func reverseList(head *ListNode) *ListNode {
	cur := head
	var pre *ListNode	// 不能用pre := &ListNode{}    输出结果会在最后多一个0
	for cur != nil {
		tmp := cur.Next
		cur.Next = pre
		pre = cur
		cur = tmp
	}
	return pre	// 是返回pre，不是cur，因为最后cur是nil
}
```



分析

```go
var pre *ListNode 和 pre := &ListNode{} 是不同的

p1 := &ListNode{}
fmt.Println("p1:", p1==nil)		// p1: false  p1的值为：{0, <nil>}，这就是上面为什么会多一个0的原因
var p2 *ListNode
fmt.Println("p2:", p2==nil)		// p2: true
```



## 92. 反转链表 II

答案

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	pre := dummy
	for i:=0; i<left-1; i++ {
		pre = pre.Next
	}
	cur := pre.Next
	// 在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置
	// cur：指向待反转区域的第一个节点 left  cur的值不变，但是位置会往后移动
	// next：永远指向 cur 的下一个节点，循环过程中，cur 变化以后 next 的值和位置都会变化
	// pre：永远指向待反转区域的第一个节点 left 的前一个节点，在循环过程中值和位置都不变

	for i:=left; i<right; i++ {
		next := cur.Next
		cur.Next = next.Next
		next.Next = pre.Next	// 不能是 cur 因为必须是在反转部分的起始位置，即pre的下一个，cur会慢慢往后移动的
		pre.Next = next
	}
	return dummy.Next
}
```



分析

```go
画图！
pre  的值和位置都不变
cur  的值不变，但是位置会往后移动
next 的值和位置都会变化
```







## 24. 两两交换链表中的节点

答案

```go
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	pre := dummy
	for head != nil && head.Next != nil {
		pre.Next = head.Next
		next := head.Next.Next
		head.Next.Next = head
		head.Next = next
		pre = head
		head = next
	}
	return dummy.Next
}
```



分析

```go
还是要画一下图

head.Next.Next = head	对结点指向哪里进行修改
pre = head				用变量pre表示head结点（即pre和head同时表示一个结点）
```





## 25. K 个一组翻转链表

答案

```go
func reverseKGroup(head *ListNode, k int) *ListNode {
	cur := head
	for i := 0; i < k; i++ {
		if cur == nil {
			return head
		}
		cur = cur.Next
	}
	newHead := reverse(head, cur)		// 从 head 开始进行反转长度为 k 的链表
	head.Next = reverseKGroup(cur, k)	// 递归处理下一个长度为 k 的链表
	return newHead
}

func reverse(head, tail *ListNode) *ListNode {
	var pre *ListNode
	cur := head
	for cur != tail {		// 从 head 到 tail 的前面一个结点进行遍历
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}
```



分析

https://leetcode.cn/problems/reverse-nodes-in-k-group/solution/jian-dan-yi-dong-go-by-dfzhou6-4cha/

```go
采用递归，这样比较易于理解
```





## 141. 环形链表

答案

```go
// 定义快慢指针 slow、fast，初始指向 head。
// 快指针每次走两步，慢指针每次走一步，不断循环。
// 当相遇时，说明链表存在环。如果循环结束依然没有相遇，说明链表不存在环。
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast!=nil && fast.Next!=nil {
		slow = slow.Next
		fast = fast.Next
		if slow == fast {
			return true
		}
	}
	return false
}
```



分析

```go
for循环的条件为：for fast!=nil && fast.Next!=nil 
原因是如果fast是链表尾结点，即fast.Next=nil，此时fast.Next.Next是非法的，会报错
因此要判断fast和fast.Next是否为nil
如果fast.Next不是nil，那么fast.Next.Next是合法的（虽然可能等于nil）
```





## 142. 环形链表 II

答案

```go
func detectCycle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next   // 任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍
        if slow == fast {   // 找到重合的节点，说明在环中
        // 当 slow 与 fast 相遇时，head指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇
            for slow != head {
                slow = slow.Next
                head = head.Next
            }
            return head
        }
    }
    return nil
}
```



分析

<img src="https://assets.leetcode-cn.com/solution-static/142/142_fig1.png" alt="fig1" style="zoom: 25%;" />

```go
设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与 fast 相遇。
此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc。

根据题意，任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍。因此，我们有 a+(n+1)b+nc = 2(a+b)  ⟹  a = c+(n−1)(b+c)

有了 a=c+(n-1)(b+c) 的等量关系，我们会发现：从相遇点到入环点的距离加上 n-1 圈的环长，恰好等于从链表头部到入环点的距离。

因此，当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇。
```











## 23.合并K个升序链表

答案

```go
func mergeKLists(lists []*ListNode) *ListNode {
    n := len(lists)
    if n == 0 {
        return nil
    }
    if n == 1 { // 返回结果
        return lists[0]
    }
    if n % 2 == 1 { // K为奇数，那么先合并最后两个列表，将其变为偶数长度的列表
        lists[n-2] = mergeTwoLists(lists[n-2], lists[n-1])
        lists, n = lists[:n-1], n-1
    }
    mid := n / 2
    for i:=0; i<mid; i++{   // 后半部分合并到前半部分。
        lists[i] = mergeTwoLists(lists[i], lists[i+mid])
    }
    return mergeKLists(lists[:mid])
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }

    if l1.Val < l2.Val {
        l1.Next = mergeTwoLists(l1.Next, l2)
        return l1
    }else{
        l2.Next = mergeTwoLists(l1, l2.Next)
        return l2
    }
}

```



分析

```go
归并法比暴力法要快非常多
暴力法：不断的把短链表合并到唯一的长链表中
归并法：将K个有序链表转换为多个合并两个有序链表的问题
```



# 树

## 二叉树递归遍历

答案

前序遍历:

```go
func preorderTraversal(root *TreeNode) (res []int) {
    var traversal func(node *TreeNode)
    traversal = func(node *TreeNode) {
	if node == nil {
            return
	}
	res = append(res,node.Val)
	traversal(node.Left)
	traversal(node.Right)
    }
    traversal(root)
    return res
}

```



中序遍历:

```go
func inorderTraversal(root *TreeNode) (res []int) {
    var traversal func(node *TreeNode)
    traversal = func(node *TreeNode) {
	if node == nil {
	    return
	}
	traversal(node.Left)
	res = append(res,node.Val)
	traversal(node.Right)
    }
    traversal(root)
    return res
}
```


后序遍历:

```go
func postorderTraversal(root *TreeNode) (res []int) {
    var traversal func(node *TreeNode)
    traversal = func(node *TreeNode) {
	if node == nil {
	    return
	}
	traversal(node.Left)
	traversal(node.Right)
        res = append(res,node.Val)
    }
    traversal(root)
    return res
}
```





## 637. 二叉树的层平均值

答案

```go
func averageOfLevels(root *TreeNode) []float64 {
	ans := []float64{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	//tmp := []int{}
	sum := 0
	for len(queue)>0 {
		length := len(queue)	//保存当前层的长度，然后处理当前层
		for i:=0; i<length; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			//tmp = append(tmp, node.Val)	//将值加入本层切片中
			sum += node.Val
		}
		ans = append(ans, float64(sum)/float64(length))	//放入结果集
		//tmp = []int{}	//清空层的数据
		sum = 0
	}
	return ans
}
```



分析

```go
求平均数，结果用float64表示
被除数和除数都得先强制转换为float64类型才可以，否则会报错

ans = append(ans, float64(sum)/float64(length))
```



## 103. 二叉树的锯齿形层序遍历

答案

```go
func zigzagLevelOrder(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {	// 漏掉了
		return ans
	}
	queue := []*TreeNode{root}
	length := len(queue)
	for level:=0; len(queue)>0; level++ {		// 是len(queue)>0   不是 queue != nil，会报错
		tmp := []int{}
		length = len(queue)
		for length > 0 {
			tmp = append(tmp, queue[0].Val)
			if queue[0].Left != nil {		// append之前要先判断是不是nil
				queue = append(queue, queue[0].Left)
			}
			if queue[0].Right != nil {		// append之前要先判断是不是nil
				queue = append(queue, queue[0].Right)
			}
			queue = queue[1:]
			length--	// 漏掉了
		}
		if level % 2 == 1 {			// 用level还可以计算层数，所以不用bool
			i, j := 0, len(tmp)-1
			for i < j {
				tmp[i], tmp[j] = tmp[j], tmp[i]
				i++
				j--
			}
		}
		ans = append(ans, tmp)
	}
	return ans
}
```



分析

```go
层序遍历
```



## 236. 二叉树的最近公共祖先

答案

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	parent := map[int]*TreeNode{}	// 题目给出：所有 Node.val 互不相同，所以可以用int存储
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
```



分析

```go
递归比较难理解，这个方法容易理解
```





# 回溯

## 39. 组合总和

答案

```go
func combinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	trace := []int{}
	var backtrace func(start, sum int)
	backtrace = func(start, sum int) {
		if sum == target {
			tmp := make([]int, len(trace))	// 注意这里  测试案例输出：[[2,2,3],[7]]
			copy(tmp, trace)				// 必须创建一个用来拷贝的，使用copy函数
			res = append(res, tmp)
			return
		}
		if sum > target {
			return
		}

		for i:=start; i<len(candidates); i++ {
			trace = append(trace, candidates[i])
			sum += candidates[i]
			backtrace(i, sum)
			trace = trace[:len(trace)-1]
			sum -= candidates[i]
		}
	}
	backtrace(0, 0)
	return res
}
```



分析

```go
----------------------------------------------------------------------
func combinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	trace := []int{}
	var backtrace func(start, sum int)
	backtrace = func(start, sum int) {
		if sum == target {	
			res = append(res, trace)	// 测试案例输出：[[7,7,7],[7]]
			return
		}
----------------------------------------------------------------------
```



## 46. 全排列

答案

```go
func permute(nums []int) [][]int {
	res := [][]int{}
	trace := []int{}
	used := [21]int{}
	var backtrace func()
	backtrace = func() {
		if len(trace) == len(nums) {
			tmp := make([]int, len(trace))
			copy(tmp, trace)
			res = append(res, tmp)	// 如果这里是res = append(res, trace)，则res里的每个值会随着trace的变化而变化
			return
		}
		for i:=0; i<len(nums); i++ {
			if used[nums[i]+10] == 0 {		// 因为 nums[i] 为 -10 ~ 10
				trace = append(trace, nums[i])
				used[nums[i]+10] = 1
				backtrace()
				trace = trace[:len(trace)-1]	// 回溯时要消除之前的影响
				used[nums[i]+10] = 0
			}
		}
	}
	backtrace()
	return res
}
```



分析

```go

```







## 47. 全排列 II

答案

```go
func permuteUnique(nums []int) [][]int {
	res := [][]int{}
	trace := []int{}
	used := [10]int{}
	sort.Ints(nums)		// 目的是为了同一树层去重
	var backtrace func()
	backtrace = func() {
		if len(trace) == len(nums) {
			tmp := make([]int, len(trace))
			copy(tmp, trace)
			res = append(res, tmp)
			return
		}
		for i:=0; i<len(nums); i++ {
			if i>0 && nums[i]==nums[i-1] && used[i-1]==0 {	// used=0，表示已经切换到新的树枝了
				continue
			}
			if used[i] == 0 {	// 目的是为了同一树枝去重
				used[i] = 1
				trace = append(trace, nums[i])
				backtrace()
				trace = trace[:len(trace)-1]
				used[i] = 0
			}
		}
	}
	backtrace()
	return res
}
```



分析

```go
注意，换到另一树枝上时，used数组其实会清空（父节点以上不清空）
所以 i>0 && nums[i]==nums[i-1] && used[i-1]==0 表明切换到另一条树枝了，前一条树枝已经用了i-1和i
```



## 51. N 皇后

答案

```go
func solveNQueens(n int) [][]string {
	res := [][]string{}
	chessboard := make([][]string, n)
	for i := 0; i < n; i++ {
		chessboard[i] = make([]string, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			chessboard[i][j] = "."
		}
	}

	var backtrace func(row int)
	backtrace = func(row int) {
		if row == n {
			tmp := make([]string, n)
			for i, rowStr := range chessboard {
				tmp[i] = strings.Join(rowStr, "")	// 将rowStr中的子串连接成一个单独的字符串，子串之间用""分隔
			}
			res = append(res, tmp)
			return
		}
		for i:=0; i<n; i++ {
			if isValid(n, row, i, chessboard) {
				chessboard[row][i] = "Q"
				backtrace(row+1)
				chessboard[row][i] = "."
			}
		}
	}
	backtrace(0)
	return res
}

func isValid(n, row, col int, chessboard [][]string) bool {
	for i := 0; i < row; i++ {
		if chessboard[i][col] == "Q" {
			return false
		}
	}
	for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
		if chessboard[i][j] == "Q" {
			return false
		}
	}
	for i, j := row-1, col+1; i >= 0 && j < n; i, j = i-1, j+1 {
		if chessboard[i][j] == "Q" {
			return false
		}
	}
	return true
}
```



分析

https://phpmianshi.com/?id=1947

```go
tmp[i] = strings.Join(rowStr, "")	// 将rowStr中的子串连接成一个单独的字符串，子串之间用""分隔
```





# 动态规划

## 377. 组合总和 Ⅳ

答案

```go
func combinationSum4(nums []int, target int) int {
	n := len(nums)
	dp := make([]int, target+1)
	dp[0] = 1
	for i:=0; i<=target; i++ {
		for j:=0; j<n; j++ {
			if i >= nums[j] {
				dp[i] += dp[i-nums[j]]
			}
		}
	}
	return dp[target]
}
```



分析

[https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#%E6%80%9D%E8%B7%AF](https://programmercarl.com/0377.组合总和Ⅳ.html#思路)

<img src="https://img-blog.csdnimg.cn/20210131174250148.jpg" alt="377.组合总和Ⅳ" style="zoom:50%;" />

```go
主要是这张图很好
不知道怎么推导dp计算过程的时候
可以参考一下这张图
-------------------------------
内外循环怎么决定也可以看链接里的分析
排列是物品可以打乱的，所以物品只能放到内循环里
```





## 322. 零钱兑换

答案

```go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for j:=1; j<=amount; j++ {
		dp[j] = math.MaxInt32
		for i:=0; i<len(coins); i++ {
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
```



分析

[https://programmercarl.com/0322.%E9%9B%B6%E9%92%B1%E5%85%91%E6%8D%A2.html#%E6%80%9D%E8%B7%AF](https://programmercarl.com/0322.零钱兑换.html#思路)

```go
dp[j] 要取所有 dp[j - coins[i]] + 1 中最小的，所以用min()
这里dp[j]是会不断赋值更新的，要求其中最小的，所以不能是dp[j] = dp[j-coins[i]]+1

本题求钱币最小个数，那么钱币有顺序和没有顺序都可以，都不影响钱币的最小个数。
所以本题并不强调集合是组合还是排列。
如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。

本题钱币数量可以无限使用，那么是完全背包。所以遍历的内循环是正序
```





## 122. 买卖股票的最佳时机 II

答案

```go
func maxProfit(prices []int) int {
	dp := [2]int{}
	dp[0] = -prices[0]
	//dp[1] = -prices[0]
	for i:=1; i<len(prices); i++ {
		dp[0] = max(dp[0], dp[1]-prices[i])			// 随想录里这里会用 dp[i%2][0] = ...
		dp[1] = max(dp[1], dp[0]+prices[i])			// 其实没有必要
	}
	return dp[1]
}

// 求两个数中的较大者
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



分析

```go
121，122，123其实都可以只用一维数组，不需要用二维的
```



## 188. 买卖股票的最佳时机 IV

答案

```go
func maxProfit(k int, prices []int) int {
	if k==0 || len(prices)==0 {
		return 0
	}
	n := 2*k+1
	dp := make([]int, n)
	for i:=1; i<n-1; i+=2 {
		dp[i] = -prices[0]
	}
	for i:=1; i<len(prices); i++ {
		for j:=0; j<n-1; j+=2 {
			dp[j+1] = max(dp[j+1], dp[j]-prices[i])
			dp[j+2] = max(dp[j+2], dp[j+1]+prices[i])
		}
	}
	return dp[n-1]
}


// 求两个数中的较大者
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



分析

```go
甚至Hard也可以只用一维数组

但是后面含冷冻期的不能用一维数组
原因可能是不能在同一天无限买卖，卖之后必须得到第二天

含手续费可以用一维数组，可能是因为虽然卖出有手续费，但是可以同一天无限买卖
```



## 5. 最长回文子串

答案

```go
func longestPalindrome(s string) string {
	n := len(s)
	 if n == 1 {
		 return s
	 }
	 start, maxLen := 0, 1
	// dp[i][j] 表示 s[i..j] 是否是回文串
	dp := make([][]bool, n)
	// 初始化：所有长度为 1 的子串都是回文串
	for i:=0; i<n; i++ {
		dp[i] = make([]bool, n)
		dp[i][i] = true
	}
	for Len :=2; Len<=n; Len++ {	// 先枚举子串长度
		for i:=0; i<n; i++ {		// 枚举左边界，左边界的上限设置可以宽松一些
			j := i+Len-1			// 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
			if j >= n {				// 如果右边界越界，就可以退出当前循环
				break
			}
			if s[i] != s[j] {
				dp[i][j] = false
			} else {
				//if j-i < 3 {	// j-i <= 2   j=i+1 或 j=i+2  在i+1和j-1后，i会>=j
				if j-i < 2 {	// j-i <= 1   j=i+1  在i+1和j-1后，i会>j，这是没有初始化的值(false)
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
			}
			// 只要 dp[i][j] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
			if dp[i][j] && j-i+1>maxLen {
				maxLen = j-i+1
				start = i
			}
		}
	}
	return s[start:start+maxLen]
}
```



分析

```go

```



## 42. 接雨水

答案

```go
func trap(height []int) int {
    left, right := 0, len(height)-1
	maxLeft, maxRight := 0, 0
	ans := 0
	// 双指针法，左右指针代表着要处理的雨水位置，最后一定会汇合
	for left <= right {		// 注意，这里可能 left==right
		// 对于位置left而言，它左边最大值一定是left_max，右边最大值“大于等于”right_max
		if maxLeft < maxRight {	// 如果left_max<right_max，那么无论右边将来会不会出现更大的right_max，都不影响这个结果
			ans += max(0, maxLeft-height[left])
			maxLeft = max(maxLeft, height[left])
			left++
		} else {	// 反之，去处理right下标
			ans += max(0, maxRight-height[right])
			maxRight = max(maxRight, height[right])
			right--
		}
	}
	return ans
}
```



分析

```go
每个位置能储存的雨水量为左边最高柱子的高度 和 右边最高柱子的高度中较小的那个 减去该位置的柱子高度
即: drop[i] = min(maxLeft, maxRight) - height[i]
```



## 300. 最长递增子序列

答案

```go
func lengthOfLIS(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	ans := 1
	dp := make([]int, len(nums))	// dp[i] 为以第 i 个数字结尾的最长上升子序列的长度，nums[i] 必须被选取
	dp[0] = 1
	for i:=1; i<len(nums); i++ {
		dp[i] = 1
		for j:=0; j<i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)		// dp[i] 从 dp[j] 这个状态转移过来
			}
		}
		ans = max(ans, dp[i])
	}
	return ans
}


// 求两个数中的较大者
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



分析

```go

```





# 设计/模拟

## 146. LRU 缓存

List包答案

```go
type LRUNode struct {
	key, value int
}

type LRUCache struct {
	capacity int
	cache map[int]*list.Element		// Element 用于代表双链表的元素
	LRUList *list.List	// List 用于表示双链表
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		capacity: capacity,
		cache: map[int]*list.Element{},
		LRUList: list.New(),	// 通过 container/list 包的 New() 函数初始化 list
	}
}


func (this *LRUCache) Get(key int) int {
	element := this.cache[key]	// 获得 valve
	if element == nil {		// 关键字 key 不在缓存中
		return -1
	}
	this.LRUList.MoveToFront(element)	// 刷新缓存使用时间	将元素 e 移动到链表的开头
	return element.Value.(LRUNode).value
}


func (this *LRUCache) Put(key int, value int)  {
	element := this.cache[key]
	if element != nil {		// 关键字 key 已经存在，则变更其数据值 value
		element.Value = LRUNode{key: key, value: value}
		this.LRUList.MoveToFront(element)	// 刷新缓存使用时间	将元素 e 移动到链表的开头
		return
	}
	// 如果不存在，则向缓存中插入该组 key-value
	this.cache[key] = this.LRUList.PushFront(LRUNode{ key: key, value: value})	//将包含了值v的元素e插入到链表的开头并返回e
	if len(this.cache) > this.capacity {	// 如果插入操作导致关键字数量超过 capacity ，则应该逐出最久未使用的关键字
		delete(this.cache, this.LRUList.Remove(this.LRUList.Back()).(LRUNode).key)
	}
}
```



分析

Go语言list（列表）	http://c.biancheng.net/view/35.html

Go标准库中文文档	container/list	http://cngolib.com/container-list.html#container-list

代码参考  https://leetcode.cn/problems/lru-cache/solution/golang-jian-ji-xie-fa-by-endlesscheng-a3b2/

原理分析  https://leetcode.cn/problems/lru-cache/solution/jian-dan-shi-li-xiang-xi-jiang-jie-lru-s-exsd/

```go
当缓存容量已满，我们不仅仅要删除最后一个 Node 节点，还要把 map 中映射到该节点的 key 同时删除，而这个 key 只能由 Node 得到。如果 Node 结构中只存储 val，那么我们就无法得知 key 是什么，就无法删除 map 中的键，造成错误。

作者：labuladong
链接：https://leetcode.cn/problems/lru-cache/solution/lru-ce-lue-xiang-jie-he-shi-xian-by-labuladong/

所以这里
delete(this.cache, this.LRUList.Remove(this.LRUList.Back()).(LRUNode).key)
是先利用this.LRUList.Back()返回了尾结点，
然后再利用this.LRUList.Remove删除了尾结点，
最后利用delete删除了map里的key
```



## 415. 字符串相加

```go
func addStrings(num1 string, num2 string) string {
	add := 0	// 维护当前是否有进位
	ans := ""
	// 从末尾到开头逐位相加
	for i, j := len(num1)-1, len(num2)-1; i>=0 || j>=0 || add!=0; i, j = i-1, j-1 {
		var x, y int	// 默认为0，即在指针当前下标处于负数的时候返回 0   等价于对位数较短的数字进行了补零操作
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		result := x + y + add
		ans = strconv.Itoa(result%10) + ans
		add = result / 10
	}
	return ans
}
```



分析

```go

```



## 54. 螺旋矩阵

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])
	ans := make([]int, 0)
	top, bottom, left, right := 0, m-1, 0, n-1
	for left <= right && top <= bottom {
		// 将矩阵看成若干层，首先输出最外层的元素，其次输出次外层的元素，直到输出最内层的元素
		for i:=left; i<=right; i++ {			// 左上方到右
			ans = append(ans, matrix[top][i])
		}
		for i:=top+1; i<=bottom; i++ {			// 右上方到下
			ans = append(ans, matrix[i][right])
		}
		// left == right : 剩余的矩阵是一竖的形状  所以不需要再往上
		// top == bottom : 剩余的矩阵是一横的形状  所以不需要再往左
		if left < right && top < bottom {	// 当 left == right 或者 top == bottom 时，不会发生右到左和下到上，否则会重复计数
			for i:=right-1; i>=left; i-- {		// 右下方到左
				ans = append(ans, matrix[bottom][i])
			}
			for i:=bottom-1; i>top; i-- {		// 左下方到上   这里的判断条件是特殊的，不加=号
				ans = append(ans, matrix[i][left])
			}
		}
		top++
		bottom--
		left++
		right--
	}
	return ans
}
```



根据59改的

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])
	ans := make([]int, 0)
	top, bottom, left, right := 0, m-1, 0, n-1
	for left <= right && top <= bottom {
		for i:=left; i<=right; i++ {			// 左上方到右
			ans = append(ans, matrix[top][i])
		}
    top++
		for i:=top; i<=bottom; i++ {			// 右上方到下
			ans = append(ans, matrix[i][right])
		}
    right--
    // 这里的判断条件必须是&&，不能是||
		if left <= right && top <= bottom {// 当 left > right 或者 top > bottom 时，不会发生右到左和下到上，否则会重复计数
			for i:=right; i>=left; i-- {		// 右下方到左
				ans = append(ans, matrix[bottom][i])
			}
      bottom--
			for i:=bottom; i>=top; i-- {		// 左下方到上
				ans = append(ans, matrix[i][left])
			}
      left++
		}
	}
	return ans
}
```



分析

```go
我个人觉得第二个解法比较好，和59是类似的，不用特判，条件也便于理解
```







## 59. 螺旋矩阵II

```go
func generateMatrix(n int) [][]int {
	top, bottom := 0, n-1
	left, right := 0, n-1
	num := 1	// 给矩阵赋值
	tar := n * n
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	for num <= tar { //
		for i := left; i <= right; i++ {	// 左上到右
			matrix[top][i] = num
			num++
		}
		top++	// 右上角往下一格
		for i := top; i <= bottom; i++ {	// 右上到下
			matrix[i][right] = num
			num++
		}
		right--	// 右下角往左一格
		for i := right; i >= left; i-- {	// 右下到左
			matrix[bottom][i] = num
			num++
		}
		bottom--	// 左下角往上一格
		for i := bottom; i >= top; i-- {	// 左下到上
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}

```



分析

```go
按照相同的原则，每条边左闭右开或左闭右闭
```







# 排序

## 215. 数组中的第K个最大元素

```go
func findKthLargest(nums []int, k int) int {
	start, end := 0, len(nums)-1
	for {
		if start >= end {
			return nums[end]
		}
		p := partition(nums, start, end)
		if p+1 == k {			// 第K大, 即k == p+1, return nums[p]
			return nums[p]
		} else if p+1 < k {		// 对p的右边数组进行分治, 即对 [p+1,right]进行分治
			start = p + 1
		} else {				// 对p的左边数组进行分治, 即对 [left,p-1]进行分治
			end = p - 1
		}
	}
}

func partition(nums []int, start, end int) int {
	// 从大到小排序
	pivot := nums[end]
	for i:=start; i<end; i++ {
		if nums[i] > pivot {	// 大的放左边
			nums[start], nums[i] = nums[i], nums[start]
			start++
		}
	}
	// for循环完毕, nums[left]左边的值, 均大于nums[left]右边的值
	nums[start], nums[end] = nums[end], nums[start]	// 此时nums[end]是nums[start]右边最大的值，需要交换一下
	return start	// 确定了nums[start]的位置
}
```



分析

代码参考+原理分析  https://leetcode.cn/problems/kth-largest-element-in-an-array/solution/partition-zhu-shi-by-da-ma-yi-sheng/

```go
便捷：
func findKthLargest(nums []int, k int) int {
	n := len(nums)
	sort.Ints(nums)
	return nums[n-k]
}
```



## 912. 排序数组

```go
func sortArray(nums []int) []int {
	var quick func(left, right int)
	quick = func(left, right int) {
		// 递归终止条件
		if left > right {
			return
		}
		i, j, pivot := left, right, nums[left]	// 左右指针及主元
		for i < j {
			// 寻找小于主元的右边元素
			for i<j && nums[j]>=pivot {
				j--
			}
			// 寻找大于主元的左边元素
			for i<j && nums[i]<=pivot {
				i++
			}
			// 交换i, j下标元素
			nums[i], nums[j] = nums[j], nums[i]
		}
		// 此时nums[left]还在第一位，需要和小于它的nums[i]，或是i,j重叠处交换一下
		nums[i], nums[left] = nums[left], nums[i]
		quick(left, i-1)
		quick(i+1,right)
	}
	quick(0, len(nums)-1)
	return nums
}
```



分析

```go
思考一下考试时快排怎么做的
// 此时nums[left]还在第一位，需要和小于它的nums[i]，或是i,j重叠处交换一下
nums[i], nums[left] = nums[left], nums[i]

每一轮排序最后要交换一下
```





## 33. 搜索旋转排序数组

```go
func search(nums []int, target int) int {
	n := len(nums)
	if n == 1 {
		if nums[0] == target {
			return 0
		} else {
			return -1
		}
	}
	left, right, mid := 0, n-1, 0
	/*
	将数组一分为二，其中一定有一个是有序的，另一个可能是有序，也能是部分有序。
	此时有序部分用二分法查找。无序部分再一分为二，其中一个一定有序，另一个可能有序，可能无序。就这样循环.
	*/
	for left <= right {
		mid = (left+right) / 2
		if nums[mid] == target {	// 判断是否找到target
			return mid
		}
		if nums[0] <= nums[mid] {	// 0~mid是有序的	这里必须加=号
			if nums[0]<=target && target<nums[mid] {	// target在0~mid范围内，进行查找
				right = mid - 1
			} else {	// target不在0~mid范围内，在无序的mid+1~n-1范围内重新查找
				left = mid + 1
			}
		} else {	// 0~mid不是有序的，mid~n是有序的
			if nums[mid]<target && target<=nums[n-1] {	// target在mid~n-1范围内，进行查找
				left = mid + 1
			} else {	// target不在mid~n-1范围内，在无序的0~mid-1范围内重新查找
				right = mid - 1
			}
		}
	}
	return -1
}
```



分析

```go
有序数组+logN ========== 二分法
```



## 704. 二分查找

```go
func search(nums []int, target int) int {
	high := len(nums)
	low := 0
	for low < high {
		mid := low + (high-low)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			low = mid + 1
		} else {
			high = mid	// 因为是左闭右开区间，所以这里不能是high = mid - 1，mid-1有可能是答案
		}
	}
	return -1
}
```



分析

```go
有序数组+logN ========== 二分法
注意边界条件
```









# 栈/队列

## 20. 有效的括号

```go
func isValid(s string) bool {
	hash := map[byte]byte{')': '(', ']': '[', '}': '{'}
	stack := []byte{}	// 注意是string中的每个字符是byte类型
	if s == "" {
		return true
	}
	for i:=0; i<len(s); i++ {
		if s[i]=='(' || s[i]=='{' ||s[i]=='[' {
			stack = append(stack, s[i])		// 注意是string中的每个字符是byte类型
		} else if len(stack)>0 && stack[len(stack)-1]==hash[s[i]] {
			stack = stack[:len(stack)-1]
		} else {
			return false
		}
	}
	return len(stack) == 0
}
```



分析

```go

```









# ACM模式

OJ（牛客网）输入输出练习 Go实现	https://blog.csdn.net/aron_conli/article/details/113462234

OJ在线编程常见输入输出练习场 	https://ac.nowcoder.com/acm/contest/5657#question

GoLang之ACM控制台输入输出	https://blog.csdn.net/weixin_52690231/article/details/125436414



## fmt

### Scan

```
func Scan(a ...interface{}) (n int, err error)
```

> Scan从标准输入扫描文本，将成功读取的空白分隔的值保存进成功传递给本函数的参数。换行视为空白。
>
> 返回成功扫描的条目个数和遇到的任何错误。如果读取的条目比提供的参数少，会返回一个错误报告原因。

```go
var a1 int
var a2 string
n, err := fmt.Scan(&a1, &a2)
```

参数间以空格或回车键进行分割。
如果输入的参数不够接收的，按回车后仍然会等待其他参数的输入。
如果输入的参数大于接收的参数，只有给定数量的参数被接收，其他参数自动忽略。



### Scanf

```go
func Scanf(format string, a ...interface{}) (n int, err error)
```

`Scanf`也可以接收多个参数，但是接收字符串的话，只能在最后接收；
否则按照`"%s%d"`的格式进行接收，无论输入多少字符（包含数字），都会被认定是第一个字符串的内容，不会被第二个参数所接收。

```go
package main

import "fmt"

func main() {
	var a string

	n, err := fmt.Scanf("%s", &a) // afdsfdsfdsfds

	fmt.Println("n = ", n)     // n =  1
	fmt.Println("err = ", err) // err =  <nil>
	fmt.Println("a = ", a)     // a =  afdsfdsfdsfds
}
```



### Scanln

```
func Scanln(a ...interface{}) (n int, err error)
```

> Scanln与Scan类似，但在换行时停止扫描，并且在最后一项之后必须有换行或EOF。

```go
package main

import "fmt"

func main() {
	var a string
	var b string

	n, err := fmt.Scanln(&a, &b)

	fmt.Println("n = ", n)
	fmt.Println("err = ", err)
	fmt.Println("a = ", a)
	fmt.Println("b = ", b)
}
```

与`Scan`的区别：如果设置接收2个参数，`Scan`在输入一个参数后进行回车，会继续等待第二个参数的键入；而`Scanln`直接认定输入了一个参数就截止了，只会接收一个参数并产生`error（unexpected newline）`，且`n = 1`。

说通俗写，就是`Scanln`认定回车标志着==阻塞接收参数==，而`Scan`认定回车只是一个==分隔符（或空白）==而已。





## bufio

`bufio`包是对IO的封装，可以操作文件等内容，同样可以用来接收键盘的输入，此时对象不是文件等，而是`os.Stdin`，也就是标准输入设备。

[bufio包文档](https://studygolang.com/pkgdoc)

`bufio`包含了Reader、Writer、Scanner等对象，封装了很多对IO内容的处理方法，但应对键盘输入来说，通过创建Reader对象，并调用其Read*系列的方法即可



### 创建Reader对象

```go
reader := bufio.NewReader(os.Stdin)
```



### ReadByte

```
func (b *Reader) ReadByte() (c byte, err error)
```

用来接收一个`byte`类型，会阻塞等待键盘输入。

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {

	reader := bufio.NewReader(os.Stdin)

	n, err := reader.ReadByte() // a

	fmt.Println("n = ", n)             // n =  97
	fmt.Println("string =", string(n)) // string = a
	fmt.Println("err = ", err)         // err =  <nil>

}
```



### ReadBytes

```
func (b *Reader) ReadBytes(delim byte) (line []byte, err error)
```

该方法，输入参数为一个byte字符，当输入遇到该字符，会停止接收并返回。
接收的内容包括**该停止字符**以及前面的内容。

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	n, err := reader.ReadBytes('\n')
	fmt.Println("n = ", n)
	fmt.Println("string =", string(n))
	fmt.Println("err = ", err)

}
```

输入的内容为【a】【空格】【b】【回车】
n同时接收了4个byte。包括空格和回车。

![命令行显示](https://img-blog.csdnimg.cn/79866a0b3b7c41e48e3d9d58d20fb0fc.png)

**Tips：**
Reader可以接收包含空格内容的字符串，而不进行分割，这是`bufio.Reader`与`fmt`系的一大不同。



### ReadString

```
func (b *Reader) ReadString(delim byte) (line string, err error)
```

> ReadString读取直到第一次遇到delim字节，返回一个包含已读取的数据和**delim字节**的字符串。
>
> 如果ReadString方法在读取到delim之前遇到了错误，它会返回在错误之前读取的数据以及该错误（一般是io.EOF）。当且仅当ReadString方法返回的切片不以delim结尾时，会返回一个非nil的错误。

```go
package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {

	reader := bufio.NewReader(os.Stdin)

	line, err := reader.ReadString('\n')

	fmt.Println("line = ", line)
	fmt.Println("err = ", err)

}
```

该函数可以接收包含空格的字符串。
如果设定回车键为delim byte，则遇到回车后结束接收，同时也会接收回车键。
当以’a’等byte为终止符时，如果没有遇到该符，即使回车也会继续接收。
如果在按下终止符后没有回车，继续键入内容，则只会接收第一次终止符及其之前的内容，之后的内容自动忽略。





### NewScanner

必须整行读（用 fmt、os、bufio、strconv、strings 实现）



### 任意数量求和

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	inputs := bufio.NewScanner(os.Stdin)	// 不用放在循环内部
	for inputs.Scan() {  //每次读入一行
		data := strings.Split(inputs.Text(), " ")  //通过空格将他们分割，并存入一个字符串切片
		var sum int
		for _, v := range data {
			val, _ := strconv.Atoi(v)   //将字符串转换为int
			sum += val
		}
		fmt.Println("sum = ", sum)
		fmt.Println("data = ", data)		// data 是 []string
		fmt.Println("data[0] = ", data[1])	// data 是 string	
	}
}
```



### 任意数量[]int{}

```go
func main() {
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	data := strings.Split(input.Text(), " ")	// data 是 []string
	fmt.Println("data = ", data)
	nums := []int{}
	for i:=0; i<len(data); i++ {
		v, _ := strconv.Atoi(data[i])	// string转int
		nums = append(nums, v)
	}
	fmt.Println("nums = ", nums)		// nums 是 []int
}
```







### 指定长宽矩阵

```go
package main

import (
	"fmt"
)

func main() {
	var m, n int
	fmt.Scanln(&m, &n)

	res := make([][]int, m)
	for i := range res {
		res[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			fmt.Scan(&res[i][j])
		}
	}
	fmt.Println(res)	// 每个切片中以“ ”分隔每个元素
}

/*
输入：
2 3
1 2 3 4 5 6

输出：
[[1 2 3] [4 5 6]]
*/
```



### 任意矩阵

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	input := bufio.NewScanner(os.Stdin)
	input.Scan() //读取一行内容
    // 因为Atoi，所以要有两个接收符：m, _
    // 不加[0]就是[]int，加了就是int
	m, _ := strconv.Atoi(strings.Split(input.Text(), " ")[0])	// 第一行的第一个字符，转换为int
	n, _ := strconv.Atoi(strings.Split(input.Text(), " ")[1])	// 第一行的第二个字符，转换为int
	res := make([][]int, m)
	for i := range res {
		res[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		input.Scan() //读取一行内容
		for j := 0; j < n; j++ {
			res[i][j], _ = strconv.Atoi(strings.Split(input.Text(), " ")[j])
		}
	}
}

```





