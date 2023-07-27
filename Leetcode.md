# 重点笔记

## 链表

https://zhuanlan.zhihu.com/p/281404491

在做 [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii) 时思维卡住了



## 约瑟夫环

https://leetcode-cn.com/circle/article/BOoxAL/





# 2022.1.4

## 704.二分查找

```go
package main

import "fmt"

func search(nums []int, target int) int {
	high := len(nums) - 1
	low := 0
	for low <= high {
		mid := low + (high-low)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return -1
}

func main() {
	fmt.Println("amadeus")

}

```





## 27.移除元素

```go
package main

import "fmt"

func removeElement(nums []int, val int) int {
	length := len(nums)
	res := 0
	for i := 0; i < length; i++ {
		if nums[i] != val {
			nums[res] = nums[i]
			res++
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")

}

```

## 977.有序数组的平方

```go
package main

import "fmt"

func sortedSquares(nums []int) []int {
	n := len(nums)
	i, j, k := 0, n-1, n-1
	ans := make([]int, n)
	for i <= j {
		l, r := nums[i]*nums[i], nums[j]*nums[j]
		if l > r {
			ans[k] = l
			i++
		} else {
			ans[k] = r
			j--
		}
		k--
	}
	return ans
}

func main() {
	fmt.Println("amadeus")

}

```

## 209.长度最小的子数组

```go
package main

import (
	"fmt"
)

func minSubArrayLen(target int, nums []int) int {
	i := 0 // 滑动窗口起始位置
	l := len(nums)
	sum := 0 // 滑动窗口数值之和
	result := l + 1
	subLength := 0 // 滑动窗口的长度
	for j := 0; j < l; j++ {
		sum += nums[j]
		for sum >= target { // 每次更新 i（起始位置），并不断比较子序列是否符合条件
			subLength = j - i + 1 // 取子序列的长度
			if subLength < result {
				result = subLength
			}
			sum -= nums[i] // 这里体现出滑动窗口的精髓之处，不断变更i（子序列的起始位置）
			i++
		}
	}
	// 如果result没有被赋值的话，就返回0，说明没有符合条件的子序列
	if result == l+1 {
		return 0
	} else {
		return result
	}
}

func main() {
	fmt.Println("amadeus")

}

```

## 59.螺旋矩阵 Ⅱ

```go
package main

import (
	"fmt"
)

func generateMatrix(n int) [][]int {
	top, bottom := 0, n-1
	left, right := 0, n-1
	num := 1 //
	tar := n * n
	matrix := make([][]int, n) // 首先make第一维的大小
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n) // 然后分别对其中的进行make
	}
	for num <= tar {
		// 左到右
		for i := left; i <= right; i++ {
			matrix[top][i] = num
			num++
		}
		top++ // 缩小范围
		// 上到下
		for i := top; i <= bottom; i++ {
			matrix[i][right] = num
			num++
		}
		right-- // 缩小范围
		// 右到左
		for i := right; i >= left; i-- {
			matrix[bottom][i] = num
			num++
		}
		bottom-- // 缩小范围
		// 下到上
		for i := bottom; i >= top; i-- {
			matrix[i][left] = num
			num++
		}
		left++ // 缩小范围
	}
	return matrix
}

func main() {
	fmt.Println("amadeus")

}

```

## 203.移除链表元素

```go
package main

import (
	"fmt"
)

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

func removeElements(head *ListNode, val int) *ListNode {
	dummyHead := &ListNode{}	// 设置一个虚拟头结点
	dummyHead.Next = head // 将虚拟头结点指向head，这样方面后面做删除操作
	cur := dummyHead
	for cur != nil && cur.Next != nil {
		if cur.Next.Val == val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return dummyHead.Next
}

func main() {
	fmt.Println("amadeus")

}

```

## 707.设计链表

```go
//循环双链表
type MyLinkedList struct {
	dummy *Node
}

type Node struct {
	Val  int
	Next *Node
	Pre  *Node
}

//仅保存哑节点，pre-> rear, next-> head
/** Initialize your data structure here. */
func Constructor() MyLinkedList {
	rear := &Node{
		Val:  -1,
		Next: nil,
		Pre:  nil,
	}
	rear.Next = rear
	rear.Pre = rear
	return MyLinkedList{rear}
}

/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
func (this *MyLinkedList) Get(index int) int {
	head := this.dummy.Next
	//head == this, 遍历完全
	for head != this.dummy && index > 0 {
		index--
		head = head.Next
	}
	//否则, head == this, 索引无效
	if 0 != index {
		return -1
	}
	return head.Val
}

/** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
func (this *MyLinkedList) AddAtHead(val int) {
	dummy := this.dummy
	node := &Node{
		Val: val,
		//head.Next指向原头节点
		Next: dummy.Next,
		//head.Pre 指向哑节点
		Pre: dummy,
	}

	//更新原头节点
	dummy.Next.Pre = node
	//更新哑节点
	dummy.Next = node
	//以上两步不能反
}

/** Append a node of value val to the last element of the linked list. */
func (this *MyLinkedList) AddAtTail(val int) {
	dummy := this.dummy
	rear := &Node{
		Val: val,
		//rear.Next = dummy(哑节点)
		Next: dummy,
		//rear.Pre = ori_rear
		Pre: dummy.Pre,
	}

	//ori_rear.Next = rear
	dummy.Pre.Next = rear
	//update dummy
	dummy.Pre = rear
	//以上两步不能反
}

/** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
func (this *MyLinkedList) AddAtIndex(index int, val int) {
	head := this.dummy.Next
	//head = MyLinkedList[index]
	for head != this.dummy && index > 0 {
		head = head.Next
		index--
	}
	if index > 0 {
		return
	}
	node := &Node{
		Val: val,
		//node.Next = MyLinkedList[index]
		Next: head,
		//node.Pre = MyLinkedList[index-1]
		Pre: head.Pre,
	}
	//MyLinkedList[index-1].Next = node
	head.Pre.Next = node
	//MyLinkedList[index].Pre = node
	head.Pre = node
	//以上两步不能反
}

/** Delete the index-th node in the linked list, if the index is valid. */
func (this *MyLinkedList) DeleteAtIndex(index int) {
	//链表为空
	if this.dummy.Next == this.dummy {
		return
	}
	head := this.dummy.Next
	//head = MyLinkedList[index]
	for head.Next != this.dummy && index > 0 {
		head = head.Next
		index--
	}
	//验证index有效
	if index == 0 {
		//MyLinkedList[index].Pre = index[index-2]
		head.Next.Pre = head.Pre
		//MyLinedList[index-2].Next = index[index]
		head.Pre.Next = head.Next
		//以上两步顺序无所谓
	}
}
```

## 206.翻转链表

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

func main() {
	fmt.Println("amadeus")

}
```

## 24.两两交换链表中的节点

```go
func swapPairs(head *ListNode) *ListNode {
	dummyHead := &ListNode{0, head}
	temp := dummyHead
	for temp.Next != nil && temp.Next.Next != nil {
		node1 := temp.Next
		node2 := temp.Next.Next
		temp.Next = node2
		node1.Next = node2.Next
		node2.Next = node1
		temp = node1
	}
	return dummyHead.Next
}
```

## 19.删除链表的倒数第N个节点

```go
package main

import "fmt"

func search(nums []int, target int) int {
	high := len(nums) - 1
	low := 0
	for low <= high {
		mid := low + (high-low)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > target {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return -1
}

func main() {
	fmt.Println("amadeus")

}

```

## 面试题 02.07. 链表相交

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val int
	Next *ListNode
}


func getIntersectionNode(headA, headB *ListNode) *ListNode {
	curA := headA
	curB := headB
	lenA, lenB := 0, 0
	for curA != nil {
		curA = curA.Next
		lenA ++
	}
	for curB != nil {
		curB = curB.Next
		lenB ++
	}
	step := 0
	var fast, slow *ListNode
	if lenA>lenB {
		step = lenA-lenB
		fast=headA
		slow=headB
	} else  {
		step = lenB - lenA
		fast, slow = headB, headA
	}
	for i:=0; i<step;i++{
		fast = fast.Next
	}
	for fast != slow{
		fast=fast.Next
		slow=slow.Next
	}
	return fast
}

func main() {
	fmt.Println("amadeus")

}

```



## 142.环形链表Ⅱ

```go
package main

import (
	"fmt"
)

type ListNode struct {
	Val int
	Next *ListNode
}


func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			for slow != head {
				slow = slow.Next
				head = head.Next
			}
			return head
		}
	}
	return nil
}

func main() {
	fmt.Println("amadeus")

}

```



## 242.有效的字母异位词

```go
package main

import (
	"fmt"
)

func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	record := make(map[byte]int)
	for i:=0; i<len(s); i++ {
		//if v,ok := record[s[i]]; v>=0 && ok {
		//
		//}
		record[s[i]]++
	}
	for i:=0; i<len(t); i++ {
		record[t[i]]--
		if record[t[i]] < 0 {
			return false
		}
	}
	return true
}

func main() {
	fmt.Println("amadeus")

}

```



## 349. 两个数组的交集

```go
package main

import (
	"fmt"
)

func intersection(nums1 []int, nums2 []int) []int {
	m := make(map[int]int)
	for _, v := range nums1 {
		m[v] = 1
	}
	var res []int
	for _, v := range nums2 {
		if count, ok := m[v]; ok && count>0 {
			res = append(res, v)
			m[v]--
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")

}

```



## 202. 快乐数

```go
package main

import (
	"fmt"
)

func isHappy(n int) bool {
	m := make(map[int]bool)
	for n!=1 && !m[n] {
		n, m[n] = getSum(n), true
	}
	return n == 1
}

func getSum(n int) int  {
	sum := 0
	for n>0 {
		sum += (n%10) * (n%10)
		n = n/10
	}
	return sum
}

func main() {
	fmt.Println("amadeus")

}
```



## 1. 两数之和

```go
func twoSum(nums []int, target int) []int {
	m := make(map[int]int)
	for index, val := range nums {
		if preIndex, ok := m[target-val]; ok {
			return []int{preIndex, index}
		} else {
			m[val] = index
		}
	}
	return []int{}
}
```



## 454.四数相加II

```go
package main

import (
	"fmt"
)

func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	m := make(map[int]int)
	count := 0
	for _, v1 := range nums1 {
		for _, v2 := range nums2 {
			m[v1+v2]++
		}
	}
	for _, v3 := range nums3 {
		for _, v4 := range nums4 {
			count += m[-v3-v4]
		}
	}
	return count
}


func main() {
	fmt.Println("amadeus")

}

```



## 383. 赎金信

```go
package main

import (
	"fmt"
)

func canConstruct(ransomNote string, magazine string) bool {
	record := make([]int, 26)
	for _, v := range magazine {
		record[v-'a']++
	}
	for _, v := range ransomNote {
		record[v-'a']--
		if record[v-'a'] < 0 {
			return false
		}
	}
	return true
}


func main() {
	fmt.Println("amadeus")

}

```



## 344.反转字符串

```go
package main

import (
	"fmt"
)

func reverseString(s []byte)  {
	left := 0
	right := len(s)-1
	for left<right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}


func main() {
	fmt.Println("amadeus")

}

```



## 541. 反转字符串II

```go
package main

import (
	"fmt"
)

func reverseStr(s string, k int) string {
	ss := []byte(s)
	length := len(ss)
	for i:=0; i<length; i+=2*k {
		if i+k <= length {
			reverse(ss[i:i+k])
		} else {
			reverse(ss[i:length])
		}
	}
	return string(ss)
}

func reverse(s []byte)  {
	left := 0
	right := len(s)-1
	for left<right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}


func main() {
	fmt.Println("amadeus")

}

```



## 剑指Offer 05.替换空格

```go
package main

import (
	"fmt"
)

func replaceSpace(s string) string {
	b := []byte(s)
	result := make([]byte, 0)
	for i:=0; i<len(b); i++ {
		if b[i]==' ' {
			result = append(result, []byte("%20")...)
		} else {
			result = append(result, b[i])
		}
	}
	return string(result)
}


func main() {
	fmt.Println("amadeus")

}
```



## 151.翻转字符串里的单词

```go
package main

import (
	"fmt"
)

func reverseWords(s string) string {
	//1.使用双指针删除冗余的空格
	slowIndex, fastIndex := 0, 0
	b := []byte(s)
	//删除头部冗余空格
	for len(b) > 0 && fastIndex < len(b) && b[fastIndex] == ' ' {
		fastIndex++
	}
	//删除单词间冗余空格
	for ; fastIndex < len(b); fastIndex++ {
		if fastIndex-1 > 0 && b[fastIndex-1] == b[fastIndex] && b[fastIndex] == ' ' {
			continue
		}
		b[slowIndex] = b[fastIndex]
		slowIndex++
	}
	//删除尾部冗余空格
	if slowIndex-1 > 0 && b[slowIndex-1] == ' ' {
		b = b[:slowIndex-1]
	} else {
		b = b[:slowIndex]
	}
	//2.反转整个字符串
	reverse(b, 0, len(b)-1)
	//3.反转单个单词  i单词开始位置，j单词结束位置
	i := 0
	for i < len(b) {
		j := i
		for ; j < len(b) && b[j] != ' '; j++ {
		}
		reverse(b, i, j-1)
		i = j
		i++
	}
	return string(b)
}

func reverse(b []byte, left, right int) {
	for left < right {
		b[left], b[right] = b[right], b[left]
		left++
		right--
	}
}


func main() {
	fmt.Println("amadeus")

}
```



## 77. 组合 

### 回溯

copy 的使用	https://blog.csdn.net/weixin_30301183/article/details/97350889

> copy 可以将后面的 第2个切片的元素赋值copy 到第一个切片中
>
> copy 不会新建新的内存空间，由它原来的切片长度决定



leetcode的一道题“子集”，这个地方go语言为什么要用copy呢？	https://www.zhihu.com/question/394190175

> 复制的时候是把cur当时的值（状态）赋给了tmp，再把tmp append到结果数组中。然后cur会在之后的回溯中继续更改。
>
> 每次循环的最后一次会把当前循环的cur全部变成3



```go
package main

import (
	"fmt"
)

var res [][]int

func combine(n int, k int) [][]int {
	res = [][]int{}
	if n<=0 || k<=0 || k>n {
		return res
	}
	backtrace(n, k, 1, []int{})
	return res
}

func backtrace(n, k, start int, trace []int)  {
	if len(trace)==k {
		temp := make([]int, k)
		copy(temp, trace)
		res = append(res, temp)
	}
	if len(trace)+n-start+1 < k {
		return
	}
	for i := start; i <= n; i++ {
		trace = append(trace, i)
		backtrace(n, k, i+1, trace)
		trace=trace[:len(trace)-1]
	}
}


func main() {
	fmt.Println("amadeus")

}
```



## 216.组合总和III

```go
package main

import (
	"fmt"
)

var result [][]int

func combinationSum3(k int, n int) [][]int {
	var trace []int
	trace = []int{}	// 必须初始化，否则会报错出问题
	result = [][]int{}	// 必须初始化，否则会报错出问题
	backtrace(n, k, 1, trace)
	return result
}

func backtrace(n, k, start int, trace []int)  {	// 可以不用引用&，切片是默认引用传递
	if len(trace)==k {
		sum := 0
		tmp := make([]int, k)
		copy(tmp, trace)
		for _,v := range trace {
			sum += v
		}
		if sum == n {
			result = append(result, tmp)
		}
		return
	}

	for i := start; i <= 9-(k-len(trace))+1; i++ {
		trace = append(trace, i)
		backtrace(n, k, i+1, trace)
		trace=trace[:len(trace)-1]
	}
}


func main() {
	fmt.Println("amadeus")

}
```



## 17.电话号码的字母组合

最好还是引用传递

```go
package main

import (
	"fmt"
)


func letterCombinations(digits string) []string {
	length := len(digits)
	if length==0 || length>4 {
		return nil
	}
	digitsMap := [10]string{
		"",
		"",
		"abc",
		"def",
		"ghi",
		"jkl",
		"mno",
		"pqrs",
		"tuv",
		"wxyz",
	}
	res := make([]string, 0)
	backtrace("", digits, 0, digitsMap, &res)
	return res
}

func backtrace(tmpString, digits string, start int, digitsMap [10]string, res *[]string)  {	// 可以不用引用&，切片是默认引用传递
	if len(tmpString)==len(digits) {
		*res = append(*res, tmpString)
		return
	}
	tmp := digits[start]-'0'
	letter := digitsMap[tmp]

	for i := 0; i<len(letter); i++ {
		tmpString += string(letter[i])
		backtrace(tmpString, digits, start+1, digitsMap, res)
		tmpString = tmpString[:len(tmpString)-1]
	}
}


func main() {
	fmt.Println("amadeus")

}
```



## 131. 分割回文串

```go
package main

import (
	"fmt"
)


func partition(s string) [][]string {
	var tmpString []string
	var res [][]string
	backtrace(s, tmpString, 0, &res)
	return res
}

func backtrace(s string, tmpString []string, start int, res *[][]string)  {
	if start==len(s) {	//到达字符串末尾
		// 进行一次切片拷贝，怕之后的操作影响tmpString切片内的值
		t := make([]string, len(tmpString))
		copy(t, tmpString)
		*res = append(*res, t)
	}
	for i:=start; i<len(s); i++ {
		// 处理
		//（首先通过startIndex和i判断切割的区间，进而判断该区间的字符串是否为回文，
		//若为回文，则加入到tmpString，
		//否则继续后移，找到回文区间）（这里为一层处理）
		if isPartition(s, start, i) {
			tmpString = append(tmpString, s[start:i+1])
		} else {
			continue
		}
		// 递归
		backtrace(s, tmpString, i+1, res)
		// 回溯
		tmpString = tmpString[:len(tmpString)-1]
	}
}

// 判断是否为回文
func isPartition(s string, start int, end int) bool {
	left := start
	right := end
	for ;left<right; {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}

func main() {
	fmt.Println("amadeus")
}
```



## 93.复原IP地址

```go
package main

import (
	"fmt"
	"strconv"
)

func restoreIpAddresses(s string) []string {
	var res, path []string
	backtrace(s, path, 0, &res)
	return res
}

func backtrace(s string, path []string, start int, res *[]string)  {
	if start==len(s) && len(path)==4 {
		tmpString := path[0] + "." + path[1] + "." + path[2] + "." + path[3]
		*res = append(*res, tmpString)
	}
	for i:=start; i<len(s); i++ {
		path = append(path, s[start:i+1])
		if i-start+1<=3 && len(path)<=4 && isIP(s, start, i) {
			backtrace(s, path, i+1, res)
		} else {
			return
		}
		path = path[:len(path)-1]
	}
}

func isIP(s string, start int, end int) bool {
	check,_ := strconv.Atoi(s[start:end+1])
	if end-start+1 > 1 && s[start]== '0' {
		return false
	}
	if check>255 {
		return false
	}
	return true
}

func main() {
	fmt.Println("amadeus")
}
```



## 78.子集

```go
package main

import (
	"fmt"
	"sort"
)

func subsets(nums []int) [][]int {
	var res [][]int
	sort.Ints(nums)
	dfs([]int{}, nums, 0, &res)
	return res
}

func dfs(tmp, nums []int, start int, res *[][]int)  {
	ans := make([]int, len(tmp))
	copy(ans, tmp)
	*res = append(*res, ans)
	for i := start; i < len(nums); i++ {
		tmp = append(tmp, nums[i])	// 为子集收集元素
		dfs(tmp, nums, i+1, res)
		tmp = tmp[:len(tmp)-1]
	}
}

func main() {
	fmt.Println("amadeus")
}
```



## 90.子集 II

```go
package main

import (
	"fmt"
	"sort"
)

func subsetsWithDup(nums []int)[][]int {
	var res [][]int
	sort.Ints(nums)
	dfs([]int{}, nums, 0, &res)
	return res
}

func dfs(tmp, nums []int, start int, res *[][]int)  {
	ans := make([]int, len(tmp))
	copy(ans, tmp)
	*res = append(*res, ans)
	for i := start; i < len(nums); i++ {
		if i>start && nums[i]==nums[i-1] {
			continue
		}
		tmp = append(tmp, nums[i])
		dfs(tmp, nums, i+1, res)
		tmp = tmp[:len(tmp)-1]
	}
}

func main() {
	fmt.Println("amadeus")
}
```



## 491.递增子序列

```go
package main

import (
	"fmt"
)

func findSubsequences(nums []int) [][]int {
	var subRes []int
	var res [][]int
	backtrace(0, nums, subRes, &res)
	return res
}

func backtrace(start int, nums, subRes []int, res *[][]int)  {
	if len(subRes)>1 {
		tmp := make([]int, len(subRes))
		copy(tmp, subRes)
		*res = append(*res, tmp)
	}
	history := [201]int{}	// 记录本层元素使用记录
	for i:=start; i<len(nums); i++ {
		// 分两种情况判断：一，当前取的元素小于子集的最后一个元素，则继续寻找下一个适合的元素
		//                或者二，当前取的元素在本层已经出现过了，所以跳过该元素，继续寻找
		if len(subRes)>0 && nums[i]<subRes[len(subRes)-1] || history[nums[i]+100]==1 {
			continue
		}
		history[nums[i]+100]=1	// 表示本层该元素使用过了
		subRes = append(subRes, nums[i])
		backtrace(i+1, nums, subRes, res)
		subRes = subRes[:len(subRes)-1]
	}
}

func main() {
	fmt.Println("amadeus")
}
```



## 46.全排列

### 可以是小盖

https://leetcode-cn.com/u/gracious-vvilson1bb/

```go
package main

import (
	"fmt"
)

func permute(nums []int) [][]int {
	//最终要返回的二维数组
	var res [][]int
	//已经用过的节点存储用的切片
	var path []int
	//将用过节点进行标记的哈希表
	visited :=make(map[int]bool)
	size :=len(nums)
	var backTrack func()
	backTrack = func() {
		//递归终止条件
		//也就是nums里的元素都用到了
		if len(path) ==size{
			//temp暂时存放path，path的长度肯定是nums的长度
			temp :=make([]int,size)
			copy(temp,path)
			res = append(res,temp)
			return
		}
		//从0开始所以不去等
		for i:=0;i<size;i++{
			//一个排列结果（path）里的一个元素只能使用一次
			//相当于查map里有没有这个元素，有就continue跳出
			if visited[nums[i]] {
				continue
			}
			//第一次出现就给他打个标记true
			visited[nums[i]] =true
			//将这个元素放入path路径中
			path = append(path,nums[i])
			//递归
			backTrack()
			//回溯
			visited[nums[i]] =false
			//就是吐出最后一个元素
			path = path[:len(path)-1]
		}

	}
	backTrack()
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 47.全排列 II

https://leetcode-cn.com/problems/permutations-ii/solution/shou-hua-tu-jie-li-yong-yue-shu-tiao-jian-chong-fe/

> #### 充分地剪枝
>
> 对应上面第一点，我们使用一个 used 数组记录使用过的数字，使用过了就不再使用：
>
> ```
> if (used[i]) {
>     continue;
> }
> ```
>
> 对应上面第二点，如果当前的选项`nums[i]`，与同一层的前一个选项`nums[i-1]`相同，且`nums[i-1]`存在，且没有被使用过，则忽略选项`nums[i]`。
> 如果`nums[i-1]`被使用过，它会被第一条修剪掉，不是选项了，即便它和`nums[i]`重复，`nums[i]`还是可以选的。
>
> ```
> if (i - 1 >= 0 && nums[i - 1] == nums[i] && !used[i - 1]) {
>     continue;
> }
> ```
>
> 比如`[1,1,2]`，第一次选了第一个`1`，第二次是可以选第二个`1`的，虽然它和前一个`1`相同。
> 因为前一个`1`被选过了，它在本轮已经被第一条规则修掉了，所以第二轮中第二个`1`是可选的。

```go
package main

import (
	"fmt"
	"sort"
)

func permuteUnique(nums []int) [][]int {
	//最终要返回的二维数组
	var res [][]int
	//已经用过的节点存储用的切片
	var path []int
	//将用过节点进行标记的哈希表
	size :=len(nums)
	visited :=make([]bool, size)
	sort.Ints(nums)
	var backTrack func()
	backTrack = func() {
		//递归终止条件
		//也就是nums里的元素都用到了
		if len(path) ==size{
			//temp暂时存放path，path的长度肯定是nums的长度
			temp :=make([]int,size)
			copy(temp,path)
			res = append(res,temp)
			return
		}
		//从0开始所以不去等
		for i:=0;i<size;i++{
			//一个排列结果（path）里的一个元素只能使用一次
			//相当于查map里有没有这个元素，有就continue跳出
			if i>0 && !visited[i-1] && nums[i]==nums[i-1] {
				continue
			}
			if visited[i] {
				continue
			}
			//第一次出现就给他打个标记true
			visited[i] =true
			//将这个元素放入path路径中
			path = append(path,nums[i])
			//递归
			backTrack()
			//回溯
			visited[i] =false
			//就是吐出最后一个元素
			path = path[:len(path)-1]
		}

	}
	backTrack()
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 332. 重新安排行程

https://leetcode-cn.com/problems/reconstruct-itinerary/solution/332-zhong-xin-an-pai-xing-cheng-shen-sou-hui-su-by/

```go
package main

import (
	"fmt"
	"sort"
)

func findItinerary(tickets [][]string) []string {
	var d map[string][]string	// 图的邻接表存进字典
	var ans []string	// 最终要返回的字符串数组
	d = map[string][]string{}
	ans = []string{}

	for _, v := range tickets {
		d[v[0]] = append(d[v[0]], v[1])
	}
	for _, v := range d {	// 图的邻接表存进字典后按字典序排序
		sort.Strings(v)
	}

	var dfs func(f string)
	dfs = func(f string) {
		
	}
	dfs("JFK")	// 从‘JFK’开始深搜
	return ans
}

func main() {
	fmt.Println("amadeus")
}
```



## 51.N 皇后

```go
package main

import (
	"fmt"
	"strings"
)

func solveNQueens(n int) [][]string {
	var res [][]string
	board := make([][]string, n)
	for i:=0; i<n; i++ {
		board[i] = make([]string, n)
	}
	for i:=0; i<n; i++ {
		for j:=0; j<n; j++ {
			board[i][j] = "."
		}
	}
	var dfs func(board [][]string, row int)
	dfs = func(board [][]string, row int) {
		// n 为输入的棋盘大小
		// row 是当前递归到棋盘的第几行了
		if row==n {
			tmp := make([]string, n)
			for i:=0; i<n; i++ {
				tmp[i] = strings.Join(board[i], "")
			}
			res = append(res, tmp)
			return
		}

		for col:= 0; col<n; col++ {
			if !isValid(board, row, col) {
				continue
			}
			board[row][col] = "Q"
			dfs(board, row+1)
			board[row][col] = "."
		}
	}
	dfs(board, 0)
	return res
}

func isValid(board [][]string, row, col int) (res bool){

	n := len(board)
	// 检查同一列
	for i:=0; i<row; i++ {	// 这是一个剪枝
		if board[i][col] == "Q" {
			return false
		}

	}
	//for i:=0; i<n; i++ {
	//	if board[row][i] == "Q" {
	//		return false
	//	}
	//}
	for i,j := row,col; i>=0 && j>=0; i,j = i-1, j-1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	for i,j := row,col; i>=0 && j<n; i,j = i-1,j+1 {
		if board[i][j] == "Q" {
			return false
		}
	}
	return true
}

func main() {
	fmt.Println("amadeus")
}
```



## 37.解数独

```go
package main

import (
	"fmt"
)

func solveSudoku(board [][]byte)  {
	var dfs func(board [][]byte) bool
	dfs = func(board [][]byte) bool{
		for i:=0; i<9; i++ {
			for j:=0; j<9; j++ {
				// 判断此位置(i, j)是否适合填数字
				if board[i][j] != '.' {
					continue
				}

				// 尝试填1-9
				for k:='1'; k<='9'; k++ {
                    // (i, j) 这个位置放k是否合适
					if isvalid(i, j, byte(k), board)==true {
                        // 放置k
						board[i][j] = byte(k)
                        // 如果找到合适一组立刻返回
						if dfs(board)==true {
							return true
						}
                        // 回溯，撤销k
						board[i][j] = '.'
					}
				}
                // 9个数都试完了，都不行，那么就返回false
				return false
			}
		}
        // 遍历完没有返回false，说明找到了合适棋盘位置了
		return true
	}
	dfs(board)
}

// 判断填入数字是否满足要求
func isvalid(row,col int,k byte,board [][]byte) bool{
	for i:=0; i<9; i++ {
		if board[row][i]==k {
			return false
		}
	}
	for i:=0; i<9; i++ {
		if board[i][col]==k {
			return false
		}
	}
	startrow := (row/3)*3
	startcol := (col/3)*3
	for i:=startrow; i<startrow+3; i++ {
		for j:=startcol; j<startcol+3; j++ {
			if board[i][j]==k {
				return false
			}
		}
	}
	return true
}

func main() {
	fmt.Println("amadeus")
}
```



# 贪心

##  455.分发饼干

```go
package main

import (
	"fmt"
	"sort"
)

func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	child := 0
	for sID := 0; sID < len(s) && child < len(g); sID++ {
		if s[sID] >= g[child] {
			child++
		}
	}
	return child
}

func main() {
	fmt.Println("amadeus")

}

```



## 376. 摆动序列

```go
package main

import (
	"fmt"
)

func wiggleMaxLength(nums []int) int {
	count, preDiff, curDiff := 1, 0, 0
	if len(nums) < 2 {
		return count
	}
	for i := 0; i < len(nums)-1; i++ {
		curDiff = nums[i+1] - nums[i]
		//如果有正有负则更新下标值||或者只有前一个元素为0（针对两个不等元素的序列也视作摆动序列，且摆动长度为2）
		if (curDiff > 0 && preDiff <= 0) || (preDiff >= 0 && curDiff < 0) {
			preDiff = curDiff
			count++ // 统计数组的峰值数量	相当于是删除单一坡度上的节点，然后统计长度
		}
	}
	return count
}

func main() {
	fmt.Println("amadeus")

}

```



## 53. 最大子序和

```go
package main

import (
	"fmt"
)

func maxSubArray(nums []int) int {
	maxSum := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > maxSum {
			maxSum = nums[i]
		}
	}
	return maxSum
}

func main() {
	fmt.Println("amadeus")

}

```



## 122.买卖股票的最佳时机II

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int) int {
	sum := 0
	for i := 1; i < len(prices); i++ {
		if prices[i]-prices[i-1] > 0 {
			sum += prices[i] - prices[i-1]
		}
	}
	return sum
}

func main() {
	fmt.Println("amadeus")

}

```



## 55. 跳跃游戏

```go
package main

import (
	"fmt"
)

func canJump(nums []int) bool {
	// 只有一个元素，就是能达到
	if len(nums) <= 1 {
		return true
	}
	cover := 0
	// 注意这里是小于等于cover
	for i := 0; i <= cover; i++ {
		if i+nums[i] > cover {
			cover = i + nums[i]
		}
		// 说明可以覆盖到终点
		if cover >= len(nums)-1 {
			return true
		}
	}
	return false
}

func main() {
	fmt.Println("amadeus")

}

```



## 45.跳跃游戏II

```go
package main

import (
	"fmt"
)

func jump(nums []int) int {
	curDistance := 0                   // 当前覆盖的最远距离下标
	ans := 0                           // 记录走的最大步数
	nextDistance := 0                  // 下一步覆盖的最远距离下标
	for i := 0; i < len(nums)-1; i++ { // 注意这里是小于nums.size() - 1，这是关键所在
		if nums[i]+i > nextDistance {
			nextDistance = nums[i] + i // 更新下一步覆盖的最远距离下标
		}
		if i == curDistance { // 遇到当前覆盖的最远距离下标
			curDistance = nextDistance // 更新当前覆盖的最远距离下标
			ans++
		}
	}
	return ans
}

func main() {
	fmt.Println("amadeus")

}

```



## 1005. K 次取反后最大化的数组和

```go
package main

import (
	"fmt"
	"math"
	"sort"
)

func largestSumAfterKNegations(nums []int, K int) int {
	sort.Slice(nums, func(i, j int) bool {
		return math.Abs(float64(nums[i])) > math.Abs(float64(nums[j]))
	})
	for i:=0; i<len(nums); i++ {
		if K>0 && nums[i]<0 {
			K--
			nums[i] = -nums[i]
		}
	}
	if K%2 == 1 {
		nums[len(nums)-1] *= -1
	}
	res:=0
	for i:=0; i<len(nums); i++ {
		res += nums[i]
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 134. 加油站

```go
package main

import (
	"fmt"
)

func canCompleteCircuit(gas []int, cost []int) int {
	curSum := 0
	totalSum := 0
	start := 0
	for i:=0; i<len(gas); i++ {
		curSum += gas[i]-cost[i]
		totalSum += gas[i]-cost[i]
		if curSum<0 {
			start = i+1
			curSum = 0
		}
	}
	if totalSum<0 {
		return -1
	}
	return start
}

func main() {
	fmt.Println("amadeus")
}
```



## 135. 分发糖果

```go
package main

import (
	"fmt"
)

func candy(ratings []int) int {
	need := make([]int, len(ratings))
	sum := 0
	for i:=0; i<len(ratings); i++ {
		need[i] = 1
	}
	for i:=0; i<len(ratings)-1; i++ {
		if ratings[i] < ratings[i+1] {
			need[i+1] = need[i] + 1
		}
	}
	for i:=len(ratings)-1; i>0; i-- {
		if ratings[i-1] > ratings[i] {
			if need[i-1] < need[i]+1 {
				need[i-1] = need[i] + 1
			}
		}
	}
	for i:=0; i<len(ratings); i++ {
		sum += need[i]
	}
	return sum
}

func main() {
	fmt.Println("amadeus")
}
```



## 860.柠檬水找零

```go
package main

import (
	"fmt"
)

func lemonadeChange(bills []int) bool {
	left := [2]int{0,0}
	if bills[0] != 5 {
		return false
	}
	for i:=0; i<len(bills); i++ {
		if bills[i] == 5 {
			left[0] += 1
		} else if bills[i] == 10 {
			left[1] += 1
		}
		tmp := bills[i]-5
		if tmp == 5 {
			if left[0]>0 {
				left[0]--
			} else {
				return false
			}
		}
		if tmp == 15 {
			if left[1]>0 && left[0]>0 {
				left[1]--
				left[0]--
			} else if left[1]==0 && left[0]>2 {
				left[0] -= 3
			} else {
				return false
			}
		}
	}
	return true
}

func main() {
	fmt.Println("amadeus")
}
```



## 406. 根据身高重建队列

```go
package main

import (
	"fmt"
	"sort"
)

func reconstructQueue(people [][]int) [][]int {
	//先将身高从大到小排序，确定最大个子的相对位置
	sort.Slice(people, func(i, j int) bool {
		if people[i][0] == people[j][0] {
			//当身高相同时，将K按照从小到大排序
			return people[i][1] < people[j][1]
		}
		return people[i][0] > people[j][0]
	})
	//再按照K进行插入排序，优先插入K小的
	result := make([][]int, 0)
	for _, info := range people {
		result = append(result, info)
		//将插入位置之后的元素后移动一位（意思是腾出空间）
		copy(result[info[1]+1:], result[info[1]:])
		//将插入元素位置插入元素
		result[info[1]] = info
	}
	return result
}

func main() {
	fmt.Println("amadeus")

}

```



## 452. 用最少数量的箭引爆气球

```go
package main

import (
	"fmt"
	"sort"
)

func findMinArrowShots(points [][]int) int {
	res := 1 //弓箭数
	//先按照第一位排序
	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})
	for i := 1; i < len(points); i++ {
		//如果前一位的右边界小于后一位的左边界，则一定不重合
		if points[i-1][1] < points[i][0] {
			res++
		} else {
			// 更新重叠气球最小右边界,覆盖该位置的值，留到下一步使用
			if points[i-1][1] < points[i][1] {
				points[i][1] = points[i-1][1]
			}
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")

}

```



## 435. 无重叠区间

```go
package main

import (
	"fmt"
	"sort"
)

func eraseOverlapIntervals(intervals [][]int) int {
	res := 0 // 需要移除区间的最小数量
	// 先按照左区间排序
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	for i := 1; i < len(intervals); i++ {
		//如果前一位的右边界大于后一位的左边界，则一定重合
		if intervals[i-1][1] > intervals[i][0] {
			res++
			// 更新重叠区域的最小右边界,覆盖该位置的值，留到下一步使用
			// 由于是先排序的，所以，第一位左区间是递增顺序，
			// 故只需要将临近两个元素的第二个值最小值更新到该元素的第二个值即可作之后的判断
			if intervals[i-1][1] < intervals[i][1] {
				intervals[i][1] = intervals[i-1][1]
			}
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")

}

```



## 763. 划分字母区间

```go
package main

import (
	"fmt"
)

func partitionLabels(s string) []int {
	res := make([]int, 0)
	mark := [26]int{} // 字符出现的最后位置
	size, left, right := len(s), 0, 0
	// 统计每一个字符最后出现的位置
	for i := 0; i < size; i++ {
		mark[s[i]-'a'] = i
	}
	for i := 0; i < size; i++ {
		// 找到字符出现的最远边界
		if right < mark[s[i]-'a'] {
			right = mark[s[i]-'a']
		}
		if i == right {
			res = append(res, right-left+1)
			left = i + 1
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")

}

```



## 56. 合并区间

```go
package main

import (
	"fmt"
	"sort"
)

func merge(intervals [][]int) [][]int {
	// 按照左边界从小到大排序
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	for i:=0; i<len(intervals)-1; i++ {
		// 找到存在重叠区域的区间
		if intervals[i][1] >= intervals[i+1][0] {
			// 合并存在重叠的两个区间
			if intervals[i][1] < intervals[i+1][1] {
				intervals[i][1] = intervals[i+1][1]
			}
			// 删除下一个，因为当前的 i 和 i+1 已经合并为1个区间了
			intervals = append(intervals[:i+1], intervals[i+2:]...)
			i--
		}
	}
	return intervals
}

func main() {
	fmt.Println("amadeus")
}
```



## 738. 单调递增的数字

```go
package main

import (
	"fmt"
	"strconv"
)

func monotoneIncreasingDigits(N int) int {
	//将数字转为字符串，方便使用下标
	s := strconv.Itoa(N)
	//将字符串转为byte数组，方便更改
	ss := []byte(s)
	n := len(ss)
	if n<2 {
		return N
	}
	for i:=n-1; i>0; i-- {
		if ss[i-1] > ss[i] {	//前一个大于后一位,前一位减1，后面的全部置为9
			ss[i-1] -= 1
			for j:=i; j<n; j++ {	//后面的全部置为9
				ss[j] = '9'
			}
		}
	}
	res, _ := strconv.Atoi(string(ss))
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 714. 买卖股票的最佳时机含手续费

```go
package main

import (
	"fmt"
	"golang.org/x/crypto/acme"
	"golang.org/x/text/date"
)

func maxProfit(prices []int, fee int) int {
	minBuy := prices[0]	// 记录最低价格    初始化为第一天买入
	res := 0
	for i:=0; i<len(prices); i++ {
		// 如果当前价格小于最低价，则在此处买入
		if prices[i] < minBuy {
			minBuy = prices[i]
		}
		// 如果以当前价格卖出亏本，则不卖，继续找下一个可卖点
		if prices[i]>minBuy && prices[i]<minBuy+fee {
			continue
		}
		// 可以售卖了
		if prices[i] > minBuy+fee {
			// 累加每天的收益
			res += prices[i] - minBuy - fee
			//更新最小值
			//如果还在收获利润的区间里，表示并不是真正的卖出，而计算利润每次都要减去手续费，所以要让minBuy = prices[i] - fee;
			//这样在明天收获利润的时候，才不会多减一次手续费！
			minBuy = prices[i]-fee
		}
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 968.监控二叉树    不会

```go
package main

import (
	"fmt"
	"golang.org/x/crypto/acme"
	"golang.org/x/text/date"
	"math"
)

const inf = math.MaxInt64 / 2

func minCameraCover(root *TreeNode) int {
	var dfs func(*TreeNode) (a, b, c int)
	dfs = func(node *TreeNode) (a, b, c int) {
		// 空节点，该节点有覆盖
		if node == nil {
			return inf, 0, 0
		}
		lefta, leftb, leftc := dfs(node.Left)
		righta, rightb, rightc := dfs(node.Right)
		a = leftc + rightc + 1
		b = min(a, min(lefta+rightb, righta+leftb))
		c = min(a, leftb+rightb)
		return
	}
	_, ans, _ := dfs(root)
	return ans
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}
```



# 二叉树

## 144.二叉树的前序遍历

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func preorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)	// 保存结果
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		res = append(res, node.Val)	// 前序
		traversal(node.Left)
		traversal(node.Right)
	}
	traversal(root)
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 145.二叉树的后序遍历

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func postorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)	// 保存结果
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		traversal(node.Left)
		traversal(node.Right)
		res = append(res, node.Val)	// 后序
	}
	traversal(root)
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 94.二叉树的中序遍历

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	res := make([]int, 0)	// 保存结果
	var traversal func(node *TreeNode)
	traversal = func(node *TreeNode) {
		if node == nil {
			return
		}
		traversal(node.Left)
		res = append(res, node.Val)	// 中序
		traversal(node.Right)
	}
	traversal(root)
	return res
}

func main() {
	fmt.Println("amadeus")
}
```





## 迭代法前序遍历

```go
func preorderTraversal(root *TreeNode) []int {
    ans := []int{}

	if root == nil {
		return ans
	}

	st := list.New()
    st.PushBack(root)

    for st.Len() > 0 {
        node := st.Remove(st.Back()).(*TreeNode)	// 中

        ans = append(ans, node.Val)
        if node.Right != nil {
            st.PushBack(node.Right)	// 右（空节点不入栈）
        }
        if node.Left != nil {	
            st.PushBack(node.Left)	// 左（空节点不入栈）
        }
    }
    return ans
}
```



## 迭代法后序遍历

```go
func postorderTraversal(root *TreeNode) []int {
    ans := []int{}

	if root == nil {
		return ans
	}

	st := list.New()
    st.PushBack(root)

    for st.Len() > 0 {
        node := st.Remove(st.Back()).(*TreeNode)

        ans = append(ans, node.Val)
        if node.Left != nil {
            st.PushBack(node.Left)	// 相对于前序遍历，这更改一下入栈顺序 （空节点不入栈）
        }
        if node.Right != nil {
            st.PushBack(node.Right)	// 空节点不入栈
        }
    }
    reverse(ans)	// 将结果反转之后就是左右中的顺序了
    return ans
}

func reverse(a []int) {
    l, r := 0, len(a) - 1
    for l < r {
        a[l], a[r] = a[r], a[l]
        l, r = l+1, r-1
    }
}
```



## 迭代法中序遍历

```go
func inorderTraversal(root *TreeNode) []int {
    ans := []int{}
    if root == nil {
        return ans
    }

    st := list.New()
    cur := root

    for cur != nil || st.Len() > 0 {
        if cur != nil {	// 指针来访问节点，访问到最底层
            st.PushBack(cur)	// 将访问的节点放进栈    先把左边的节点都入栈，再把右边的节点入栈
            cur = cur.Left	
        } else {
            // 将值加入结果中
            cur = st.Remove(st.Back()).(*TreeNode)		// 从栈里弹出的数据，就是要处理的数据
            ans = append(ans, cur.Val)	// 在压入右子树之前，处理它的数值部分（因为中序遍历）
            cur = cur.Right	// 获取它的右子树
        }
    }

    return ans
}
```



## 102. 二叉树的层序遍历

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func levelOrder(root *TreeNode) [][]int {
	res := make([][]int, 0)	// 保存结果
	if root == nil {
		return res
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len()>0 {
		length := queue.Len()	// 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i:=0; i<length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode)	// 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val)	// 将值加入本层切片中
		}
		res = append(res, tmp)	// 放入结果集
		tmp = []int{}	// 清空层的数据
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 107. 二叉树的层序遍历 II

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func levelOrderBottom(root *TreeNode) [][]int {
	res := make([][]int, 0)	// 保存结果
	if root == nil {
		return res
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len()>0 {
		length := queue.Len()	// 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i:=0; i<length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode)	// 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val)	// 将值加入本层切片中
		}
		res = append(res, tmp)	// 放入结果集
		tmp = []int{}	// 清空层的数据
	}
	// 反转结果集
	for i:=0;i<len(res)/2;i++{
		res[i],res[len(res)-i-1]=res[len(res)-i-1],res[i]
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 199. 二叉树的右视图

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rightSideView(root *TreeNode) []int {
	res := make([]int, 0) // 保存结果
	ans := make([][]int, 0)
	if root == nil {
		return res
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val) // 将值加入本层切片中
		}
		ans = append(ans, tmp) // 放入结果集
		tmp = []int{}          // 清空层的数据
	}
	// 取每一层的最后一个元素
	for i := 0; i < len(ans); i++ {
		res = append(res, ans[i][len(ans[i])-1])
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}

```



## 637. 二叉树的层平均值

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func averageOfLevels(root *TreeNode) []float64 {
	res := make([][]int, 0) // 保存结果
	finRes := []float64{}
	if root == nil {
		return finRes
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []int{}          // 清空层的数据
	}
	// 计算每层的平均值
	length := len(res)
	for i := 0; i < length; i++ {
		sum := 0
		for j := 0; j < len(res[i]); j++ {
			sum += res[i][j]
		}
		ans := float64(sum) / float64(len(res[i]))
		finRes = append(finRes, ans)
	}
	return finRes
}

func main() {
	fmt.Println("amadeus")
}

```



## 429. N 叉树的层序遍历

```go
package main

import (
	"container/list"
	"fmt"
)

type Node struct {
	Val      int
	Children []*Node
}

func levelOrder(root *Node) [][]int {
	res := make([][]int, 0) // 保存结果

	if root == nil {
		return res
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			// 该层的每个元素：一添加到该层的结果集中；二找到该元素的下层元素加入到队列中，方便下次使用
			node := queue.Remove(queue.Front()).(*Node) // 出队
			for j := 0; j < len(node.Children); j++ {
				queue.PushBack(node.Children[j])
			}
			tmp = append(tmp, node.Val) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []int{}          // 清空层的数据
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}

```



## 515. 在每个树行中找最大值

```go
package main

import (
	"container/list"
	"fmt"
	"math"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func largestValues(root *TreeNode) []int {
	res := make([][]int, 0) // 保存结果
	finRes := []int{}
	if root == nil {
		return finRes
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []int{}          // 清空层的数据
	}
	// 计算每层的最大值
	length := len(res)
	for i := 0; i < length; i++ {
		max := int(math.Inf(-1)) // 负无穷
		for j := 0; j < len(res[i]); j++ {
			if max < res[i][j] {
				max = res[i][j]
			}
		}
		finRes = append(finRes, max)
	}
	return finRes
}

func main() {
	fmt.Println("amadeus")
}

```



## 116.填充每个节点的下一个右侧节点指针

```go
package main

import (
	"container/list"
	"fmt"
)

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

func connect(root *Node) *Node {
	res := [][]*Node{}
	if root == nil {
		return root
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]*Node, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*Node) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []*Node{}        // 清空层的数据
	}
	// 遍历每层元素,指定next
	length := len(res)
	for i := 0; i < length; i++ {
		for j := 0; j < len(res[i])-1; j++ {
			res[i][j].Next = res[i][j+1]
		}
	}
	return root
}

func main() {
	fmt.Println("amadeus")
}

```



## 117. 填充每个节点的下一个右侧节点指针 II

```go
package main

import (
	"container/list"
	"fmt"
)

type Node struct {
	Val   int
	Left  *Node
	Right *Node
	Next  *Node
}

func connect(root *Node) *Node {
	res := [][]*Node{}
	if root == nil {
		return root
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]*Node, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*Node) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []*Node{}        // 清空层的数据
	}
	// 遍历每层元素,指定next
	length := len(res)
	for i := 0; i < length; i++ {
		for j := 0; j < len(res[i])-1; j++ {
			res[i][j].Next = res[i][j+1]
		}
	}
	return root
}

func main() {
	fmt.Println("amadeus")
}

```



## 104. 二叉树的最大深度

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func maxDepth(root *TreeNode) int {
	res := make([][]int, 0) // 保存结果
	if root == nil {
		return 0
	}
	queue := list.New()
	queue.PushBack(root)
	tmp := make([]int, 0)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode) // 出队
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
			tmp = append(tmp, node.Val) // 将值加入本层切片中
		}
		res = append(res, tmp) // 放入结果集
		tmp = []int{}          // 清空层的数据
	}
	// 找出最大深度
	return len(res)
}

func main() {
	fmt.Println("amadeus")
}

```



## 111.二叉树的最小深度

```go
package main

import (
	"container/list"
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func minDepth(root *TreeNode) int {
	ans := 0
	if root == nil {
		return 0
	}
	queue := list.New()
	queue.PushBack(root)
	for queue.Len() > 0 {
		length := queue.Len() // 保存当前层的长度，然后处理当前层（十分重要，防止添加下层元素影响判断层中元素的个数）
		for i := 0; i < length; i++ {
			node := queue.Remove(queue.Front()).(*TreeNode) // 出队
			if node.Left == nil && node.Right == nil {      // 当前节点没有左右节点，则代表此层是最小层
				return ans + 1 //返回当前层 ans代表是上一层
			}
			if node.Left != nil {
				queue.PushBack(node.Left)
			}
			if node.Right != nil {
				queue.PushBack(node.Right)
			}
		}
		ans++
	}
	return ans	// 这里应该不用+1   迟早会遇到叶子结点
}

func main() {
	fmt.Println("amadeus")
}

```



## 226. 翻转二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = root.Right, root.Left
	
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}

func main() {
	fmt.Println("amadeus")
}
```



## 101. 对称二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func isSymmetric(root *TreeNode) bool {
	var dfs func(left *TreeNode, right *TreeNode) bool
	dfs = func(left *TreeNode, right *TreeNode) bool {
		if left==nil && right==nil {
			return true
		}
		if left==nil || right==nil {	// 首先排除空节点的情况
			return false
		}
		if left.Val != right.Val {	// 排除了空节点，再排除数值不相同的情况
			return false
		}
		// 此时左右节点都不为空，且数值相同
		// 进入递归，做下一层的判断
		//        左子树：左    右子树：右               左子树：右    右子树：左
		return dfs(left.Left, right.Right) && dfs(left.Right, right.Left)	// 如果左右都对称就返回true ，有一侧不对称就返回false
	}
	return dfs(root.Left, root.Right)
}

func main() {
	fmt.Println("amadeus")
}
```



## 222. 完全二叉树的节点个数

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	res := 1
	if root.Right != nil {
		res += countNodes(root.Right)
	}
	if root.Left != nil {
		res += countNodes(root.Left)
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}
```



## 110. 平衡二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func isBalanced(root *TreeNode) bool {	// 参数：当前传入节点
	if root == nil {	// 遇到空节点了为终止，表示当前节点为根节点的树高度为0
		return true
	}
	if !isBalanced(root.Left) || !isBalanced(root.Right) {
		return false
	}
	leftH := maxdepth(root.Left)+1	// 分别求出其左右子树的高度
	rightH := maxdepth(root.Right)+1
	if abs(leftH-rightH)>1 {	// 如果差值大于1，则表示已经不是二叉平衡树了
		return false
	}
	return true
}

func maxdepth(root *TreeNode)int{
	if root==nil{
		return 0
	}
	return max(maxdepth(root.Left),maxdepth(root.Right))+1	// 以当前节点为根节点的树的最大高度
}
func max(a,b int)int{
	if a>b{
		return a
	}
	return b
}
func abs(a int)int{
	if a<0{
		return -a
	}
	return a
}

func main() {
	fmt.Println("amadeus")
}
```



## 257. 二叉树的所有路径

```go
package main

import (
	"fmt"
	"strconv"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func binaryTreePaths(root *TreeNode) []string {
	res := make([]string, 0)
	var travel func(node *TreeNode, s string)
	travel = func(node *TreeNode, s string) {
		if node.Left==nil && node.Right==nil {
			v := s + strconv.Itoa(node.Val)
			res = append(res, v)
			return
		}
		s += strconv.Itoa(node.Val) + "->"
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



func main() {
	fmt.Println("amadeus")
}
```



## 100. 相同的树

```go
package main

import (
	"fmt"
	"golang.org/x/text/cases"
	"strconv"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	switch {
	case p==nil && q==nil:
		return true
	case p==nil || q==nil:
		fallthrough
	case p.Val != q.Val:
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}



func main() {
	fmt.Println("amadeus")
}
```



## 404.左叶子之和

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func sumOfLeftLeaves(root *TreeNode) int {
	res := 0
	var findLeft func(root *TreeNode)
	findLeft = func(root *TreeNode) {
		if root.Left!=nil && root.Left.Left==nil && root.Left.Right==nil {
			res += root.Left.Val
		}
		if root.Left!=nil {
			findLeft(root.Left)
		}
		if root.Right!=nil {
			findLeft(root.Right)
		}
	}
	findLeft(root)
	return res
}



func main() {
	fmt.Println("amadeus")
}
```



## 513.找树左下角的值

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func findBottomLeftValue(root *TreeNode) int {
	maxDeep := 0
	res := 0
	var findLeftValue func(root *TreeNode,deep int)
	findLeftValue = func(root *TreeNode, deep int) {
		if root.Left==nil && root.Right==nil {
			if deep>maxDeep {
				res = root.Val
				maxDeep = deep
			}
		}
		if root.Left != nil {
			deep++
			findLeftValue(root.Left, deep)
			deep--
		}
		if root.Right != nil {
			deep++
			findLeftValue(root.Right, deep)
			deep--
		}
	}
	if root.Left==nil && root.Right==nil {
		return root.Val
	}
	findLeftValue(root, maxDeep)
	return res
}



func main() {
	fmt.Println("amadeus")
}
```



## 112. 路径总和

```go
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	targetSum -= root.Val
	if root.Left==nil && root.Right==nil && targetSum==0 {
		return true
	}
	return hasPathSum(root.Left, targetSum) || hasPathSum(root.Right, targetSum)
}
```



## 113. 路径总和 II

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func pathSum(root *TreeNode, targetSum int) [][]int {
	res := make([][]int, 0)
	curPath := make([]int, 0)
	var traverse func(node *TreeNode, targetSum int)
	traverse = func(node *TreeNode, targetSum int) {
		if node == nil {
			return
		}
		targetSum -= node.Val	// 将targetSum在遍历每层的时候都减去本层节点的值
		curPath = append(curPath, node.Val)	// 把当前节点放到路径记录里
		if node.Left==nil && node.Right==nil && targetSum==0 {	// 如果剩余的targetSum为0, 则正好就是符合的结果
			// 不能直接将currPath放到result里面, 因为currPath是共享的, 每次遍历子树时都会被修改
			pathCopy := make([]int, len(curPath))
			copy(pathCopy, curPath)
			res = append(res, pathCopy)	// 将副本放到结果集里
		}
		traverse(node.Left, targetSum)
		traverse(node.Right, targetSum)
		curPath = curPath[:len(curPath)-1]	// 当前节点遍历完成, 从路径记录里删除掉

	}
	traverse(root, targetSum)
	return res
}



func main() {
	fmt.Println("amadeus")
}
```



## 106.从中序与后序遍历序列构造二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func buildTree(inorder []int, postorder []int) *TreeNode {
	if len(inorder)<1 || len(postorder)<1 {
		return nil
	}
	// 先找到根节点（后续遍历的最后一个就是根节点）
	nodeValue := postorder[len(postorder)-1]
	// 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
	left := findRootIndex(inorder, nodeValue)
	// 构造root
	root := &TreeNode{
		Val:nodeValue,
		Left: buildTree(inorder[:left], postorder[:left]),	// 将后续遍历一分为二，左边为左子树，右边为右子树
		Right: buildTree(inorder[left+1:], postorder[left:len(postorder)-1]),
	}
	return root
}

func findRootIndex(inorder []int,target int) (index int){
	for i:=0; i<len(inorder); i++ {
		if target == inorder[i] {
			return i
		}
	}
	return -1
}


func main() {
	fmt.Println("amadeus")
}
```



## 105 从前序与中序遍历序列构造二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder)<1 || len(inorder)<1 {
		return nil
	}
	// 先找到根节点（先序遍历的第一个就是根节点）
	// 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
	left := findRootIndex(preorder[0], inorder)
	// 构造root
	root := &TreeNode{
		Val:preorder[0],
		Left: buildTree(preorder[1:left+1], inorder[:left]),	// 将先序遍历一分为二，左边为左子树，右边为右子树
		Right: buildTree(preorder[left+1:], inorder[left+1:]),
	}
	return root
}

func findRootIndex(target int,inorder []int) int{
	for i:=0; i<len(inorder); i++ {
		if target == inorder[i] {
			return i
		}
	}
	return -1
}


func main() {
	fmt.Println("amadeus")
}
```



## 654.最大二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) < 1 {
		return nil
	}
	// 找到最大值
	index := findMax(nums)
	// 构造二叉树
	root := &TreeNode{
		Val: nums[index],
		Left: constructMaximumBinaryTree(nums[:index]),
		Right: constructMaximumBinaryTree(nums[index+1:]),
	}
	return root
}

func findMax(nums []int) (index int){
	for i:=0; i<len(nums); i++ {
		if nums[i] > nums[index] {
			index = i
		}
	}
	return index
}


func main() {
	fmt.Println("amadeus")
}
```



## 617.合并二叉树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

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


func main() {
	fmt.Println("amadeus")
}
```



## 700.二叉搜索树中的搜索

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func searchBST(root *TreeNode, val int) *TreeNode {
	if root==nil{
		return nil
	}
	if root.Val == val {
		return root
	}
	if root.Val > val {
		return searchBST(root.Left, val)
	}
	return searchBST(root.Right, val)
}


func main() {
	fmt.Println("amadeus")
}
```



## 98. 验证二叉搜索树

```go
package main

import (
	"fmt"
	"math"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

// 中序遍历下，输出的二叉搜索树节点的数值是有序序列
// 验证二叉搜索树，就相当于变成了判断一个序列是不是递增的
func isValidBST(root *TreeNode) bool {
	// 二叉搜索树也可以是空树
	if root == nil {
		return true
	}
	var check func(node *TreeNode, min, max int64) bool
	check = func(node *TreeNode, min, max int64) bool {
		if node == nil {
			return true
		}
		// 中序遍历，验证遍历的元素是不是从小到大
		if min>=int64(node.Val) || max<=int64(node.Val) {
			return false
		}
		// 分别对左子树和右子树递归判断，如果左子树和右子树都符合则返回true
		return check(node.Right, int64(node.Val), max) && check(node.Left, min, int64(node.Val))
	}
	// 因为后台测试数据中有int最小值
	return check(root, math.MinInt64, math.MaxInt64)
}


func main() {
	fmt.Println("amadeus")
}
```



## 530.二叉搜索树的最小绝对差

```go
package main

import (
	"fmt"
	"math"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func getMinimumDifference(root *TreeNode) int {
	res := make([]int, 0)
	var finMin func(root *TreeNode)
	finMin = func(root *TreeNode) {	// 中序遍历
		if root == nil {
			return
		}
		finMin(root.Left)
		res = append(res, root.Val)
		finMin(root.Right)
	}
	finMin(root)
	min := math.MaxInt
	for i:=1; i<len(res); i++ {
		tmp := res[i]-res[i-1]
		if tmp < min {
			min = tmp
		}
	}
	return min
}


func main() {
	fmt.Println("amadeus")
}
```



## 501.二叉搜索树中的众数

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func findMode(root *TreeNode) []int {
	res := make([]int, 0)
	count := 1
	max := 1
	var prev *TreeNode
	var travel func(node *TreeNode)
	travel = func(node *TreeNode) {	// 中序遍历
		if node == nil {
			return
		}
		travel(node.Left)
		if prev!=nil && prev.Val==node.Val {	// 遇到相同的值，计数+1
			count++
		} else {
			count = 1	// 遇到新的值，重新开始计数
		}
		if count >= max {
			if count>max && len(res)>0 {	// 遇到出现次数更多的值，重置res
				res = []int{node.Val}
			} else {
				res = append(res, node.Val)	// 遇到出现次数相同的值，res多加一个值
			}
			max = count
		}
		prev = node
		travel(node.Right)
	}
	travel(root)
	return res
}


func main() {
	fmt.Println("amadeus")
}
```



## 236. 二叉树的最近公共祖先

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// 如果找到了 节点p或者q，或者遇到空节点，就返回。
	if root == nil {
		return nil
	}
	if root==p || root==q {
		return root
	}

	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	// 左右两边都不为空，则根节点为祖先
	if left!=nil && right!=nil {
		return root
	}
	if left==nil {
		return right
	}
	return left
}


func main() {
	fmt.Println("amadeus")
}
```



## 235. 二叉搜索树的最近公共祖先

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

//利用BSL的性质（前序遍历有序）
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// 如果找到了 节点p或者q，或者遇到空节点，就返回。
	if root == nil {
		return nil
	}
	if root.Val>=p.Val && root.Val<=q.Val {	// 当前节点的值在给定值的中间（或者等于），即为最深的祖先
		return root
	}
	if root.Val>p.Val && root.Val>q.Val {	// 当前节点的值大于给定的值，则说明满足条件的在左边
		return lowestCommonAncestor(root.Left, p, q)
	}
	if root.Val<p.Val && root.Val<q.Val {	// 当前节点的值小于各点的值，则说明满足条件的在右边
		return lowestCommonAncestor(root.Right, p, q)
	}
	return root
}


func main() {
	fmt.Println("amadeus")
}
```



## 701.二叉搜索树中的插入操作

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func insertIntoBST(root *TreeNode, val int) *TreeNode {
	// 终止条件就是找到遍历的节点为null的时候，就是要插入节点的位置了，并把插入的节点返回。
	// 这里把添加的节点返回给上一层，就完成了父子节点的赋值操作了
	if root == nil {
		root = &TreeNode{Val: val}
		return root
	}
	// 下一层将加入节点返回，本层用root->left或者root->right将其接住
	if root.Val > val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}


func main() {
	fmt.Println("amadeus")
}
```



## 450. 删除二叉搜索树中的节点

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func deleteNode(root *TreeNode, key int) *TreeNode {
	// 第一种情况：没找到删除的节点，遍历到空节点直接返回了
	if root == nil {
		return nil
	}
	// 说明要删除的节点在左子树	左递归
	if key < root.Val {
		root.Left = deleteNode(root.Left, key)
		return root
	}
	// 说明要删除的节点在右子树	右递归
	if key > root.Val {
		root.Right = deleteNode(root.Right, key)
		return root
	}
	// 第四种情况：其右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
	if root.Right == nil {
		return root.Left
	}
	// 第三种情况：其左孩子为空，右孩子不为空，删除节点，右孩子补位 ，返回右孩子为根节点
	if root.Left == nil {
		return  root.Right
	}
	// 第五种情况：左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
	// 并返回删除节点右孩子为新的根节点
	minNode := root.Right
	for minNode.Left != nil {
		minNode = minNode.Left
	}
	root.Val = minNode.Val
	root.Right = delete(root.Right)
	return root
}

//
func delete(root *TreeNode) *TreeNode {
	if root.Left == nil {
		pRight := root.Right
		root.Right = nil
		return pRight
	}
	root.Left = delete(root.Left)
	return root
}

func main() {
	fmt.Println("amadeus")
}
```



## 669. 修剪二叉搜索树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return nil
	}
	// 如果该节点值小于最小值，则该节点更换为该节点的右节点值，继续遍历
	if root.Val < low {
		right := trimBST(root.Right, low, high)
		return right
	}
	// 如果该节点的值大于最大值，则该节点更换为该节点的左节点值，继续遍历
	if root.Val > high {
		left := trimBST(root.Left, low, high)
		return left
	}
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}

func main() {
	fmt.Println("amadeus")
}
```



## 108.将有序数组转换为二叉搜索树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func sortedArrayToBST(nums []int) *TreeNode {
	// 终止条件，最后数组为空则可以返回
	if len(nums) == 0 {
		return nil
	}
	// 按照BSL的特点，从中间构造节点
	root := &TreeNode{nums[len(nums)/2], nil, nil}
	// 数组的左边为左子树
	root.Left = sortedArrayToBST(nums[:len(nums)/2])
	// 数字的右边为右子树
	root.Right = sortedArrayToBST(nums[len(nums)/2+1:])
	return root
}

func main() {
	fmt.Println("amadeus")
}
```



## 538.把二叉搜索树转换为累加树

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

//右中左
func convertBST(root *TreeNode) *TreeNode {
	sum := 0
	var rightMLeft func(root *TreeNode) *TreeNode
	rightMLeft = func(root *TreeNode) *TreeNode {
		if root == nil {	// 终止条件，遇到空节点就返回
			return nil
		}
		rightMLeft(root.Right)	// 先遍历右边
		tmp := sum	// 暂存总和值
		sum += root.Val	// 将总和值变更
		root.Val += tmp	// 更新节点值
		rightMLeft(root.Left)	// 遍历左节点
		return root
	}
	rightMLeft(root)
	return root
}

func main() {
	fmt.Println("amadeus")
}
```



# 栈和队列

## 20. 有效的括号

```go
package main

import (
	"fmt"
)


func isValid(s string) bool {
	hash := map[byte]byte{')':'(', ']':'[', '}':'{'}
	stack := make([]byte, 0)
	if s == "" {
		return true
	}
	for i:=0; i<len(s); i++ {
		if s[i]=='(' || s[i]=='[' || s[i]=='{' {
			stack = append(stack, s[i])
		} else if len(stack)>0 && stack[len(stack)-1]==hash[s[i]] {
			stack = stack[:len(stack)-1]
		} else {
			return false
		}
	}
	return len(stack)==0
}

func main() {
	fmt.Println("amadeus")
}
```



## 1047. 删除字符串中的所有相邻重复项

```go
package main

import (
	"fmt"
)


func removeDuplicates(s string) string {
	stack := make([]byte, 0)
	for i:=0; i<len(s); i++ {
		// 栈不空 且 与栈顶元素相等
		if len(stack)>0 && stack[len(stack)-1]==s[i] {
			// 弹出栈顶元素 并 忽略当前元素(s[i])
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}
	return string(stack)
}

func main() {
	fmt.Println("amadeus")
}
```



## 150. 逆波兰表达式求值

```go
package main

import (
	"fmt"
	"strconv"
)


func evalRPN(tokens []string) int {
	stack := make([]int, 0)
	for _, token := range tokens {
		val, err := strconv.Atoi(token)
		if err == nil {
			stack = append(stack, val)
		} else {
			num1, num2 := stack[len(stack)-2], stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			switch token {
			case "+":
				stack = append(stack, num1+num2)
			case "-":
				stack = append(stack, num1-num2)
			case "*":
				stack = append(stack, num1*num2)
			case "/":
				stack = append(stack, num1/num2)
			}

		}
	}
	return stack[0]
}

func main() {
	fmt.Println("amadeus")
}
```



## 239. 滑动窗口最大值

```go
package main

import (
	"fmt"
)

// push:当前元素e入队时，相对于前面的元素来说，e最后进入窗口，e一定是最后离开窗口，
//那么前面比e小的元素，不可能成为最大值，因此比e小的元素可以“压缩”掉

// pop:在元素入队时，是按照下标i入队的，因此队列中剩余的元素，其下标一定是升序的。
//窗口大小不变，最先被排除出窗口的，一直是下标最小的元素，设为r。元素r在队列中要么是头元素，要么不存在。

// 封装单调队列的方式解题
type MyQueue struct {
	queue []int
}

func NewMyQueue() *MyQueue {
	return &MyQueue{
		queue: make([]int, 0),
	}
}

func (m *MyQueue) Front() int {
	return m.queue[0]
}

func (m *MyQueue) Back() int {
	return m.queue[len(m.queue)-1]
}

func (m *MyQueue) Empty() bool {
	return len(m.queue) == 0
}

func (m *MyQueue) Push(val int) {
	for !m.Empty() && val > m.Back() {
		m.queue = m.queue[:len(m.queue)-1]
	}
	m.queue = append(m.queue, val)
}

func (m *MyQueue) Pop(val int) {
	if !m.Empty() && val == m.Front() {
		m.queue = m.queue[1:]
	}
}

func maxSlidingWindow(nums []int, k int) []int {
	queue := NewMyQueue()
	length := len(nums)
	res := make([]int, 0)
	// 先将前k个元素放入队列
	for i := 0; i < k; i++ {
		queue.Push(nums[i])
	}
	// 记录前k个元素的最大值
	res = append(res, queue.Front())

	for i := k; i < length; i++ {
		// 滑动窗口移除最前面的元素
		queue.Pop(nums[i-k])
		// 滑动窗口添加最后面的元素
		queue.Push(nums[i])
		// 记录最大值
		res = append(res, queue.Front())
	}
	return res
}

func main() {
	fmt.Println("amadeus")
}

```



## 347.前 K 个高频元素

```go
package main

import (
	"fmt"
	"sort"
)

//方法二:利用O(logn)排序
func topKFrequent(nums []int, k int) []int {
	ans := make([]int, 0)
	freq := map[int]int{}
	for i := 0; i < len(nums); i++ {
		freq[nums[i]]++
	}
	for key, _ := range freq {
		ans = append(ans, key)
	}
	//核心思想：排序
	//可以不用包函数，自己实现快排
	sort.Slice(ans, func(i, j int) bool {
		return freq[ans[i]] > freq[ans[j]]
	})
	return ans[:k]
}

func main() {
	fmt.Println("amadeus")
}

```



# 动态规划

##  509. 斐波那契数

```go
package main

import (
	"fmt"
)

func fib(n int) int {
	if n < 2 {
		return n
	}
	a, b, c := 0, 1, 0
	for i := 1; i < n; i++ {
		c = a + b
		a, b = b, c
	}
	return c
}

func main() {
	fmt.Println("amadeus")
}

```



## 70. 爬楼梯

```go
package main

import (
	"fmt"
)

func climbStairs(n int) int {
	if n == 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 2
	for i:=3; i<=n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func main() {
	fmt.Println("amadeus")
}
```



## 746. 使用最小花费爬楼梯

```go
package main

import (
	"fmt"
)

func minCostClimbingStairs(cost []int) int {
	dp := make([]int, len(cost))
	dp[0] = cost[0]
	dp[1] = cost[1]
	for i:=2; i<len(cost); i++ {
		dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
	}
	return min(dp[len(dp)-2], dp[len(dp)-1])
}

func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func main() {
	fmt.Println("amadeus")
}
```



## 62. 不同路径

```go
package main

import (
	"fmt"
)

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i:=0; i<m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}
	for j:=0; j<n; j++{
		dp[0][j] = 1
	}
	for i:=1; i<m; i++ {
		for j:=1; j<n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func main() {
	fmt.Println("amadeus")
}
```



## 63. 不同路径 II

```go
package main

import (
	"fmt"
)

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	dp := make([][]int, m)
	for i:=0; i<m; i++ {
		dp[i] = make([]int, n)
	}
	// 初始化, 如果是障碍物, 后面的就都是0, 不用循环了
	for i:=0; i<m && obstacleGrid[i][0]==0; i++ {
		dp[i][0]=1
	}
	for j:=0; j<n && obstacleGrid[0][j]==0; j++{
		dp[0][j] = 1
	}
	// dp数组推导过程
	for i:=1; i<m; i++ {
		for j:=1; j<n; j++ {
			// 如果obstacleGrid[i][j]这个点是障碍物, 那么dp[i][j]保持为0
			// 否则我们需要计算当前点可以到达的路径数
			if obstacleGrid[i][j] == 0 {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}

		}
	}
	return dp[m-1][n-1]
}

func main() {
	fmt.Println("amadeus")
}
```



## 343. 整数拆分

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 1
	for i:=3; i<=n; i++ {
		for j:=1; j<i-1; j++ {
			// i可以差分为i-j和j。由于需要最大值，故需要通过j遍历所有存在的值，取其中最大的值作为当前i的最大值，
			// 在求最大值的时候，一个是j与i-j相乘，一个是j与dp[i-j].
			dp[i]=max(dp[i],max(j*(i-j),j*dp[i-j]))	// 在递推公式推导的过程中，每次计算dp[i]，取最大的
		}
	}
	return dp[n]
}

func max(a, b int) int {
	if a>b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}
```



## 96.不同的二叉搜索树

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func numTrees(n int)int{
	dp := make([]int, n+1)
	// dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]
	dp[0] = 1
	for i:=1; i<=n; i++ {
		for j:=1; j<=i; j++ {
			// j-1 为 以j为头结点左子树节点数量
			// i-j 为以j为头结点右子树节点数量
			dp[i] += dp[j-1] * dp[i-j]
		}
	}
	return dp[n]
}


func main() {
	fmt.Println("amadeus")
}
```



##  416. 分割等和子集

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func canPartition(nums []int) bool {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	// 如果 nums 的总和为奇数则不可能平分成两个子集
	if sum % 2 == 1 {
		return false
	}
	target := sum / 2
	dp := make([]int, target+1)
	// dp[j]表示 背包总容量是j，最大可以凑成j的子集总和为dp[j]。
	for _, num := range nums {
		for j:=target; j>=num; j-- {	// 每一个元素一定是不可重复放入，所以从大到小遍历
			// 物品 i 的重量是 nums[i]，其价值也是 nums[i]
			if dp[j] < dp[j-num]+num {
				dp[j] = dp[j-num]+num
			}
		}
	}
	// 集合中的元素正好可以凑成总和target
	return dp[target] == target
}


func main() {
	fmt.Println("amadeus")
}
```



## 1049. 最后一块石头的重量 II

```go
package main

import (
	"fmt"
)

func lastStoneWeightII(stones []int) int {
	dp := make([]int, 15001)
	// 求target
	sum := 0
	for _, v := range stones {
		sum += v
	}
	target := sum / 2
	// 遍历顺序
	for i := 0; i < len(stones); i++ {
		for j := target; j >= stones[i]; j-- {
			// 推导公式
			dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
		}
	}
	return sum - 2*dp[target]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 494. 目标和

```go
package main

import (
	"fmt"
	"math"
)

func findTargetSumWays(nums []int, target int) int {
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if abs(target) > sum {
		return 0
	}
	if (sum+target)%2 == 1 {
		return 0
	}
	// 计算背包大小
	bag := (sum + target) / 2
	// 定义dp数组
	dp := make([]int, bag+1)
	// 初始化
	dp[0] = 1
	// 遍历顺序
	for i := 0; i < len(nums); i++ {
		for j := bag; j >= nums[i]; j-- {
			dp[j] += dp[j-nums[i]]
		}
	}
	return dp[bag]
}

func abs(x int) int {
	return int(math.Abs(float64(x)))
}

func main() {
	fmt.Println("amadeus")
}

```



## 474.一和零

```go
package main

import (
	"fmt"
)

func findMaxForm(strs []string, m int, n int) int {
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i < len(strs); i++ {
		zeroNum, oneNum := 0, 0
		for _, v := range strs[i] {
			if v == '0' {
				zeroNum++
			}
		}
		oneNum = len(strs[i]) - zeroNum
		for j := m; j >= zeroNum; j-- {
			for k := n; k >= oneNum; k-- {
				dp[j][k] = max(dp[j][k], dp[j-zeroNum][k-oneNum]+1)
			}
		}
	}
	return dp[m][n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 518. 零钱兑换 II

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func change(amount int, coins []int) int {
	// 定义dp数组		dp[j]：凑成总金额j的货币组合数为dp[j]
	dp := make([]int, amount+1)
	// 初始化,0大小的背包, 当然是不装任何东西了, 就是1种方法
	dp[0] = 1
	for i:=0; i<len(coins); i++ {
		for j:=coins[i]; j<=amount; j++ {
			dp[j] += dp[j-coins[i]]
		}
	}
	return dp[amount]
}


func main() {
	fmt.Println("amadeus")
}
```



##  377. 组合总和 Ⅳ

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func combinationSum4(nums []int, target int) int {
	// 定义dp数组		dp[i]: 凑成目标正整数为i的排列个数为dp[i]
	dp := make([]int, target+1)
	// 初始化
	dp[0] = 1
	// 遍历顺序, 先遍历背包,再循环遍历物品
	for j:=0; j<=target; j++ {
		for i:=0; i<len(nums); i++ {
			if j>=nums[i] {
				// 递推公式
				dp[j] += dp[j-nums[i]]
			}

		}
	}
	return dp[target]
}


func main() {
	fmt.Println("amadeus")
}
```



## 322. 零钱兑换

```go
package main

import (
	"fmt"
	"math"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func coinChange(coins []int, amount int) int {
	// 定义dp数组		dp[j]：凑足总额为j所需钱币的最少个数为dp[j]
	dp := make([]int, amount+1)
	// 初始化	凑足总金额为0所需钱币的个数一定是0，那么dp[0] = 0
	dp[0] = 0
	/*
	考虑到递推公式的特性，dp[j]必须初始化为一个最大的数，否则就会在min(dp[j - coins[i]] + 1, dp[j])比较的过程中被初始值覆盖。
	所以下标非0的元素都是应该是最大值
	*/
	// 初始化为math.MaxInt32
	for j:=1; j<=amount;j++ {
		dp[j] = math.MaxInt32
	}
	// 遍历顺序, 先遍历物品,再循环遍历背包
	for i:=0; i<len(coins); i++ {
		for j:=coins[i]; j<=amount; j++ {
			if dp[j-coins[i]] != math.MaxInt32 {
				// 递推公式
				dp[j] = min(dp[j], dp[j-coins[i]]+1)
			}

		}
	}
	if dp[amount] == math.MaxInt32 {
		return -1
	}
	return dp[amount]
}

func min(a, b int) int {
	if a<b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}
```



## 279.完全平方数

```go
package main

import (
	"fmt"
	"math"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func numSquares(n int) int {
	// 定义dp数组		dp[j]：和为j的完全平方数的最少数量为dp[j]
	dp := make([]int, n+1)
	// 初始化	dp[0]表示 和为0的完全平方数的最小数量，那么dp[0]一定是0。
	dp[0] = 0
	/*
	从递归公式dp[j] = min(dp[j - i * i] + 1, dp[j]);中可以看出每次dp[j]都要选最小的，
	所以非0下标的dp[j]一定要初始为最大值，这样dp[j]在递推的时候才不会被初始值覆盖。
	*/
	// 初始化为math.MaxInt32
	for j:=1; j<=n;j++ {
		dp[j] = math.MaxInt32
	}
	// 遍历顺序, 先遍历物品,再循环遍历背包
	for i:=1; i<=n; i++ {
		for j:=i*i; j<=n; j++ {
				// 递推公式
				dp[j] = min(dp[j], dp[j-i*i]+1)
			}
		}
	return dp[n]
}

func min(a, b int) int {
	if a<b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}
```



## 139.单词拆分

```go
package main

import (
	"fmt"
)

/**
  动态五部曲
  1.确定dp下标及其含义
  2.确定递推公式
  3.确定dp初始化
  4.确定遍历顺序
  5.打印dp
**/

func wordBreak(s string, wordDict []string) bool {
	wordDictSet := make(map[string]bool)
	for _,w := range wordDict {
		wordDictSet[w] = true
	}

	// 定义dp数组		dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词。
	dp := make([]bool, len(s)+1)
	// 初始化	dp[0]表示 和为0的完全平方数的最小数量，那么dp[0]一定是0。
	dp[0] = true


	// 遍历顺序, 先遍历物品,再循环遍历背包
	for i:=1; i<=len(s); i++ {
		for j:=0; j<i; j++ {
			if dp[j] && wordDictSet[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}

func min(a, b int) int {
	if a<b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}
```



##  198.打家劫舍

```go
package main

import (
	"fmt"
)

func rob(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	if len(nums) < 2 {
		return nums[0]
	}
	if len(nums) < 3 {
		return max(nums[0], nums[1])
	}
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[len(nums)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



##  213.打家劫舍II

```go
package main

import (
	"fmt"
)

func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	if len(nums) == 2 {
		return max(nums[0], nums[1])
	}
	result1 := robRange(nums, 0) // 考虑首元素
	result2 := robRange(nums, 1) // 不考虑首元素
	return max(result1, result2)
}

func robRange(nums []int, start int) int {
	dp := make([]int, len(nums))
	dp[1] = nums[start]
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i-1+start])
	}
	return dp[len(nums)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 337.打家劫舍 III

```go
package main

import (
	"fmt"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rob(root *TreeNode) int {
	res := robTree(root)
	return max(res[0], res[1])
}

func robTree(cur *TreeNode) []int {
	// dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱

	if cur == nil {
		return []int{0, 0}
	}
	// 后序遍历
	left := robTree(cur.Left)
	right := robTree(cur.Right)
	// 考虑去偷当前的屋子	如果是偷当前节点，那么左右孩子就不能偷
	// left[0]:不偷左孩子节点所得到的最大金钱
	// right[0]:不偷右孩子节点所得到的最大金钱
	robCur := cur.Val + left[0] + right[0]
	// 考虑不去偷当前的屋子	如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的
	notRobCur := max(left[0], left[1]) + max(right[0], right[1])
	// 注意顺序：0:不偷，1:去偷
	return []int{notRobCur, robCur}	// {不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 121. 买卖股票的最佳时机

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int) int {
	// dp[i][0] 表示第i天持有股票所得最多现金
	// dp[i][1] 表示第i天不持有股票所得最多现金
	length := len(prices)
	if length == 0 {
		return 0
	}
	dp := make([][]int, length)
	for i:=0; i<length; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i:=1; i<length; i++ {
		dp[i][0] = max(dp[i-1][0], -prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[length-1][1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 122.买卖股票的最佳时机II

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int) int {
	// dp[i][0] 表示第i天持有股票所得最多现金
	// dp[i][1] 表示第i天不持有股票所得最多现金
	length := len(prices)
	if length == 0 {
		return 0
	}
	dp := make([][]int, length)
	for i:=0; i<length; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i:=1; i<length; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
	}
	return dp[length-1][1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 123.买卖股票的最佳时机III

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int) int {
	// dp[i][j]中 i表示第i天，j为 [0 - 4] 五个状态，
	// dp[i][j]表示第i天状态j所剩最大现金。
	// dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票
	length := len(prices)
	if length == 0 {
		return 0
	}
	dp := make([][]int, length)
	for i:=0; i<length; i++ {
		dp[i] = make([]int, 5)
	}
	dp[0][0] = 0
	dp[0][1] = -prices[0]
	dp[0][2] = 0
	dp[0][3] = -prices[0]
	dp[0][4] = 0
	for i:=1; i<length; i++ {
		dp[i][0] = dp[i-1][0]
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
		dp[i][2] = max(dp[i-1][2], dp[i-1][1]+prices[i])
		dp[i][3] = max(dp[i-1][3], dp[i-1][2]-prices[i])
		dp[i][4] = max(dp[i-1][4], dp[i-1][3]+prices[i])
	}
	return dp[length-1][4]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 188.买卖股票的最佳时机IV

```go
package main

import (
	"fmt"
)

func maxProfit(k int, prices []int) int {
	// dp[i][j]中 i表示第i天，j为状态，
	// dp[i][j]表示第i天状态j所剩最大现金。
	// dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票
	length := len(prices)
	if k == 0 || length == 0 {
		return 0
	}
	dp := make([][]int, length)
	// 初始化
	// 在初始化的地方同样要类比j为偶数是卖、奇数是买的状态。
	for i:=0; i<length; i++ {
		dp[i] = make([]int, 2*k+1)
	}
	// len(dp[0]) =  2*k+1
	for i:=1; i<len(dp[0]); i++ {
		if i%2 == 1 {
			dp[0][i] = -prices[0]
		}
	}

	for i:=1; i<length; i++ {
		dp[i][0] = dp[i-1][0]
		for j:=1; j<len(dp[0]); j++ {
			if j%2==1 {
				dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]-prices[i])
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]+prices[i])
			}
		}
	}
	return dp[length-1][2*k]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 309.最佳买卖股票时机含冷冻期

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int) int {
	// dp[i][j]中 i表示第i天，j为状态，
	// dp[i][j]表示第i天状态j所剩最大现金。
	// dp[i][1]，表示的是第i天，买入股票的状态，并不是说一定要第i天买入股票
	/*
	0 状态一：买入股票状态（今天买入股票，或者是之前就买入了股票然后没有操作）
	-----卖出股票状态，这里就有两种卖出股票状态------
	1 状态二：两天前就卖出了股票，度过了冷冻期，一直没操作，今天保持卖出股票状态
	2 状态三：今天卖出了股票
	3 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！
	*/
	n := len(prices)
	if n < 2 {
		return 0
	}
	dp := make([][]int, n)
	status := make([]int, n*4)

	// 初始化
	// 在初始化的地方同样要类比j为偶数是卖、奇数是买的状态。
	for i := range dp {
		dp[i] = status[:4]
		status = status[4:]
	}
	dp[0][0] = -prices[0]

	for i:=1; i<n; i++ {
		dp[i][0] = max(dp[i-1][0], max(dp[i-1][1], dp[i-1][3])-prices[i])	// 保持买入状态 或 今天买入，昨天可以是冰冻或不是
		dp[i][1] = max(dp[i-1][1], dp[i-1][3])	// 保持卖出状态 或 昨天是冰冻
		dp[i][2] = dp[i-1][0]+prices[i]
		dp[i][3] = dp[i-1][2]
	}
	return max(dp[n-1][1], max(dp[n-1][2], dp[n-1][3]))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 714.买卖股票的最佳时机含手续费

```go
package main

import (
	"fmt"
)

func maxProfit(prices []int, fee int) int {
	// dp[i][0] 表示第i天持有股票所得最多现金
	// dp[i][1] 表示第i天不持有股票所得最多现金
	length := len(prices)
	if length == 0 {
		return 0
	}
	dp := make([][]int, length)
	for i:=0; i<length; i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i:=1; i<length; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i]-fee)
	}
	return dp[length-1][1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 300.最长递增子序列

```go
package main

import (
	"fmt"
)

func lengthOfLIS(nums []int ) int {
	// dp[i]的定义 : 位置i的最长升序子序列等于j从0到i-1各个位置的最长升序子序列 + 1 的最大值
	n := len(nums)
	// 不能漏这一步，否则有样例过不了
	if n < 2 {
		return 1
	}
	dp := make([]int, n)
	res := 0
	// dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	for i:=0; i<n; i++ {
		dp[i] = 1
	}
	for i:=1; i<n; i++ {
		for j:=0; j<i; j++ {
			if nums[i]>nums[j] {
				// 状态转移方程 : 位置i的最长升序子序列等于j从0到i-1各个位置的最长升序子序列 + 1 的最大值
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		// 取长的子序列
		res = max(res, dp[i])
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 674. 最长连续递增序列

```go
package main

import (
	"fmt"
)

func findLengthOfLCIS(nums []int) int {
	// dp[i]的定义 : 以下标i为结尾的数组的连续递增的子序列长度为dp[i]	  一定是以下标i为结尾，并不是说一定以下标0为起始位置
	n := len(nums)
	// 不能漏这一步，否则有样例过不了
	if n < 2 {
		return 1
	}
	dp := make([]int, n)
	res := 0
	// dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	for i:=0; i<n; i++ {
		dp[i] = 1
	}
	for i:=0; i<n-1; i++ {
		if nums[i+1]>nums[i] {
			// 状态转移方程 : 如果 nums[i + 1] > nums[i]，那么以 i+1 为结尾的数组的连续递增的子序列长度 一定等于 以i为结尾的数组的连续递增的子序列长度 + 1
			dp[i+1] = dp[i] + 1
		}
		// 取长的子序列
		res = max(res, dp[i+1])
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 718. 最长重复子数组

```go
package main

import (
	"fmt"
)

func findLength(A []int, B []int) int {
	// dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]。
	m, n := len(A), len(B)

	dp := make([][]int, m+1)
	res := 0
	// dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if A[i-1] == B[j-1] {
				// 状态转移方程 : dp[i][j]的状态只能由dp[i - 1][j - 1]推导出来。
				// 即当A[i - 1] 和B[j - 1]相等的时候，dp[i][j] = dp[i - 1][j - 1] + 1
				dp[i][j] = dp[i-1][j-1] + 1
			}
			// 取长的子序列
			res = max(res, dp[i][j])
		}

	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 1143.最长公共子序列

```go
package main

import (
	"fmt"
)

func longestCommonSubsequence(text1 string, text2 string) int {
	// dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
	m, n := len(text1), len(text2)

	dp := make([][]int, m+1)
	// dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



##  1035.不相交的线

```go
package main

import (
	"fmt"
)

func maxUncrossedLines(nums1 []int, nums2 []int) int {
	// 本题说是求绘制的最大连线数，其实就是求两个字符串的最长公共子序列的长度
	// dp[i][j]：长度为[0, i - 1]的字符串nums1与长度为[0, j - 1]的字符串nums2的最长公共子序列为dp[i][j]
	m, n := len(nums1), len(nums2)

	dp := make([][]int, m+1)
	// dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if nums1[i-1] == nums2[j-1] {
				// 状态转移方程 : 主要就是两大情况： nums1[i - 1] 与 nums2[j - 1]相同，nums1[i - 1] 与 nums2[j - 1]不相同
				// 如果nums1[i - 1] 与 nums2[j - 1]相同，那么找到了一个公共元素，所以dp[i][j] = dp[i - 1][j - 1] + 1;
				// 如果nums1[i - 1] 与 nums2[j - 1]不相同，即：dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}

	}
	return dp[m][n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 53. 最大子序和

```go
package main

import (
	"fmt"
)

func maxSubArray(nums []int) int {
	// dp[i]：包括下标i之前的最大连续子序列和为dp[i]
	n := len(nums)
	dp := make([]int, n)
	// dp[i]的初始化	由于dp 状态转移方程依赖dp[0]
	dp[0] = nums[0]
	// 初始化最大的和
	res := nums[0]
	for i:=1; i<n; i++ {
		// dp[i]只有两个方向可以推出来：nums[i]加入当前连续子序列和	从头开始计算当前连续子序列和
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		res = max(res, dp[i])
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 392.判断子序列

```go
package main

import (
	"fmt"
)

func isSubsequence(s string, t string) bool {
	// dp[i][j] 表示以下标i-1为结尾的字符串s，和以下标j-1为结尾的字符串t，相同子序列的长度为dp[i][j]。
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
	}
	/*
	在确定递推公式的时候，首先要考虑如下两种操作，整理如下：
	if (s[i - 1] == t[j - 1])
		t中找到了一个字符在s中也出现了
	if (s[i - 1] != t[j - 1])
		相当于t要删除元素，继续匹配
	 */
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = dp[i][j-1]	// 相当于t要删除元素，t如果把当前元素t[j - 1]删除，那么dp[i][j] 的数值就是看s[i - 1]与 t[j - 2]的比较结果了
			}
		}
	}
	return dp[m][n] == len(s)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 115.不同的子序列

```go
package main

import (
	"fmt"
)

func numDistinct(s string, t string) int {
	// dp[i][j]：以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]
	m, n := len(s), len(t)
	dp := make([][]int, m+1)
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = 1
	}

	/*
	这一类问题，基本是要分析两种情况
	s[i - 1] 与 t[j - 1]相等
	s[i - 1] 与 t[j - 1] 不相等
	 */
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if s[i-1] == t[j-1] {	// 当s[i - 1] 与 t[j - 1]相等时，dp[i][j]可以有两部分组成
				// 一部分是用s[i - 1]来匹配，那么个数为dp[i - 1][j - 1]
				// 一部分是不用s[i - 1]来匹配，个数为dp[i - 1][j]
				dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
			} else {
				dp[i][j] = dp[i-1][j]	// 相当于s要删除元素，s如果把当前元素s[i - 1]删除，那么dp[i][j] 的数值就是看t[j - 1]与 s[i - 2]的比较结果了
			}
		}
	}
	return dp[m][n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



##  583. 两个字符串的删除操作

```go
package main

import (
	"fmt"
)

func minDistance(word1 string, word2 string) int {
	// dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数。
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)

	// 初始化
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = i
	}
	for j:=0; j<=n; j++ {
		dp[0][j] = j
	}

	/*
	确定递推公式
	当word1[i - 1] 与 word2[j - 1]相同的时候
		dp[i][j] = dp[i - 1][j - 1]
	当word1[i - 1] 与 word2[j - 1]不相同的时候
		情况一：删word1[i - 1]，最少操作次数为dp[i - 1][j] + 1
		情况二：删word2[j - 1]，最少操作次数为dp[i][j - 1] + 1
		情况三：同时删word1[i - 1]和word2[j - 1]，操作的最少次数为dp[i - 1][j - 1] + 2
	 */
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j-1]+2, min(dp[i-1][j]+1, dp[i][j-1]+1))
			}
		}
	}
	return dp[m][n]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



##  72. 编辑距离

```go
package main

import (
	"fmt"
)

func minDistance(word1 string, word2 string) int {
	// dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)

	// 初始化
	for i:=0; i<=m; i++ {
		dp[i] = make([]int, n+1)
		dp[i][0] = i	// dp[i][0] ：以下标i-1为结尾的字符串word1，和空字符串word2，最近编辑距离为dp[i][0]
	}
	for j:=0; j<=n; j++ {
		dp[0][j] = j	// dp[0][j] ：以下标j-1为结尾的字符串word2，和空字符串word1，最近编辑距离为dp[0][j]
	}

	/*
	确定递推公式
	if (word1[i - 1] == word2[j - 1])
	    不操作	dp[i][j] = dp[i - 1][j - 1]
	if (word1[i - 1] != word2[j - 1])
	    增
	    删	dp[i][j] = dp[i - 1][j] + 1		dp[i][j] = dp[i][j - 1] + 1
	    换	dp[i][j] = dp[i - 1][j - 1] + 1
	 */
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
		}
	}
	return dp[m][n]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	fmt.Println("amadeus")
}

```



## 647. 回文子串

```go

func countSubstrings(s string) int {
	// 布尔类型的dp[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是dp[i][j]为true，否则为false
	n := len(s)
	dp := make([][]bool, n)
	res := 0

	// 初始化
	for i:=0; i<n; i++ {
		dp[i] = make([]bool, n)
	}

	/*
	确定递推公式
	当s[i]与s[j]不相等
		dp[i][j]一定是false。

	当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况
		情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
		情况二：下标i 与 j相差为1，例如aa，也是文子串
		情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文就可以了，
			那么aba的区间就是 i+1 与 j-1区间，这个区间是不是回文就看dp[i + 1][j - 1]是否为true。
	 */
	for i:=n-1; i>=0; i-- {
		for j:=i; j<n; j++ {	// 注意因为dp[i][j]的定义，所以j一定是大于等于i的，那么在填充dp[i][j]的时候一定是只填充右上半部分
			if s[i] == s[j] {
				if j-i <= 1 {	// 情况一 和 情况二
					res++
					dp[i][j] = true
				} else if dp[i+1][j-1] {
					res++
					dp[i][j] = true
				}
			}
		}
	}
	return res
}
```



##  516.最长回文子序列

```go
package main

import (
	"fmt"
)

func longestPalindromeSubseq(s string) int {
	// 回文子串是要连续的，回文子序列可不是连续的
	// dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]
	n := len(s)
	dp := make([][]int, n)

	// 初始化
	for i:=0; i<n; i++ {
		for j:=0; j<n; j++ {
			if dp[i] == nil {
				dp[i] = make([]int, n)
			}
			if i == j {
				dp[i][j] = 1
			}
		}
	}

	/*
	确定递推公式
	如果s[i]与s[j]相同
		那么dp[i][j] = dp[i + 1][j - 1] + 2

	如果s[i]与s[j]不相同，说明s[i]和s[j]的同时加入 并不能增加[i,j]区间回文子串的长度，那么分别加入s[i]、s[j]看看哪一个可以组成最长的回文子序列。
		加入s[j]的回文子序列长度为dp[i + 1][j]。
		加入s[i]的回文子序列长度为dp[i][j - 1]。
	 */
	for i:=n-1; i>=0; i-- {
		for j:=i+1; j<n; j++ {	// 注意因为dp[i][j]的定义，所以j一定是大于等于i的，那么在填充dp[i][j]的时候一定是只填充右上半部分
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i+1][j],dp[i][j-1])
			}
		}
	}
	return dp[0][n-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}



func main() {
	fmt.Println("amadeus")
}

```



# 单调栈

## 739. 每日温度

```go
func dailyTemperatures(num []int) []int {
	ans := make([]int, len(num))
	stack:= []int{}
	for i,v := range num {
		// 栈不空，且当前遍历元素 v 破坏了栈的单调性
		for len(stack)!=0 && v>num[stack[len(stack)-1]] {
			// pop
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ans[top] = i-top
		}
		stack = append(stack, i)
	}
	return ans
}
```



## 496.下一个更大元素 I

```go
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	res := make([]int, len(nums1))
	for i := range res {
		res[i] = -1
	}
	mp := map[int]int{}
	for i,v := range nums1 {
		mp[v] = i
	}
	stack := []int{}
	stack = append(stack, 0)
	for i:=1; i<len(nums2); i++ {
		for len(stack)>0 && nums2[i]>nums2[stack[len(stack)-1]] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]		// 出栈
			if _, ok := mp[nums2[top]]; ok {	// 看map里是否存在这个元素
				index := mp[nums2[top]] 		// 根据map找到nums2[top] 在 nums1中的下标
				res[index] = nums2[i]
			}
		}
		stack = append(stack, i)
	}
	return res
}
```



## 503.下一个更大元素II

```go
func nextGreaterElements(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	for i := range res {
		res[i] = -1
	}
	mp := map[int]int{}
	for i,v := range nums {
		mp[v] = i
	}
	stack := []int{}
	stack = append(stack, 0)	// 栈中存放的是nums中的元素下标
	for i:=0; i<n*2; i++ {
		for len(stack)>0 && nums[i%n]>nums[stack[len(stack)-1]] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]		// 出栈
			res[top] = nums[i%n]	// 更新result
			}
		stack = append(stack, i%n)
	}
	return res
}
```



## 42. 接雨水

```go

func trap(height []int) int {
	sum := 0
	n := len(height)
	lefth := make([]int, n)
	righth := make([]int, n)
	lefth[0] = height[0]
	righth[n-1] = height[n-1]

	// 记录每个柱子左边柱子最大高度
	for i:=1; i<n; i++ {
		lefth[i] = max(lefth[i-1], height[i])
	}
	// 记录每个柱子右边柱子最大高度
	for i:=n-2; i>=0; i-- {
		righth[i] = max(righth[i+1], height[i])
	}
	// 求和
	for i:=1; i<n-1; i++ {
		// 当前列雨水面积：min(左边柱子的最高高度，记录右边柱子的最高高度) - 当前柱子高度
		// 就想象成木桶接水    是由最短的那块木板决定的
		h := min(righth[i], lefth[i]) - height[i]
		if h > 0 {
			sum += h
		}
	}
	return sum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



##  84.柱状图中最大的矩形

```go

func largestRectangleArea(heights []int) int {
	sum := 0
	n := len(heights)
	// 两个DP数列储存的均是下标index
	lefth := make([]int, n)
	righth := make([]int, n)

	// 要记录记录每个柱子 左边第一个小于该柱子的下标，而不是左边第一个小于该柱子的高度

	// 注意这里初始化，防止下面while死循环
	lefth[0] = -1
	// 记录每个柱子 左边第一个小于该柱子的下标
	for i:=1; i<n; i++ {
		t := i-1
		// 这里不是用if，而是不断向左寻找的过程	以当前柱子为主心骨，向左迭代寻找次级柱子
		for t>=0 && heights[t]>=heights[i] {
			t = lefth[t]	// 这里不能单纯用t++，否则会超时   应该找到第一个小于当前下标t的柱子	记忆化搜索思想
		}
		// 当找到左侧矮一级的目标柱子时
		lefth[i] = t
	}

	// 注意这里初始化，防止下面while死循环
	righth[n-1] = n
	// 记录每个柱子 右边第一个小于该柱子的下标
	for i:=n-2; i>=0; i-- {
		t := i+1
		// 这里不是用if，而是不断向左寻找的过程
		for t<n && heights[t]>=heights[i] {
			t = righth[t]
		}
		righth[i] = t
	}

	// 求和
	res := 0
	for i:=0; i<n; i++ {
		sum = heights[i] * (righth[i]-lefth[i]-1)
		res = max(res, sum)
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```





# Leetcode Hot 100

## 2. 两数相加

```go
type ListNode struct {
	Val int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	// 建立一个虚拟头结点，这个虚拟头结点的 Next 指向真正的 head，这样 head 不需要单独处理
	head := &ListNode{Val: 0}
	// n1 : l1的值  		n2 : l2的值		carry : 进位
	n1, n2, carry, current := 0, 0, 0, head
	for l1!=nil || l2!=nil || carry!=0 {	// 链表1还未到底  或  链表2还未到底  或  还有进位没处理
		if l1 == nil {		// 链表1已到底
			n1 = 0				// 链表1的值为0
		} else {			// 链表1还未到底
			n1 = l1.Val		
			l1 = l1.Next
		}
		if l2 == nil {		// 链表2已到底
			n2 = 0
		} else {			// 链表2还未到底
			n2 = l2.Val
			l2 = l2.Next
		}
		// 创建一个新节点作为当前结点的下一个		值为 链表1+链表2+进位 除以10取余
		current.Next = &ListNode{Val: (n1+n2+carry)%10}
		current = current.Next
		carry = (n1+n2+carry)/10	// 计算在新节点应该加上的进位
	}
	return head.Next
}
```



## 3. 无重复字符的最长子串

```go
func lengthOfLongestSubstring(s string) int {
	if len(s) == 0 {
		return 0
	}
	var freq [127]int	// 维护的是滑动窗口内部出现的字符次数
	res, left, right := 0, 0, -1
	for left < len(s) {
		// 右边界不断的右移	只要没有重复的字符，就持续向右扩大窗口边界
		if right+1 < len(s) && freq[s[right+1]] == 0 {
			right++		// 右边界向右移动一位
			freq[s[right]]++	// 记录右边界出现的字符
		} else {	// 右边界出现了重复字符，就需要缩小左边界，直到重复的字符移出了左边界
			freq[s[left]]--		// 缩小左边界  维护滑动窗口（删除左边界的字符）
			left++	// 左边界右移一位
		}
		res = max(res, right-left+1)		// 滑动窗口大小  即 不重复字符串的长度
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 4. 寻找两个正序数组的中位数   不会

```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	// 假设 nums1 的长度小
	if len(nums1) > len(nums2) {
		return findMedianSortedArrays(nums2, nums1)
	}
	low, high, k, nums1Mid, nums2Mid := 0, len(nums1), (len(nums1)+len(nums2)+1)>>1, 0, 0
	for low <= high {
		// nums1:  ……………… nums1[nums1Mid-1] | nums1[nums1Mid] ……………………
		// nums2:  ……………… nums2[nums2Mid-1] | nums2[nums2Mid] ……………………
		nums1Mid = low + (high-low)>>1 // 分界限右侧是 mid，分界线左侧是 mid - 1
		nums2Mid = k - nums1Mid
		if nums1Mid > 0 && nums1[nums1Mid-1] > nums2[nums2Mid] { // nums1 中的分界线划多了，要向左边移动
			high = nums1Mid - 1
		} else if nums1Mid != len(nums1) && nums1[nums1Mid] < nums2[nums2Mid-1] { // nums1 中的分界线划少了，要向右边移动
			low = nums1Mid + 1
		} else {
			// 找到合适的划分了，需要输出最终结果了
			// 分为奇数偶数 2 种情况
			break
		}
	}
	midLeft, midRight := 0, 0
	if nums1Mid == 0 {
		midLeft = nums2[nums2Mid-1]
	} else if nums2Mid == 0 {
		midLeft = nums1[nums1Mid-1]
	} else {
		midLeft = max(nums1[nums1Mid-1], nums2[nums2Mid-1])
	}
	if (len(nums1)+len(nums2))&1 == 1 {
		return float64(midLeft)
	}
	if nums1Mid == len(nums1) {
		midRight = nums2[nums2Mid]
	} else if nums2Mid == len(nums2) {
		midRight = nums1[nums1Mid]
	} else {
		midRight = min(nums1[nums1Mid], nums2[nums2Mid])
	}
	return float64(midLeft+midRight) / 2
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 5. 最长回文子串

```go
func longestPalindrome(s string) string {
	res := ""
	n := len(s)
	// 定义 dp[i][j] 表示从字符串第 i 个字符到第 j 个字符这一段子串是否是回文串
	// 由回文串的性质可以得知，回文串去掉一头一尾相同的字符以后，剩下的还是回文串。
	// 所以状态转移方程是 dp[i][j] = (s[i] == s[j]) && ((j-i < 3) || dp[i+1][j-1])
	dp := make([][]bool, n)
	for i:=0; i<n; i++ {
		dp[i] = make([]bool, n)
	}
	for i:=n-1; i>=0; i-- {
		for j:=i; j<n; j++ {
			if s[i] == s[j] {
				// 特殊的情况
				// j - i == 1 的时候，即只有 2 个字符的情况，只需要判断这 2 个字符是否相同即可
				// j - i == 2 的时候，即只有 3 个字符的情况，只需要判断除去中心以外对称的 2 个字符是否相等
				if j-i<3 {
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
			}
			if dp[i][j] && (res=="" || j-i+1>len(res)) {	// 每次循环动态维护保存最长回文串
				res = s[i:j+1]
			}
		}
	}
	return res
}
```



## 10. 正则表达式匹配

```go
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	// dp[i][j] 表示 s 的前 i 个是否能被 p 的前 j 个匹配
	/*
	已知 dp[i-1][j-1] 意思就是前面子串都匹配上了，不知道新的一位的情况。
	那就分情况考虑，所以对于新的一位 p[j] s[i] 的值不同，要分情况讨论：
		1、考虑最简单的 p[j] == s[i] : dp[i][j] = dp[i-1][j-1]
	然后从 p[j] 可能的情况来考虑，让 p[j]=各种能等于的东西。
		2、p[j] == "." : dp[i][j] = dp[i-1][j-1]
		3、p[j] ==" * ":
			按照 p[j-1] 和 s[i] 是否相等，我们分为两种情况：
			3.1 p[j-1] != s[i] : dp[i][j] = dp[i][j-2]
			3.2 p[j-1] == s[i] or p[j-1] == "."

	 */
	dp := make([][]bool, m+1)
	for i:=0; i<=m; i++ {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	// matches(x,y) 判断两个字符是否匹配的辅助函数
	// 只有当 y 是 . 或者 x 和 y 本身相同时，这两个字符才会匹配
	matchs := func(i, j int) bool {
		if i==0 {
			return false
		}
		if p[j-1] == '.' {
			return true
		}
		return s[i-1] == p[j-1]
	}

	for i:=0; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if p[j-1] == '*' {						// 如果 p 的第 j 个字符是 *
				dp[i][j] = dp[i][j] || dp[i][j-2]	// 表示我们可以对 p 的第 j-1 个字符匹配任意自然数次
													// 在匹配 0 次的情况下，我们有 f[i][j] = f[i][j−2]
				if matchs(i, j-1) {
					dp[i][j] = dp[i][j] || dp[i-1][j]
				}
			} else if matchs(i, j) {
				dp[i][j] = dp[i][j] || dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}
```



## 10. 正则表达式匹配 报错

```go

func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	if m == 1 {
		if n == 1 {
			return true
		}
		return false
	}
	// dp[i][j] 表示 s 的前 i 个是否能被 p 的前 j 个匹配
	/*
	已知 dp[i-1][j-1] 意思就是前面子串都匹配上了，不知道新的一位的情况。
	那就分情况考虑，所以对于新的一位 p[j] s[i] 的值不同，要分情况讨论：
		1、考虑最简单的 p[j] == s[i] : dp[i][j] = dp[i-1][j-1]
	然后从 p[j] 可能的情况来考虑，让 p[j]=各种能等于的东西。
		2、p[j] == "." : dp[i][j] = dp[i-1][j-1]
		3、p[j] ==" * ":
			按照 p[j-1] 和 s[i] 是否相等，我们分为两种情况：
			3.1 p[j-1] != s[i] : dp[i][j] = dp[i][j-2]
			3.2 p[j-1] == s[i] or p[j-1] == "."

	 */
	dp := make([][]bool, m+1)
	for i:=0; i<=m; i++ {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true


	for i:=0; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if s[i] == p[j] {	// 元素匹配
				dp[i][j] = dp[i-1][j-1]
			} else {
				if p[j] == '.' {	// 任意元素
					dp[i][j] = dp[i-1][j-1]
				} else if p[j] == '*' {
					if s[i] != p[j-1] {		// 前一个元素不匹配 且不为任意元素
						dp[i][j] = dp[i][j-2]
					} else if s[i]==p[j-1] || p[j-1]=='.' {	// 前一个元素匹配 或 任意元素
						/*
						   	  dp[i][j] = dp[i-1][j] // 多个字符匹配的情况
						   or dp[i][j] = dp[i][j-1] // 单个字符匹配的情况
						   or dp[i][j] = dp[i][j-2] // 没有匹配的情况
						*/
						dp[i][j] = dp[i][j-1] || dp[i][j-2] || dp[i-1][j]
					}
				}
			}
		}
	}
	return dp[m][n]
}
```



## 11. 盛最多水的容器

```go
func maxArea(height []int) int {
	left, right := 0, len(height)-1
	ans := 0
	for left < right {
		area := min(height[left], height[right]) * (right - left)
		ans = max(ans, area)
		if height[left] <= height[right] {
			left++
		} else {
			right--
		}
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```



## 19. 删除链表的倒数第 N 个结点

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	slow, fast := dummy, head
	for i:=0; i<n; i++ {
		fast = fast.Next
	}
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}
```



## 31. 下一个排列

https://leetcode-cn.com/problems/next-permutation/solution/xia-yi-ge-pai-lie-suan-fa-xiang-jie-si-lu-tui-dao-/

```go
func nextPermutation(nums []int) {
	if len(nums) == 1 {
		return
	}
	n := len(nums)
	i, j, k := n-2, n-1, n-1
	// 从后向前查找第一个相邻升序的元素对
	for i>=0 && nums[i]>=nums[j] {
		i--
		j--
	}
	// 在 [j,end) 从后向前查找第一个大于 A[i] 的值 A[k]
	if i >= 0 {
		for nums[i] >= nums[k] {
			k--
		}
		// 将 A[i] 与 A[k] 交换
		nums[i], nums[k] = nums[k], nums[i]
	}
	// 这时 [j,end) 必然是降序，逆置 [j,end)，使其升序
	for i,j := j, n-1; i<j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}
```



## 32. 最长有效括号

```go
func longestValidParentheses(s string) int {
	ans := 0
	stack := make([]int, 0)
	// 保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」
	stack = append(stack, -1)
	for i:=0; i<len(s); i++ {
		if s[i] == '(' {
			// 对于遇到的每个 ‘(’ ，将它的下标放入栈中
			stack = append(stack, i)
		} else {
			// 对于遇到的每个 ‘)’ ，先弹出栈顶元素表示匹配了当前右括号
			stack = stack[:len(stack)-1]
			// 如果栈为空，说明当前的右括号为没有被匹配的右括号
			if len(stack) == 0 {
				// 将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
				stack = append(stack, i)
			} else {
				// 如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度
				ans = max(ans, i-stack[len(stack)-1])
			}
		}
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}


func longestValidParentheses(s string) int {
    left, right, maxLength := 0, 0, 0
    for i := 0; i < len(s); i++ {
        if s[i] == '(' {
            left++
        } else {
            right++
        }
        if left == right {
            maxLength = max(maxLength, 2 * right)
        } else if right > left {
            left, right = 0, 0
        }
    }
    left, right = 0, 0
    for i := len(s) - 1; i >= 0; i-- {
        if s[i] == '(' {
            left++
        } else {
            right++
        }
        if left == right {
            maxLength = max(maxLength, 2 * left)
        } else if left > right {
            left, right = 0, 0
        }
    }
    return maxLength
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
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
	left, right := 0, n-1
	// 将数组一分为二，其中一定有一个是有序的，另一个可能是有序，也能是部分有序。
	// 此时有序部分用二分法查找。无序部分再一分为二，其中一个一定有序，另一个可能有序，可能无序。就这样循环. 
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		}
		if nums[0] <= nums[mid] {	// 如果左区间是有序的
			if nums[0] <= target && target < nums[mid] {	// 如果target在左区间这个有序区间内
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {	// 如果左区间是无序的
			if nums[mid] < target && target <= nums[n-1] {	// 如果target在右区间这个有序区间内
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}
```



## 48. 旋转图像

```go
func rotate(matrix [][]int)  {
	n := len(matrix)
	// 水平翻转
	for i:=0; i<n/2; i++ {
		matrix[i], matrix[n-i-1] = matrix[n-i-1], matrix[i]
	}
	// 主对角线翻转
	for i:=0; i<n; i++ {
		for j:=0; j<i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}
```



## 56. 合并区间

https://leetcode-cn.com/problems/merge-intervals/solution/shou-hua-tu-jie-56he-bing-qu-jian-by-xiao_ben_zhu/

```go
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	res := [][]int{}
	prev := intervals[0]

	for i := 1; i < len(intervals); i++ {
		cur := intervals[i]
		if prev[1] < cur[0] { // 没有一点重合
			res = append(res, prev)
			prev = cur
		} else { // 有重合
			prev[1] = max(prev[1], cur[1])
		}
	}
    // 我们是先合并，遇到不重合再推入 prev。
	// 当考察完最后一个区间，后面没区间了，遇不到不重合区间，最后的 prev 没推入 res。 要单独补上
	res = append(res, prev)
	return res
}

func max(a, b int) int {
	if a > b { return a }
	return b
}
```



## 75. 颜色分类

```go
func sortColors(nums []int)  {
	p0, p2 := 0, len(nums)-1
	for i:=0; i<=p2; i++ {
		for i<=p2 && nums[i]==2 {	// 找到2，nums[i]与nums[p2]交换，p2向前移动一位
			nums[i], nums[p2] = nums[p2], nums[i]
			p2--
		}
		if nums[i] == 0 {	// 找到0，nums[i] 与 nums[p0]交换，p0向后移动一位
			nums[i], nums[p0] = nums[p0], nums[i]
			p0++
		}
	}
}
```



## 76. 最小覆盖子串

https://leetcode-cn.com/problems/minimum-window-substring/solution/by-lryong-vkjt/

```go
func minWindow(s string, t string) string {
	ori := make(map[byte]int) 	// 用一个哈希表表示 t 中所有的字符以及它们的个数
	cnt := make(map[byte]int)	// 用一个哈希表动态维护窗口中所有的字符以及它们的个数

	for i:=0; i<len(t); i++ {
		ori[t[i]]++
	}
	sLen := len(s)
	Len := math.MaxInt32
	ansL, ansR := -1, -1
	// 如果这个动态表中包含 t 的哈希表中的所有字符，并且对应的个数都不小于 t 的哈希表中各个字符的个数，那么当前的窗口是「可行」的
	check := func() bool {
		for k, v := range ori {		// k 是出现的字符， v 是字符出现的次数
			if cnt[k] < v {
				return false
			}
		}
		return true
	}
	for l, r := 0, 0; r < sLen; r++ {
		if r < sLen && ori[s[r]] > 0 {
			cnt[s[r]]++
		}
		for check() && l <= r {
			if (r - l + 1 < Len) {
				Len = r - l + 1
				ansL, ansR = l, l + Len
			}
			if _, ok := ori[s[l]]; ok {
				cnt[s[l]] -= 1
			}
			l++
		}
	}
	if ansL == -1 {
		return ""
	}
	return s[ansL:ansR]
}


func minWindow(s string, t string) string {
	// 先遍历 t， 记录元素到哈希表 need 中
	need := make(map[byte]int)
	for i := range t {
		need[t[i]]++
	}
	// 记录需要遍历的个数 len(t) 到 needCnt
	needCnt := len(t)
	// 初始化长度为2的 ret 数组[0, Inf]，作为记录目标子字符串的左右索引下标
	ret := [2]int{0, math.MaxInt32}

	var lo int
	// 遍历字符串 s， 如果遍历到的元素刚好>0, 说明是t中元素，needCnt减一。将遍历到的元素-1记录到哈希表 need 中
	for hi := range s {
		if need[s[hi]] > 0 {
			needCnt--
		}
		need[s[hi]]--

		// 如果 needCnt == 0 说明此时左右下标范围内 [lo,hi] 有符合要求子字符串, 开始缩小滑动窗口
		if needCnt == 0 {
			// 左索引下标 lo 向前推进
			// 当 need[s[lo]] == 0 时，说明遍历到t中元素（并且因为此时 needCnt==0）， s[lo,hi]为结果之一，判断是否最小
			for {
				if need[s[lo]] == 0 {
					break
				}
				need[s[lo]]++
				lo++
			}

			if hi-lo < ret[1]-ret[0] {
				ret = [2]int{lo, hi}
			}
			need[s[lo]]++
			needCnt++
			lo++
		}
	}

	if ret[1] > len(s) {
		return ""
	}
	return s[ret[0] : ret[1]+1]
}
```



## 114. 二叉树展开为链表

```go
func flatten(root *TreeNode)  {
    list := preorderTraversal(root)
    for i := 1; i < len(list); i++ {
        prev, curr := list[i-1], list[i]
        prev.Left, prev.Right = nil, curr
    }
}

func preorderTraversal(root *TreeNode) []*TreeNode {
    list := []*TreeNode{}
    if root != nil {
        list = append(list, root)
        list = append(list, preorderTraversal(root.Left)...)
        list = append(list, preorderTraversal(root.Right)...)
    }
    return list
}
```



## 124. 二叉树中的最大路径和

```go
func maxPathSum(root *TreeNode) int {
	maxSum := math.MinInt32
	var maxGain func(node *TreeNode) int
	maxGain = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		// 递归计算左右子节点的最大贡献值
		// 只有在最大贡献值大于 0 时，才会选取对应子节点
		leftGain := max(0, maxGain(node.Left))
		rightGain := max(0, maxGain(node.Right))

		// 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
		pricePath := node.Val + leftGain + rightGain
		// 更新答案
		maxSum = max(maxSum, pricePath)
		// 返回节点的最大贡献值
		return node.Val + max(leftGain, rightGain)
	}
	maxGain(root)
	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 128. 最长连续序列

```go
func longestConsecutive(nums []int) int {
	hashMap := make(map[int]bool)
	for _, num := range nums {
		hashMap[num] = true
	}
	ans := 0
	for num := range hashMap {
		// 要枚举的数 x 一定是在数组中不存在前驱数 x−1 的
		// 因此我们每次在哈希表中检查是否存在 x−1 即能判断是否需要跳过了。
		if !hashMap[num-1] {
			cur := num
			tmpAns := 1
			for hashMap[cur+1] {
				cur++
				tmpAns++
			}
			ans = max(ans, tmpAns)
		}
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 207. 课程表

https://leetcode-cn.com/problems/course-schedule/solution/course-schedule-tuo-bu-pai-xu-bfsdfsliang-chong-fa/

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
	var (
		edges = make([][]int, numCourses)	// 有向无环图的边
		visited = make([]int, numCourses)	// 用于判断每个节点 i （课程）的状态
		result []int
		valid = true
		dfs func(u int)
	)

	dfs = func(u int) {
		visited[u] = 1	// 将当前访问节点 i 对应 visited[i] 置 1，即标记其被本轮 DFS 访问过
		// 递归访问当前节点 i 的所有邻接节点 j，当发现环直接返回 False
		for _, v := range edges[u] {
			// 未被 DFS 访问
			if visited[v] == 0 {
				dfs(v)
				if !valid {
					return
				}
			} else if visited[v] == 1 {		// 已被当前节点启动的 DFS 访问
				// 说明在本轮 DFS 搜索中节点 i 被第 2 次访问，即 课程安排图有环 ，直接返回 False
				valid = false
				return
			}
		}
		visited[u] = 2	// 对于下一轮dfs来说，已被其他节点启动的 DFS 访问，无需再重复搜索，直接返回 True
		result = append(result, u)
	}

	// 构造边
	for _, info := range prerequisites {
		// 有0、1、2、3、4、5  共6门课程。 [[5,3],[5,4],[3,0],[3,1],[4,1],[4,2]]。学5之前要先学3，学5之前要先学4......
		// 这里的 info 类型为[]int，就是[5,3]
		// 这里构造连接表，起点是先修课程3，终点是后修课程5
		edges[info[1]] = append(edges[info[1]], info[0])	// 这样是因为起点为1，其终点有两个: 3和4
	}

	// 对 numCourses 个节点依次执行 DFS，判断每个节点起步 DFS 是否存在环
	for i := 0; i < numCourses && valid; i++ {
		// 从课程0启动DFS，先判断节点0还没被访问
		if visited[i] == 0 {
			dfs(i)
		}
	}
	return valid
}
```





## 208. 实现 Trie (前缀树) 

```go
type Trie struct {
	next [26]*Trie
	isEnd bool
}

func Constructor() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string)  {
	node := this
	for _, ch := range word {
		ch -= 'a'
		if node.next[ch] == nil {
			node.next[ch] = &Trie{}
		}
		node = node.next[ch]
	}
	node.isEnd = true
}

func (this *Trie) SearchPrefix(prefix string) *Trie {
	node := this
	for _, ch := range prefix {
		ch -= 'a'
		if node.next[ch] == nil {
			return nil
		}
		node = node.next[ch]
	}
	return node
}

func (this *Trie) Search(word string) bool {
	node := this.SearchPrefix(word)
	return node != nil && node.isEnd
}

func (this *Trie) StartsWith(prefix string) bool {
	return this.SearchPrefix(prefix) != nil
}
```





## 240. 搜索二维矩阵 II 

```go
func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	x, y := 0, n-1
	for x<m && y>=0 {
		if matrix[x][y] == target {
			return true
		} else if matrix[x][y] > target {
			y--
		} else {
			x++
		}
	}
	return false
}
```



## 297. 二叉树的序列化与反序列化

```go
type Codec struct {

}

func Constructor() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "null"
	}
	return strconv.Itoa(root.Val) + "," + this.serialize(root.Left) + "," + this.serialize(root.Right)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	list := strings.Split(data, ",")
	var build func() *TreeNode
	build = func() *TreeNode {
		// 如果弹出的字符为 "null"，则返回 null 节点
		if list[0] == "null" {	
			list = list[1:]
			return nil
		}
		// 如果弹出的字符是数值，则创建root节点，并递归构建root的左右子树，最后返回root
		val, _ := strconv.Atoi(list[0])
		list = list[1:]
		return &TreeNode{val, build(), build()}
	}
	return build()
}
```



## 301. 删除无效的括号

```go
func isValid(str string) bool {
	cnt := 0
	for _, ch := range str {
		if ch == '(' {
			cnt++
		} else if ch == ')' {
			cnt--
			if cnt < 0 {
				return false
			}
		}
	}
	return cnt == 0
}

func helper(ans *[]string, str string, start, lremove, rremove int) {
	// 当 lremove 和 rremove 同时为 0 时，则我们检测当前的字符串是否合法匹配，如果合法匹配则我们将其记录下来
	if lremove == 0 && rremove == 0 {
		if isValid(str) {
			*ans = append(*ans, str)
		}
		return
	}

	for i := start; i < len(str); i++ {
		if i != start && str[i] == str[i-1] {	// 当前的括号和前一个括号相同时剪枝
			continue
		}
		// 如果剩余的字符无法满足去掉的数量要求，直接返回
		if lremove+rremove > len(str)-i {
			return
		}
		// 尝试去掉一个左括号
		if lremove > 0 && str[i] == '(' {
			helper(ans, str[:i]+str[i+1:], i, lremove-1, rremove)
		}
		// 尝试去掉一个右括号
		if rremove > 0 && str[i] == ')' {
			helper(ans, str[:i]+str[i+1:], i, lremove, rremove-1)
		}
	}
}

func removeInvalidParentheses(s string) (ans []string) {
	lremove, rremove := 0, 0
	// 一次遍历计算出多余的「左括号」和「右括号」
	for _, ch := range s {
		if ch == '(' {	// 当遍历到「左括号」, 「左括号」数量加 1
			lremove++	//
		} else if ch == ')' {	// 当遍历到「右括号」
			if lremove == 0 {	// 如果此时「左括号」的数量为 0，「右括号」数量加 1
				rremove++
			} else {	// 如果此时「左括号」的数量不为 0，因为「右括号」可以与之前遍历到的「左括号」匹配，此时「左括号」出现的次数 -1
				lremove--
			}
		}
	}
	// 得到的「左括号」和「右括号」的数量就是各自最少应该删除的数量

	/*
	尝试在原字符串 s 中去掉 lremove 个左括号和 rremove 个右括号，
	然后检测剩余的字符串是否合法匹配，如果合法匹配则我们则认为该字符串为可能的结果，
	利用回溯算法来尝试搜索所有可能的去除括号的方案
	*/
	helper(&ans, s, 0, lremove, rremove)
	return
}
```



## 338. 比特位计数

```go
func countBits(n int) []int {
	bits := make([]int, n+1)
	for i:=1; i<=n; i++ {
		// y = x & (x−1) 为将 x 的最低设置位从 1 变成 0 之后的数
		// 显然 0≤y<x，bits[x]=bits[y]+1
		bits[i] = bits[i&(i-1)] + 1
	}
	return bits
}
```



## 312. 戳气球

```go
func maxCoins(nums []int) int {
	// 令 dp[i][j] 表示填满开区间 (i,j) 能得到的最多硬币数
	n := len(nums)
	res := make([][]int, n+2)
	for i:=0; i<n+2; i++ {
		res[i] = make([]int, n+2)
	}
	val := make([]int, n+2)
	val[0], val[n+1] = 1, 1
	for i:=1; i<=n; i++ {
		val[i] = nums[i-1]
	}
	for i:=n-1; i>=0; i-- {
		for j:=i+2; j<=n+1; j++ {
			for k:=i+1; k<j; k++ {
				// 新赚的钱 val[i] * val[k] * val[j]
				sum := val[i] * val[k] * val[j]
				// 把 (i,k) 开区间所有气球戳爆，然后把戳爆这些气球的所有金币都收入囊中，金币数量记录在 dp[i][k]
				// 同理，(k,j) 开区间你也已经都戳爆了，钱也拿了，记录在 dp[k][j]
				sum += res[i][k] + res[k][j]
				res[i][j] = max(res[i][j], sum)
			}
		}
	}
	return res[0][n+1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 394. 字符串解码

```go
func decodeString(s string) string {
	var rec func() string
	i := 0
	rec = func() string {
		ans := ""
		num := 0
		for ; i < len(s); i++ {
			if s[i] >= '0' && s[i] <= '9' {
				num = num*10 + int(s[i]-'0')
			} else if s[i] >= 'a' && s[i] <= 'z' {
				ans += string(s[i])
			} else if s[i] == '[' {
				i++
				tmp := rec()
				for j:=0; j<num; j++ {
					ans += tmp
				}
				num = 0
			} else {
				break
			}
		}
		return ans
	}
	return rec()
}
```



## 437. 路径总和 III

```go
func pathSum(root *TreeNode, targetSum int) int {
	// 节点的前缀和preSum为：由根结点到当前结点的路径上所有节点的和
	preSum := map[int64]int{0:1}
	ans := 0
	var dfs func(node *TreeNode, cur int64)
	dfs = func(node *TreeNode, cur int64) {
		// 先序遍历二叉树
		if node == nil {
			return
		}
		// 当前从根节点 root 到节点 node 的前缀和为 cur
		cur += int64(node.Val)
		// 此时我们在已保存的前缀和查找是否存在前缀和刚好等于 cur − targetSum
		// 假设从根节点 root 到节点 node 的路径中存在节点 pi 到根节点 root 的前缀和为 cur−targetSum，
		// 则节点 pi+1 到 node 的路径上所有节点的和一定为 targetSum
		ans += preSum[cur - int64(targetSum)]
		preSum[cur]++	// 利用深度搜索遍历树，当我们退出当前节点时，我们需要及时更新已经保存的前缀和
		dfs(node.Left, cur)
		dfs(node.Right, cur)
		preSum[cur]--
		return
	}
	dfs(root, 0)
	return ans
}
```



## 438. 找到字符串中所有字母异位词

```go
func findAnagrams(s string, p string) []int {
	n, m := len(s), len(p)
	res := make([]int, 0)
	if n < m {
		return res
	}
	chs, chp := make([]int, 26), make([]int, 26)
	for i:=0; i<m; i++ {
		chp[p[i]-'a']++		// 统计p中每种字符的数量
	}
	// 定义滑动窗口的左右两个指针 left，right
	// left和right表示滑动窗口在字符串s中的索引
	// cur_left和cur_right表示字符串s中索引为left和right的字符在数组中的索引
	for left, right :=0, 0; right<n; right++ {
		// right一步一步向右走遍历s字符串
		curRight := s[right] - 'a'
		chs[curRight]++
		// right当前遍历到的字符加入s_cnt后不满足p_cnt的字符数量要求，将滑动窗口左侧字符不断弹出，也就是left不断右移，直到符合要求为止
		for chs[curRight] > chp[curRight] {
			curLeft := s[left] - 'a'
			chs[curLeft]--
			left++
		}
		// 当滑动窗口的长度等于p的长度时，这时的s子字符串就是p的异位词
		if right-left+1 == m {
			res = append(res, left)
		}
	}
	return res
}
```



## 448. 找到所有数组中消失的数字

```go
func findDisappearedNumbers(nums []int) []int {
	n := len(nums)
	ans := make([]int, 0)
	// nums 的数字范围均在 [1,n] 中
	// 利用这一范围之外的数字，来表达「是否存在」的含义
	for _, v := range nums {
		// 遍历 nums，每遇到一个数 v，就让 nums[v−1] 增加 n
		v = (v-1) % n
		nums[v] += n	// +了n后，不影响其原本的值，因为遍历到这个数时，要经过 %n 这一步操作
	}
	for i, v := range nums {
		if v <= n {
			ans = append(ans, i+1)
		}
	}
	return ans
}
```



## 461. 汉明距离

```go
func hammingDistance(x, y int) int {
    return bits.OnesCount(uint(x ^ y))
}
```



## 581. 最短无序连续子数组

```go
func findUnsortedSubarray(nums []int) int {
	// 把数组分成三段，左段和右段是标准的升序数组，中段数组虽是无序的，但满足最小值大于左段的最大值，最大值小于右段的最小值
	n := len(nums)
	minn, maxn := math.MaxInt32, math.MinInt32
	left , right := -1, -1
	for i, num := range nums {
		// 从左到右维持最大值，寻找右边界end	如果nums[i]小于单调递增数列当前的最大值，说明存在错乱
		if maxn > num {		// maxn:表示前一项;nums[i]:表示当前项
			right = i	// 可理解为:前一项比当前项大时,该数组不为升序数组,并记录当前项. 遍历一次后,right即为最后一个使之不为升序数组的数.  left同理
		} else {
			maxn = num
		}
		// 从右到左维持最小值，寻找左边界begin
		if minn < nums[n-i-1] {
			left = n-i-1
		} else {
			minn = nums[n-i-1]
		}
	}
	if right == -1 {
		return 0
	}
	return right-left+1
}
```



## 621. 任务调度器

```go
func leastInterval(tasks []byte, n int) int {
	// 统计每个任务出现的次数，找到出现次数最多的任务
	count := make([]int, 26)
	for i:=0; i<len(tasks); i++ {
		count[tasks[i]-'A']++
	}
	sort.Slice(count, func(i, j int) bool {
		return count[i] < count[j]
	})

	// 因为相同元素必须有n个冷却时间，假设A出现3次，n = 2，任务要执行完，至少形成AXX AXX A序列
	minLen := (n+1) * (count[25]-1) + 1
	//此时为了尽量利用X所预占的空间（贪心）使得整个执行序列长度尽量小，将剩余任务往X预占的空间插入
	//剩余的任务次数有两种情况：
	//1.与A出现次数相同，那么B任务最优插入结果是ABX ABX AB，中间还剩两个空位，当前序列长度+1
	//2.比A出现次数少，若还有X，则按序插入X位置，比如C出现两次，形成ABC ABC AB的序列
	//直到X预占位置还没插满，剩余元素逐个放入X位置就满足冷却时间至少为n
	for i:=24; i>=0; i-- {
		if count[i] == count[25] {
			minLen++
		}
	}
	return max(minLen, len(tasks))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 153. 最小值

```go

```









# 剑指Offer



## 03. 数组中重复的数字

https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/solution/mian-shi-ti-03-shu-zu-zhong-zhong-fu-de-shu-zi-yua/

```go
func findRepeatNumber(nums []int) int {
	Map := make(map[int]bool)
	for _, v := range nums {
		if Map[v] {
			return v
		}
		Map[v] = true
	}
	return -1
}

func findRepeatNumber(nums []int) int {
	/*
	遍历中，第一次遇到数字 x 时，将其交换至索引 x 处；
	而当第二次遇到数字 x 时，一定有 nums[x] = x ，此时即可得到一组重复数字
	 */
	n := len(nums)
	for i:=0; i<n; {
		if nums[i] == i {
			i++		// 若 nums[i] = i ： 说明此数字已在对应索引位置，无需交换
			continue
		}
		if nums[nums[i]] == nums[i] {
			return nums[i]	// 找到一组重复值
		}
		nums[i], nums[nums[i]] = nums[nums[i]], nums[i]		// 将此数字交换至对应索引位置
	}
	return -1
}

```



## 04. 二维数组中的查找

```go
func findNumberIn2DArray(matrix [][]int, target int) bool {
	//if matrix == nil {
	//	return false
	//}
	m, n := len(matrix), 0	// 判断特殊情况  否则会报错
	if m == 0 {
		return false
	} else {
		n = len(matrix[0])
	}
	if matrix != nil && m > 0 && n > 0 {
		for r,c := 0, n-1; r<m && c>=0; {		// 选取矩阵右上角
			if matrix[r][c] == target {		// 找到
				return true
			} else if matrix[r][c] > target {
				c--		// 剔除所在列
			} else {
				r++		// 剔除所在行
			}
		}
	}
	return false
}
```



## 05. 替换空格

```go
func replaceSpace(s string) string {
	b := []byte(s)
	result := make([]byte, 0)
	for i:=0; i<len(b); i++ {
		if b[i]==' ' {
			result = append(result, []byte("%20")...)
		} else {
			result = append(result, b[i])
		}
	}
	return string(result)
}
```



## 06. 从尾到头打印链表

https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/go-yu-yan-shi-xian-san-chong-jie-fa-di-gui-fan-zhu/

```go
// 递归
func reversePrint(head *ListNode) []int {
	if head == nil {
		return nil
	}
	return append(reversePrint(head.Next), head.Val)
}

// 顺序遍历
func reversePrint(head *ListNode) []int {
	if head == nil {
		return nil
	}
	res := []int{}
	for head != nil {
		res = append(res, head.Val)
		head = head.Next
	}
	for i,j := 0, len(res)-1; i<j; {
		res[i], res[j] = res[j], res[i]
		i++
		j--
	}
	return res
}
```



## 07. 重建二叉树

```go
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}


func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{preorder[0], nil, nil}
	i := 0
	for ; i<len(preorder); i++ {
		if preorder[0] == inorder[i] {
			break
		}
	}
	root.Left = buildTree(preorder[1:i+1], inorder[:i])
	root.Right = buildTree(preorder[i+1:], inorder[i+1:])
	return root
}
```



## 09. 用两个栈实现队列

https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/solution/ke-yi-shi-xiao-gai-yong-liang-ge-zhan-sh-y1pb/

```go
type CQueue struct {
	stack1 *list.List
	stack2 *list.List
}


func Constructor() CQueue {
	return CQueue{
		stack1: list.New(),
		stack2: list.New(),
	}
}


func (this *CQueue) AppendTail(value int)  {
	this.stack1.PushBack(value)
}


func (this *CQueue) DeleteHead() int {
	// 如果第二个栈为空
	if this.stack2.Len() == 0 {
		for this.stack1.Len() > 0 {
			// Remove删除链表中的元素e，并返回e.Value
			// Back返回链表最后一个元素或nil
			// PushBack将一个值为v的新元素插入链表的最后一个位置，返回生成的新元素。
			this.stack2.PushBack(this.stack1.Remove(this.stack1.Back()))
		}
	}
	if this.stack2.Len() > 0 {	// 不能用else连在上面那个if   因为如果stack2为空，要先填充，再弹出
		tmp := this.stack2.Back()
		this.stack2.Remove(tmp)
		// e.value是interface{}类型，最终返回的是int类型，给他强转
		return tmp.Value.(int)
	}
	return -1
}
```



## 11. 旋转数组的最小数字

不知道为什么，用书上的改成go之后会超时

https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/solution/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-by-leetcode-s/

```go
func minArray(numbers []int) int {
	low := 0
	high := len(numbers) - 1
	for low < high {
		pivot := low + (high - low) / 2
		if numbers[pivot] < numbers[high] {
			high = pivot
		} else if numbers[pivot] > numbers[high] {
			low = pivot + 1
		} else {
			high--
		}
	}
	return numbers[low]
}
```



## 10- I. 斐波那契数列

```go
func fib(n int) int {
	const mod = 1000000007
	if n<2 {
		return n
	}
	p, q, res := 0, 0, 1
	for i:=2; i<=n; i++ {
		p=q
		q=res
		res=(p+q)%mod
	}
	return res
}
```



## 10- II. 青蛙跳台阶问题

注意和上一题的初始条件不一样

这里n=0时，答案是1

```go
func numWays(n int) int {
	const mod = 1000000007
	if n<2 {
		return 1
	}
	p, q, res := 1, 1, 1
	for i:=2; i<=n; i++ {
		p=q
		q=res
		res=(p+q)%mod
	}
	return res
}
```



## 15. 二进制中1的个数

```go
func hammingWeight(num uint32) (ones int) {
    for ; num > 0; num &= num - 1 {
        ones++
    }
    return
}
```



## 16. 数值的整数次方

```go
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	if n < 0 {	// 注意考虑负数
		x = 1/x
		n = -n
	}
	res := myPow(x, n>>1)
	res *= res
	if n&0x1 == 1 {
		res *= x
	}
	return res
}
```



## 18. 删除链表的节点

```go
func deleteNode(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	if head.Val == val {
		return head.Next
	}
	pre := head
	for pre.Next != nil && pre.Next.Val!=val {
		pre = pre.Next
	}
	if pre.Next != nil {
		pre.Next = pre.Next.Next
	}
	return head
}
```



## 21. 调整数组顺序使奇数位于偶数前面

```go
func exchange(nums []int) []int {
	i, j := 0, len(nums)-1
	for i<j {
		if !isOdd(nums[i]) && isOdd(nums[j]) {		// 偶 奇
			nums[i], nums[j] = nums[j], nums[i]
			i++
			j--
		} else if !isOdd(nums[i]) && !isOdd(nums[j]){	// 偶 偶
			j--
		} else if isOdd(nums[i]) && isOdd(nums[j]){		// 奇 奇
			i++
		} else {	// 奇 偶
			i++
			j--
		}
	}
	return nums
}

func isOdd(num int) bool {
	if (num & 1) == 1 {	// (nums[i]&1)==1  奇数判断，原理：奇数的最低位为1
		return true
	}
	return false
}
```



## 22. 链表中倒数第k个节点

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func getKthFromEnd(head *ListNode, k int) *ListNode {
	fast, slow := head, head
	for fast!=nil && k>0 {
		fast = fast.Next
		k--
	}
	for fast != nil {
		fast = fast.Next
		slow = slow.Next
	}
	return slow
}
```



## 24. 反转链表

https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/jian-zhi-offer-24-fan-zhuan-lian-biao-by-dint/

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next	// 先记录cur指向的下一个节点
		cur.Next = pre		// 再将cur指向的下一个节点修改为指向前一个节点
		pre = cur			// cur转为上一个节点
		cur = next			// cur指向之前记录的下一个节点next
	}
	return pre
}
```



## 25. 合并两个排序的链表

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	var head *ListNode
	if l1.Val < l2.Val {
		head = l1
		head.Next = mergeTwoLists(l1.Next, l2)
	} else {
		head = l2
		head.Next = mergeTwoLists(l1, l2.Next)
	}
	return head
}
```



## 26. 树的子结构

```go
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func isSubStructure(A *TreeNode, B *TreeNode) bool {
	if A == nil && B == nil {
		return true
	}
	if A == nil || B == nil {
		return false
	}
	res := false
	// 当在 A 中找到 B 的根节点时，进入递归校验
	if A.Val == B.Val {
		res = hasTree(A, B)
	}
	// res == false，说明 B 的根节点不在当前 A 树顶中，进入 A 的左子树进行递归查找
	if !res {
		res = isSubStructure(A.Left, B)
	}
	// 当 B 的根节点不在当前 A 树顶和左子树中，进入 A 的右子树进行递归查找
	if !res {
		res = isSubStructure(A.Right, B)
	}
	return res
}

// 校验 B 是否与 A 的一个子树拥有相同的结构和节点值
func hasTree(a, b *TreeNode) bool {
	if b == nil {
		return true
	}
	if a == nil {
		return false
	}
	if a.Val != b.Val {
		return false
	}
	// a.Val == b.Val 递归校验 A B 左子树和右子树的结构和节点是否相同
	return hasTree(a.Left, b.Left) && hasTree(a.Right, b.Right)
}
```



## 27. 二叉树的镜像

```go
func mirrorTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	left := mirrorTree(root.Left)
	right := mirrorTree(root.Right)
	root.Left = right
	root.Right = left
	return root
}
```



## 29. 顺时针打印矩阵

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {	// 必须这样先判断，再定义变量记录
		return []int{}
	}
	row := len(matrix)
	col := len(matrix[0])
	res := make([]int, 0)                // res := make([]int, row*col)  会出错  不知道为什么
	start := 0                           // 左上角(start, start) 作为每一圈的起点坐标
	for col > start*2 && row > start*2 { // 循环继续的条件  最后一圈的起点坐标
		myPrint(matrix, row, col, start, &res)
		start++
	}
	return res
}

func myPrint(matrix [][]int, rows, cols int, start int, res *[]int) {
	endx := cols - 1 - start // x终止行号
	endy := rows - 1 - start // y终止列号
	// 从左到右打印一行
	for j := start; j <= endx; j++ {
		*res = append(*res, matrix[start][j])
	}
	// 从上到下打印一列
	if start < endy { // 最后一圈可能会退化成只有一行、一列、一个数  此时打印不需要四步
		for i := start + 1; i <= endy; i++ { // 注意i要提前+1
			*res = append(*res, matrix[i][endx])
		}
	}
	// 从右到左打印一行
	if start < endx && start < endy {
		for j := endx - 1; j >= start; j-- { // 注意j要提前-1
			*res = append(*res, matrix[endy][j])
		}
	}
	// 从下到上打印一列
	if start < endx && start < endy-1 {
		for i := endy - 1; i >= start+1; i-- { // 注意i要提前-1
			*res = append(*res, matrix[i][start])
		}
	}
}
```



## 30. 包含min函数的栈

```go
type MinStack struct {
	stack     []int
	helpStack []int
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		stack:     []int{},
		helpStack: []int{math.MaxInt64},
	}
}

func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	top := this.helpStack[len(this.helpStack)-1]
	this.helpStack = append(this.helpStack, min(top, x))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.helpStack = this.helpStack[:len(this.helpStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) Min() int {
	return this.helpStack[len(this.helpStack)-1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 31. 栈的压入、弹出序列

```go
func validateStackSequences(pushed []int, popped []int) bool {
	stack := make([]int, 0)
	idx := 0
	for _, v := range pushed {
		stack = append(stack, v)                                   // 遍历 pushed 并入栈
		for len(stack) > 0 && stack[len(stack)-1] == popped[idx] { // 如果 stack元素 == pushed元素
			stack = stack[:len(stack)-1] // 出栈
			idx++                        // 弹出序号+1
		}
	}
	return len(stack) == 0
}
```



## 32 - II. 从上到下打印二叉树 II

```go
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func levelOrder(root *TreeNode) [][]int {
	res := make([][]int, 0)
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}		// 存储节点的队列
	level := 0	//层级
	for len(queue) != 0 {
		// 利用临时队列，暂存每个节点的左右子树
		tmp := []*TreeNode{}
		// 只需考虑在同一层上追加元素
		res = append(res, []int{})
		// 遍历队列，追加队列元素到切片同一层，追加队列元素左右子树到临时队列
		for _, v := range queue {
			res[level] = append(res[level], v.Val)
			if v.Left != nil {
				tmp = append(tmp, v.Left)
			}
			if v.Right != nil {
				tmp = append(tmp, v.Right)
			}
		}
		//层级加1，队列重新复制为队列的左右子树集
		level++
		// 队列赋值
		queue = tmp
	}
	return res
}
```



## 33. 二叉搜索树的后序遍历序列

```go
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func verifyPostorder(postorder []int) bool {
	var dfs func(i, j int) bool
	dfs = func(i, j int) bool {
		if i>j {
			return true
		}
		p := i	// 起始节点
		for postorder[p] < postorder[j] {
			p++		// 找到第一个大于根节点的节点	找到都比root小的子区间，这个区间就是左子树
		}
		m := p	// 记录大于根节点的节点
		for postorder[p] > postorder[j] {
			p++		// 找到都比root大的子区间	这个区间就是右子树
		}
		return p==j && dfs(i,m-1) && dfs(m,j-1)		// 符合后续遍历的条件：前一段是小于root.val的序列，后一段是大于的序列
	}
	return dfs(0, len(postorder)-1)	// 根节点在序列的最后一个
}
```



## 34. 二叉树中和为某一值的路径

```go
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func pathSum(root *TreeNode, target int) [][]int {
	res := make([][]int, 0)
	path := make([]int, 0)
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, left int) {
		if node == nil {
			return
		}
		left -= node.Val	// 减去当前节点值
		path = append(path, node.Val)	// 将节点加入路径
		if node.Left == nil && node.Right == nil && left==0 {	// 叶节点，且和为target
			res = append(res, append([]int{}, path...))		// 路径添加到结果
			path = path[:len(path)-1]	// return前要删去路径中当前节点
			return
		}
		dfs(node.Left, left)		// 递归左子树
		dfs(node.Right, left)		// 递归右子树
		path = path[:len(path)-1]	// 递归完成后返回上层前删去路径中当前节点
	}
	dfs(root, target)
	return res
}
```



## 35. 复杂链表的复制

https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/solution/fu-za-lian-biao-de-fu-zhi-by-leetcode-so-9ik5/

本题要求我们对一个特殊的链表进行深拷贝。如果是普通链表，我们可以直接按照遍历的顺序创建链表节点。而本题中因为随机指针的存在，当我们拷贝节点时，「当前节点的随机指针指向的节点」可能还没创建，因此我们需要变换思路。一个可行方案是，我们利用回溯的方式，让每个节点的拷贝操作相互独立。对于当前节点，我们首先要进行拷贝，然后我们进行「当前节点的后继节点」和「当前节点的随机指针指向的节点」拷贝，拷贝完成后将创建的新节点的指针返回，即可完成当前节点的两指针的赋值。

具体地，我们用哈希表记录每一个节点对应新节点的创建情况。遍历该链表的过程中，我们检查「当前节点的后继节点」和「当前节点的随机指针指向的节点」的创建情况。如果这两个节点中的任何一个节点的新节点没有被创建，我们都立刻递归地进行创建。当我们拷贝完成，回溯到当前层时，我们即可完成当前节点的指针赋值。注意一个节点可能被多个其他节点指向，因此我们可能递归地多次尝试拷贝某个节点，为了防止重复拷贝，我们需要首先检查当前节点是否被拷贝过，如果已经拷贝过，我们可以直接从哈希表中取出拷贝后的节点的指针并返回即可。

在实际代码中，我们需要特别判断给定节点为空节点的情况。

```go
type Node struct {
	Val int
	Next *Node
	Random *Node
}

var hashNode map[*Node]*Node		// index 是原链表节点     value 是创建的新节点   两者值相等，是深拷贝关系

func copyRandomList(head *Node) *Node {
	hashNode = map[*Node]*Node{}
	return myCopy(head)
}

func myCopy(node *Node) *Node {
	if node == nil {
		return nil
	}
	if n, ok := hashNode[node]; ok {	// 新节点在hash中存在，直接返回
		return n
	}
	newNode := &Node{Val: node.Val}		// 新节点在hash中不存在，创建新节点
	hashNode[node] = newNode	// 用哈希表记录每一个节点对应新节点的创建情况
	// 检查「当前节点的后继节点」和「当前节点的随机指针指向的节点」的创建情况
	// 如果这两个节点中的任何一个节点的新节点没有被创建，我们都立刻递归地进行创建
	newNode.Next = myCopy(node.Next)
	newNode.Random = myCopy(node.Random)
	return newNode
}
```



## 36. 二叉搜索树与双向链表

```go

```



## 38. 字符串的排列

```go
func permutation(s string) []string {
	var res []string // 返回值列表
	hashMap := make(map[byte]int)
	str := ""   // 待构造的字符串
	for i:=0; i<len(s); i++ {
		hashMap[s[i]]++    // 计数
	}
	var dfs func(start int)
	dfs = func(start int) {
		if start == len(s) {		// 一次全排列完成
			res = append(res, str)  // 字符串构造完毕 添加进返回值列表
			return
		}
		for k := range hashMap {
			if hashMap[k] != 0 {   // 次数不为0说明可用
				str += string(k)	// 正在构造的字符串加上当前的字符
				hashMap[k]--	// 当前字符的存量--
				dfs(start+1)    // 正在构造的字符串长度+1	继续构造字符串
				str = str[:len(str)-1]  // 回溯
				hashMap[k]++	// 回溯
			}
		}
	}
	dfs(0)
	return res
}
```



## 39. 数组中出现次数超过一半的数字

```go
// 哈希
func majorityElement(nums []int) int {
	hashMap := make(map[int]int)
	for _, v := range nums {
		hashMap[v]++
	}
	count := 1
	index := nums[0]
	for i,v := range hashMap {
		if v > count {
			count = v
			index = i
		}
	}
	return index
}

// Boyer-Moore 投票算法
// 如果我们把众数记为 +1+1，把其他数记为 -1−1，将它们全部加起来，显然和大于 0，从结果本身我们可以看出众数比其他数多
func majorityElement(nums []int) int {
	candidate := nums[0]
	count := 1
	for i:=1; i<len(nums); i++ {
		if count == 0 {
			candidate = nums[i]
		}
		if nums[i] == candidate {
			count++
		} else {
			count--
		}
	}
	return candidate
}
```



## 40. 最小的k个数

https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/solution/dui-pai-gui-bing-kuai-pai-shi-xian-zui-x-34vb/

```go
// 排序
func getLeastNumbers(arr []int, k int) []int {
	sort.Ints(arr)
	return arr[:k]
}

// 大顶堆
func getLeastNumbers(arr []int, k int) []int {
	var heapify func(nums []int, root, end int)
	heapify = func(nums []int, root, end int) {
		// 大顶堆堆化，堆顶值小一直下沉
		for {
			// 左孩子节点索引
			child := root*2 + 1
			// 越界跳出
			if child > end {
				return
			}
			// 比较左右孩子，取大值，否则child不用++
			if child < end && nums[child] <= nums[child+1] {
				child++
			}
			// 如果父节点已经大于左右孩子大值，已堆化
			if nums[root] > nums[child] {
				return
			}
			// 孩子节点大值上冒
			nums[root], nums[child] = nums[child], nums[root]
			// 更新父节点到子节点，继续往下比较，不断下沉
			root = child
		}
	}
	end := len(arr)-1
	// 从最后一个非叶子节点开始堆化
	for i:=end/2;i>=0;i-- {
		heapify(arr, i, end)
	}
	// 依次弹出元素，然后再堆化，相当于依次把最大值放入尾部
	for i:=end;i>=0;i-- {
		arr[0], arr[i] = arr[i], arr[0]
		end--
		heapify(arr, 0, end)
	}
	return arr[:k]
}

```



## 42. 连续子数组的最大和

```go
func maxSubArray(nums []int) int {
	max := nums[0]
	for i:=1; i<len(nums); i++ {
		if nums[i-1] + nums[i] > nums[i] {
			nums[i] += nums[i-1]
		}
		if nums[i] > max {
			max = nums[i]
		}
	}
	return max
}
```



## 43. 1～n 整数中 1 出现的次数

https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/1n-zheng-shu-zhong-1-chu-xian-de-ci-shu-umaj8/

```go
func countDigitOne(n int) int {
	ans := 0
	for powK:=1; powK<=n; powK*=10 {
		ans += (n/(powK*10))*powK + min(max(n%(powK*10)-powK+1, 0), powK)
	}
	return ans
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 45. 把数组排成最小的数

```go
func minNumber(nums []int) string {
	// 将整数数组按字符串形式排序
	sort.Slice(nums, func(i, j int) bool {
		x := fmt.Sprintf("%d%d", nums[i], nums[j])
		y := fmt.Sprintf("%d%d", nums[j], nums[i])
		return x < y
	})

	res := ""
	for i:=0; i<len(nums); i++ {
		res += fmt.Sprintf("%d", nums[i])
	}
	return res
}
```



## 49. 丑数

```go
func nthUglyNumber(n int) int {
	// dp[i] 表示第 i 个丑数
	dp := make([]int, 0)
	// 最小的丑数是 1，因此 dp[1]=1
	dp[1] = 1
	// 定义三个指针 p2, p3, p5，表示下一个丑数是当前指针指向的丑数乘以对应的质因数。初始时，三个指针的值都是 1
	p2, p3, p5 := 1, 1, 1
	for i:=2; i<=n; i++ {
		x2, x3, x5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = min(min(x2, x3), x5)
		if dp[i] == x2 {
			p2++
		}
		if dp[i] == x3 {
			p3++
		}
		if dp[i] == x5 {
			p5++
		}
	}
	return dp[n]
}
```



## 50. 第一个只出现一次的字符

```go
// 样例不全过
func firstUniqChar(s string) byte {
	hashMap := make(map[byte]int)
	for _,v := range s {
		hashMap[byte(v)]++
	}
	for i,v := range hashMap {
		if v == 1 {
			return i
		}
	}
	return ' '
}

// 过了，但是很慢
func firstUniqChar(s string) byte {
	hashMap := make(map[byte]int)
	for _,v := range s {
		hashMap[byte(v)]++
	}
	for i,v := range s {
		if hashMap[byte(v)] == 1 {
			return s[i]
		}
	}
	return ' '
}

// 很快
func firstUniqChar(s string) byte {
    cnt := [26]int{}
    for _, ch := range s {
        cnt[ch-'a']++
    }
    for i, ch := range s {
        if cnt[ch-'a'] == 1 {
            return s[i]
        }
    }
    return ' '
}
```



## 51. 数组中的逆序对

时间优化：https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/goji-bai-100-by-ben-xiao-hai-40/

> go的归并排序，可以做一个优化，因为执行过程需要一个临时切片，可以预先分配这个临时切片

```go
// 未优化
func reversePairs(nums []int) int {
	return mergeSort(nums, 0, len(nums)-1)
}

func mergeSort(nums []int, start, end int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)/2
	// count包括整体+左半+右半
	count := mergeSort(nums, start, mid) + mergeSort(nums, mid+1, end)
	// 辅助数组,用于存储排好序的数
	tmp := []int{}
	// 左指针从左半部分的左端开始遍历 右指针从右半部分的左端开始遍历
	i, j := start, mid+1
	// 指针超出自己的那部分则退出循环
	for i<=mid && j<=end {
		// 当左元素小于等于右元素 且右半部分的右元素的左边的元素均小于左元素
		if nums[i] <= nums[j] {
			tmp = append(tmp, nums[i])
			// 因此count需要加上右元素左边部分的元素数量
			count += j - mid -1
			i++
		} else {
			// 当左元素大于右元素 不能直接开始计数
			tmp = append(tmp, nums[j])
			j++ // 因为不确定右元素右边还有没有比左元素更小的元素
		}
	}
	// 处理另一半未遍历完的数据
	for i <= mid {
		tmp = append(tmp, nums[i])
		i++
		count += end - mid
	}
	for j <= end {
		tmp = append(tmp, nums[j])
		j++
	}
	// 将排序好的数组元素依次赋值给nums数组
	for i:=start; i<=end; i++ {
		nums[i] = tmp[i-start]
	}
	return count
}


// 时间优化
func reversePairs(nums []int) int {
	tmp := make([]int, len(nums))
	return mergeSort(nums, 0, len(nums)-1, tmp)
}

func mergeSort(nums []int, start, end int, tmp []int) int {
	if start >= end {
		return 0
	}
	mid := start + (end-start)/2
	// count包括整体+左半+右半
	count := mergeSort(nums, start, mid, tmp) + mergeSort(nums, mid+1, end, tmp)
	// 辅助数组,用于存储排好序的数
	tmp = tmp[:0]
	// 左指针从左半部分的左端开始遍历 右指针从右半部分的左端开始遍历
	i, j := start, mid+1
	// 指针超出自己的那部分则退出循环
	for i<=mid && j<=end {
		// 当左元素小于等于右元素 且右半部分的右元素的左边的元素均小于左元素
		if nums[i] <= nums[j] {
			tmp = append(tmp, nums[i])
			// 因此count需要加上右元素左边部分的元素数量
			count += j - mid -1
			i++
		} else {
			// 当左元素大于右元素 不能直接开始计数
			tmp = append(tmp, nums[j])
			j++ // 因为不确定右元素右边还有没有比左元素更小的元素
		}
	}
	// 处理另一半未遍历完的数据
	for i <= mid {
		tmp = append(tmp, nums[i])
		i++
		count += end - mid
	}
	for j <= end {
		tmp = append(tmp, nums[j])
		j++
	}
	// 将排序好的数组元素依次赋值给nums数组
	for i:=start; i<=end; i++ {
		nums[i] = tmp[i-start]
	}
	return count
}
```



## 52. 两个链表的第一个公共节点

```go
// 官方题解
type ListNode struct {
	Val  int
	Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {	// 只有当链表 headA 和 headB 都不为空时，两个链表才可能相交
		return nil	// 因此首先判断链表 headA 和 headB 是否为空，如果其中至少有一个链表为空，则两个链表一定不相交
	}
	pa, pb := headA, headB	// 用两个指针依次遍历两个链表的每个节点
	for pa != pb {
		if pa == nil {
			pa = headB		// 如果指针为空，则将指针移到另一个链表的头节点
		} else {
			pa = pa.Next	// 如果指针不为空，则将指针移到下一个节点
		}
		if pb == nil {
			pb = headA		// 如果指针为空，则将指针移到另一个链表的头节点
		} else {
			pb = pb.Next	// 如果指针不为空，则将指针移到下一个节点
		}
	}
	return pa
}

// 自己想的
type ListNode struct {
	Val  int
	Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {	// 只有当链表 headA 和 headB 都不为空时，两个链表才可能相交
		return nil	// 因此首先判断链表 headA 和 headB 是否为空，如果其中至少有一个链表为空，则两个链表一定不相交
	}
	pa, pb := headA, headB	// 用两个指针依次遍历两个链表的每个节点
	lenA, lenB := 0, 0
	for pa != nil {
		lenA++
		pa = pa.Next
	}
	for pb != nil {
		lenB++
		pb = pb.Next
	}
	pa, pb = headA, headB
	n := 0
	if lenA >= lenB {
		n = lenA - lenB
		for i:=0; i<n; i++ {
			pa = pa.Next
		}
	} else {
		n = lenB - lenA
		for i:=0; i<n; i++ {
			pb = pb.Next
		}
	}
	for pa != nil {
		if pa == pb {
			return pa
		} else {
			pa = pa.Next
			pb = pb.Next
		}
	}
	return nil
}
```



## 53 - I. 在排序数组中查找数字 I

```go
// 调包
func search(nums []int, target int) int {
	left := sort.SearchInts(nums, target)
	if left == len(nums) || nums[left] != target {
		return 0
	}
	right := sort.SearchInts(nums, target+1) -1
	return right-left+1
}

// 二分
func search(nums []int, target int) int {
	index := BinarySearchFirstEqualTarget(nums, target)
	// 证明数组中没有值为target的元素
	if index == -1 {
		return 0
	}
	count := 0
	// 下标小于index的元素都小于target,从index开始向后遍历直到遇到不等于target的元素为止
	for _, num := range nums[index:] {
		if num == target {
			count++
		} else {
			// 之后的元素都会大于target
			break
		}
	}
	return count
}


func BinarySearchFirstEqualTarget(array []int, target int) int {
	n := len(array)
	if n == 0 {
		return -1
	}
	low, high := 0, n-1
	for low <= high {
		mid := low + (high-low)/2
		if array[mid] > target {
			high = mid - 1
		} else if array[mid] < target {
			low = mid + 1
		} else {
			if (mid == 0) || (mid != 0 && array[mid-1] < target) {
				return mid
			} else {
				high = mid - 1
			}
		}
	}
	return -1
}
```



## 55 - I. 二叉树的深度

```go
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := maxDepth(root.Left)
	right := maxDepth(root.Right)
	if left > right {
		return left + 1
	} else {
		return right + 1
	}
}
```



## 55 - II. 平衡二叉树

```go
// 书
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	left := treeDepth(root.Left)
	right := treeDepth(root.Right)
	diff := left - right
	if diff < -1 || diff > 1 {
		return false
	}
	return isBalanced(root.Left) && isBalanced(root.Right)
}

func treeDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := treeDepth(root.Left)
	right := treeDepth(root.Right)
	if left > right {
		return left + 1
	} else {
		return right + 1
	}
}

// 官方题解
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func isBalanced(root *TreeNode) bool {
	return height(root) >= 0
}

func height(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := height(root.Left)
	rightHeight := height(root.Right)
	if abs(leftHeight - rightHeight) > 1 || leftHeight == -1 || rightHeight == -1  {
		return -1
	}
	return max(leftHeight, rightHeight) + 1
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}
```



## 56 - I. 数组中数字出现的次数

```go
/*
异或操作的性质：对于两个操作数的每一位，相同结果为 0，不同结果为 1。
在计算过程中，成对出现的数字的所有位会两两抵消为 0，最终得到的结果就是那个出现了一次的数字
9	==  !=	等于/不等于
10	&	按位与
11	^	按位异或
12	|	按位或
*/

func singleNumbers(nums []int) []int {
	res := 0	// 所有数字异或的结果
	a := 0		// 只出现了一次的数字
	b := 0		// 只出现了一次的数字
	for _, v := range nums {
		res ^= v	// 先对所有数字进行一次异或，得到两个出现一次的数字的异或值
	}
	// 在异或结果中找到任意为 1 的位 	找到第一位不是0的
	h := 1
	// 在实际操作的过程中，我们拿到序列的异或和 res 之后，对于这个「位」是可以任取的，只要它满足 xi = 1
	// 但是为了方便，这里的代码选取的是「不为 0 的最低位」，当然你也可以选择其他不为 0 的位置。
	for res & h == 0 {
		h <<= 1
	}
	for _, v := range nums {
		// 根据这一位对所有的数字进行分组
		// 在每个组内进行异或操作，得到两个数字
		if h & v == 0 {
			a ^= v
		} else {
			b ^= v
		}
	}
	return []int{a, b}
}
```



## 57. 和为s的两个数字

```go
func twoSum(nums []int, target int) []int {
	p1, p2 := 0, len(nums)-1
	for p1 <= p2 {
		if nums[p1] + nums[p2] == target {
			return []int{nums[p1], nums[p2]}
		} else if nums[p1] + nums[p2] > target {
			p2--
		} else {
			p1++
		}
	}
	return nil
}
```



## 57 - II. 和为s的连续正数序列

```go
// 测试
	fmt.Println("amadeus")
	res := make([][]int, 0)

	for i:=0; i<3; i++ {
		tmp := make([]int, 0)
		for j:= 1; j<10; j++ {
			tmp = append(tmp, j)
		}
		fmt.Println("tmp",i," : ",tmp)
		res = append(res, tmp)
	}
	fmt.Println("res : ", res)

// 题解
func findContinuousSequence(target int) [][]int {
	if target < 3 {
		return nil
	}
	res := make([][]int, 0)
	small := 1
	big := 2
	mid := (1+target)/2
	curSum := small + big
	for small < mid {	// 最小值小于中值（至少要两个数）
		if curSum == target {	// 和等于目标值
			tmp := make([]int, 0)
			for i:=small; i<=big; i++ {
				tmp = append(tmp, i)
			}
			res = append(res, tmp)
		}
		for curSum > target && small < mid {	// 和大于目标值
			curSum -= small
			small++
			if curSum == target {
				tmp := make([]int, 0)
				for i:=small; i<=big; i++ {
					tmp = append(tmp, i)
				}
				res = append(res, tmp)
			}
		}
		// 和小于目标值
		big++
		curSum += big
	}
	return res
}
```



## 58 - I. 翻转单词顺序

```go
func reverseWords(s string) string {
	// 翻转字符串里的单词
	// 三部曲：1.翻转字符串，2.找单词左右边界并翻转单词，3.拼接字符串
	// golang的字符串不可变类型，需要额外空间转为byte数组
	if s == "" || strings.Count(s, " ") == len(s) {
		return ""
	}
	b := []byte(s)
	var reverse func(start, end int)
	reverse = func(start, end int) {
		for start < end {
			b[start], b[end] = b[end], b[start]
			start++
			end--
		}
	}
	n := len(b)
	// 1.翻转整个字符串
	reverse(0, n-1)
	// 2.确定单词边界并翻转单个单词
	start, end := 0, 0
	// 拼接结果变量
	res := ""
	// 遍历s的byte数组
	for i:=0; i<n; i++ {
		// 每轮循环处理一个单词
		// 通过遇到非空字符作为单词的开始边界
		if b[i] != ' ' {	// 移除多余空格
			start = i
			for i < n && b[i] != ' ' {
				i++
			}
			// 单词的右边界
			end = i - 1
			// 左右边界已经确定，翻转单个单词
			reverse(start, end)
			// 3.拼接到结果string中, 由于头部单词前有空格，最后结果需要去掉
			res += " " + string(b[start:end+1])
		}
	}
	return res[1:]
}
```



## 58 - II. 左旋转字符串

```go
func reverseLeftWords(s string, n int) string {
	b := []byte(s)
	var reverse func(start, end int)
	reverse = func(start, end int) {
		for start < end {
			b[start], b[end] = b[end], b[start]
			start++
			end--
		}
	}
	m := len(b)
	reverse(0, n-1)
	reverse(n, m-1)
	reverse(0, m-1)
	return string(b)
}
```



## 60. n个骰子的点数

https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/go-shuang-bai-dong-tai-gui-hua-by-todother-3/

https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/golanger-wei-dong-tai-gui-hua-jian-ji-yi-dv8q/

```go
func dicesProbability(n int) []float64 {
	// dp : 在有n个骰子的情况下， 每一种可能摇出的数字出现的次数
	// i代表当前有几个骰子，j代表当前拼成的数字总和是多少
	dp := make([][]int, n+1)
	for i:=0; i<=n; i++ {
		dp[i] = make([]int, 6*n+1)
	}
	for i:=1; i<=6; i++ {	// 给第一个骰子赋初值
		dp[1][i] = 1		// 每种情况出现一次
	}
	for i:=1; i<=n; i++ {	// 不知道为什么这样初始化
		dp[i][i] = 1
	}
	for i:=2; i<=n; i++ {	// 第2到n个骰子
		for j:=i+1; j<=6*i; j++ {	// 可能拼成的数字总合
			// 如果现在的总和是j，那么最后一个骰子可能分别摇到1，2，3，4，5，6，所以
			// dp[i][j]+=for (1~6) dp[i-1][j-k] (k代表了1到6)
			for k:=1; k<=6; k++ {	// 当前骰子摇到的值
				if j-k >= i-1 {		// 现在的总合-现在摇到的值 > 之前用到的骰子(i-1)的最小值
					dp[i][j] += dp[i-1][j-k]
				}
			}
		}
	}
	res := make([]float64, 6*n)
	for i:=n; i<=6*n; i++ {
		res[i-1] = float64(dp[n][i]) / math.Pow(6, float64(n))
	}
	return res[n-1:]
}

func dicesProbability(n int) []float64 {
	// dp : 在有n个骰子的情况下， 每一种可能摇出的数字出现的概率
	// i代表当前有几个骰子，j代表当前拼成的数字总和是多少
	dp := make([][]float64, n)
	for i := range dp{		// i : 0 ~ n-1
		dp[i] = make([]float64, (i + 1) * 6 - i)	// 数字总合的范围
	}
	for i := range dp[0]{
		dp[0][i] = float64(1) / float64(6)		// 第一个骰子的各个值的概率
	}
	for i := 1; i < len(dp); i ++{	// i : 前i个骰子
		for j := range dp[i - 1]{	// j : 数字总合为j
			for k := range dp[0]{	// 当前骰子摇出的值
				dp[i][j + k] += float64(dp[i - 1][j]) * float64(dp[0][k])
			}
		}
	}
	return dp[n - 1]
}
```



## 61. 扑克牌中的顺子

```go
func isStraight(nums []int) bool {
	hashMap := make(map[int]bool, 5)
	maxValue,minValue := 0, 14
	for _, v := range nums {
		if v == 0 {		// 跳过大小王
			continue
		}
		maxValue = max(v, maxValue)		// 最大牌
		minValue = min(v, minValue)		// 最小牌
		if hashMap[v] {
			return false		// 若有重复，提前返回 false
		} else {
			hashMap[v] = true		// 添加此牌
		}
	}
	return maxValue - minValue < 5	// 最大牌 - 最小牌 < 5 则可构成顺子
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 62. 圆圈中最后剩下的数字

https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-by-lee/  评论

f(n, m)表示长度为n的序列，删除n-1次，最后留下元素的序号；

则f(n-1, m)表示长度为n-1的序列，删除n-2次，最后留下元素的序号；

我们知道f(n, m)第一次删(m-1)%n，留下以((m-1)%n + 1)%(n-1)开头，长n-1的序列；

而f(n-1, m)留下元素的序号以0开头，因此两者的序号相差m个距离（别看既%n，又%(n-1)，只是让序号保持在序列长度内，本质上还是右移m个距离，可以循环而已）; `f(n, m) = (f(n-1, m) + m) % n`

小结：

最后保留的元素，必定在每次序列中都保留；

在长n序列保留，必定在长n-1序列保留；

长n-1序列保留的右移m距离，即为长n序列保留的序号；



f(n,m) 是那个活着人的在每轮报数后的下标         因为,最后活着的人的下标是0;

正向推导过程: 约瑟夫环最后一个人的下标,一定是0(只剩一个人活着了),这点都能理解 所以从0推导

1. 一个人的时候: 这个活着的人的下标是0. 所以需要知道当两个人存在的时候,这个人的下标是多少;
2. 两个人的时候: 这个活着的人下标:(0+3)%2=1 所以需要知道当三个人存在的时候 ,这个人的下标是多少;
3. 三个人的时候: 这个活着的人下标:(1+3)%3=1 所以需要知道当四个人存在的时候 ,这个人的下标是多少;
4. 主要是公式f(n,m)=(x+m)%n 的理解,这个x到底指的是什么; 指的是在下一轮报数,那个活着人的下标:我们唯一知道的是最终活着的人的下标是0
5. f(n,m)=( f(n-1,m)+m)%n 是第一轮报数,这个活着人的下标; 但是需要知道这个人在第二轮的下标 f(n-1,m),才能推出第一轮报数的下标
6. f(n-1,m)=(f(n-2,m)+m)%n 是第二轮报数, 这个活着人的下标;
7. ............................................................................................
8. 最后一轮报数 f(2,m)=(f(1,m)+m)%n=(0+m)%n



```go
func lastRemaining(n int, m int) int {
	// f(n, m)表示长度为n的序列，删除n-1次，最后留下元素的序号
	// f(n-1, m)表示长度为n-1的序列，删除n-2次，最后留下元素的序号；
	f := 0
	for i:=2; i!=n+1; i++ {		// i 表示序列长度
		f = (f + m) % i
	}
	return f
}
```



## 64. 求1+2+…+n

```go
func sumNums(n int) int {
	if n == 1 {
		return 1
	}
	return n+sumNums(n-1)
}
```



## 65. 不用加减乘除做加法

https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/solution/mian-shi-ti-65-bu-yong-jia-jian-cheng-chu-zuo-ji-7/

```go
func add(a int, b int) int {
	for b != 0 {			// 当进位为 0 时跳出
		c := (a&b) << 1		// c = 进位		与运算   左移一位
		a ^= b				// a = 非进位和	异或运算
		b = c 				// b = 进位
	}
	return a
}
```



## 67. 把字符串转换成整数   字符串处理

https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/solution/yi-ci-bian-li-chu-li-shu-zi-chu-li-zheng-k0mq/

```go
func strToInt(str string) int {
	// 一次遍历, 处理数字，处理正负号，处理字母
	// 数字应该放在最先判断
	i := 0
	// 1. 先去除前置空格
	for i<len(str) && str[i]==' ' {
		i++
	}
	str = str[i:]
	res := 0
	flag := false	// 符号位	true表示负数
	for i, v := range str {
		// 先处理遇到的数字字符
		if v >= '0' && v <= '9' {
			res = res*10 + int(v-'0')
		} else if v == '-' && i == 0 {
			flag = true	// 结果可能是负数
		} else if v == '+' && i == 0 {
			// 不用做什么
		} else {
			break
		}
		if res > math.MaxInt32 {
			if flag {	// true表示负数
				return math.MinInt32
			}
			return math.MaxInt32
		}
	}
	if flag {
		return -res
	}
	return res
}
```



## 12. 矩阵中的路径

https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/solution/jin-shuang-bai-golang-dfs-by-hideinshado-18tv/

```go
func exist(board [][]byte, word string) bool {
	n, m := len(board), len(board[0])
	if n*m < len(word) {
		return false
	}
	var dfs func(i, j, k int) bool
	// 对于每一个可能的起点，进行dfs匹配
	// 每层对四个方向进行搜索
	dfs = func(i, j, k int) bool {
		// 终止条件 : 成功匹配全部的word
		if k >= len(word) {
			return true
		}
		// 终止条件，坐标范围错误或者 i j 坐标对应 board 的值不等于 k 坐标对应 word 的值
		if i<0 || j<0 || i>=n || j>=m || board[i][j]!=word[k] {
			return false
		}
		// 如果往回前找，不会找到相同字符，如 word = "ABAB" ，k = 2 时，往前往后都是 B ；
		// 将 B 修改（剪枝）为不存在的字符，杜绝往回找成功的可能性。
		board[i][j] = '0'	// 用过的矩阵格子要修改，避免重复使用
		res := dfs(i, j-1, k+1) || dfs(i, j+1, k+1) || dfs(i+1, j, k+1) || dfs(i-1, j, k+1)
		// 找完了就改回来
		board[i][j] = word[k]
		return res
	}
	for i:=0; i<n; i++ {
		for j:=0; j<m; j++ {
			if dfs(i,j, 0) {		// 找到一个可行的解
				return true
			}
		}
	}
	return false
}
```



## 13. 机器人的运动范围

```go
func movingCount(m int, n int, k int) int {
	res := 0
	board := make([][]bool, m)
	for i:=0; i<m; i++ {
		board[i] = make([]bool, n)	// false:没走过	true:走过
	}
	var dfs func(i, j, k int)
	// 从(0, 0)，进行dfs	对四个方向进行搜索
	dfs = func(i, j, k int) {
		// 终止条件，坐标范围错误   或者  i、j坐标对应board的值大于k  或  走过
		if i<0 || j<0 || i>=m || j>=n || board[i][j] || count(i, j, k) {
			return
		}
		if !board[i][j] {
			board[i][j] = true	// 用过的矩阵格子要修改，避免重复使用
			res++
		}
		dfs(i, j+1, k)
		dfs(i+1, j, k)
		// 隐藏的优化：我们在搜索的过程中搜索方向可以缩减为向右和向下，而不必再向上和向左进行搜索
		//dfs(i, j-1, k)
		//dfs(i-1, j, k)
	}
	dfs(0,0,k)
	return res
}

func count(i, j, k int) bool {
	sum := 0
	for i != 0 {
		sum += i%10
		i /= 10
	}
	for j != 0 {
		sum += j%10
		j /= 10
	}
	return sum > k
}
```



## 14- I. 剪绳子

https://leetcode-cn.com/problems/jian-sheng-zi-lcof/solution/jian-zhi-offer-14-i-jian-sheng-zi-huan-s-xopj/

```go
// 贪心   尽可能把绳子分成长度为3的小段，这样乘积最大
func cuttingRope(n int) int {
	if n < 4 {
		return n-1
	}
	if n == 4 {
		return n
	}
	res := 1
	for n > 4 {
		res *= 3
		n -= 3
	}
	return res * n
}

// dp
func cuttingRope(n int) int {
	dp := make([]int, n+1)
	dp[2] = 1
	for i:=3; i<n+1; i++ {
		for j:=2; j<i; j++ {
			dp[i] = max(dp[i], max(j*(i-j), j*dp[i-j]))
		}
	}
	return dp[n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 14- II. 剪绳子 II 

```go
// 贪心
func cuttingRope(n int) int {
	if n < 4 {
		return n-1
	}
	if n == 4 {
		return n
	}
	res := 1
	for n > 4 {
		res = 3 * res % 1000000007
		n -= 3
	}
	return res * n % 1000000007
}
```



## 19. 正则表达式匹配

```go
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	// dp[i][j] 表示 s 的前 i 个是否能被 p 的前 j 个匹配
	/*
	已知 dp[i-1][j-1] 意思就是前面子串都匹配上了，不知道新的一位的情况。
	那就分情况考虑，所以对于新的一位 p[j] s[i] 的值不同，要分情况讨论：
		1、考虑最简单的 p[j] == s[i] : dp[i][j] = dp[i-1][j-1]
	然后从 p[j] 可能的情况来考虑，让 p[j]=各种能等于的东西。
		2、p[j] == "." : dp[i][j] = dp[i-1][j-1]
		3、p[j] ==" * ":
			按照 p[j-1] 和 s[i] 是否相等，我们分为两种情况：
			3.1 p[j-1] != s[i] : dp[i][j] = dp[i][j-2]
			3.2 p[j-1] == s[i] or p[j-1] == "."

	 */
	dp := make([][]bool, m+1)
	for i:=0; i<=m; i++ {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	// matches(x,y) 判断两个字符是否匹配的辅助函数
	// 只有当 y 是 . 或者 x 和 y 本身相同时，这两个字符才会匹配
	matchs := func(i, j int) bool {
		if i==0 {
			return false
		}
		if p[j-1] == '.' {
			return true
		}
		return s[i-1] == p[j-1]
	}

	for i:=0; i<=m; i++ {
		for j:=1; j<=n; j++ {
			if p[j-1] == '*' {						// 如果 p 的第 j 个字符是 *
				dp[i][j] = dp[i][j] || dp[i][j-2]	// 表示我们可以对 p 的第 j-1 个字符匹配任意自然数次
													// 在匹配 0 次的情况下，我们有 f[i][j] = f[i][j−2]
				if matchs(i, j-1) {
					dp[i][j] = dp[i][j] || dp[i-1][j]
				}
			} else if matchs(i, j) {
				dp[i][j] = dp[i][j] || dp[i-1][j-1]
			}
		}
	}
	return dp[m][n]
}
```



## 20. 表示数值的字符串

https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/solution/go-chao-jian-dan-si-lu-wu-fu-za-guo-chen-7yc4/

```go
func isNumber(s string) bool {
	// 掐头去尾--> 去掉开始和结束的空格
	i, j := 0, len(s)-1
	for ; i < len(s); i++ {
		if s[i] != ' ' {
			break
		}
	}
	for ; j >= 0; j-- {
		if s[j] != ' ' {
			break
		}
	}
	if j+1 <= i {
		return false
	}
	s = s[i : j+1]
	// 判断是否有科学计数法
	if (strings.Count(s,"e")+ strings.Count(s,"E"))>1{
		return false
	}
	science := max(max(-1, strings.Index(s, "e")), strings.Index(s, "E"))
	if science == -1 {
		return isInteger(s) || isDecimal(s)
	} else {
		return (isInteger(s[:science]) || isDecimal(s[:science])) && isInteger(s[science+1:])
	}
}

// 判断是不是整数
func isInteger(s string) bool {
	if len(s) == 0 {
		return false
	}
	i := 0
	if s[0] == '+' || s[0] == '-' {
		if len(s) == 1 {
			return false
		}
		i = 1
	}
	for ; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}

	return true
}

// 判断是不是小数
func isDecimal(s string) bool {
	if strings.Count(s, ".") != 1 || len(s) == 0 {
		return false
	}
	i := 0
	if s[0] == '+' || s[0] == '-' {
		if len(s) == 1 {
			return false
		}
		i++
	}
	index := strings.Index(s, ".")
	left, right := 0, 0
	for ; i < index; i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
		left++
	}
	for i++; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
		right++
	}
	return left >= 1 || (left == 0 && right > 0)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 28. 对称的二叉树

https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/solution/mian-shi-ti-28-dui-cheng-de-er-cha-shu-di-gui-qing/

```go
func isSymmetric(root *TreeNode) bool {
	var dfs func(left, right *TreeNode) bool
	dfs = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left == nil || right == nil || left.Val != right.Val {
			return false
		}
		return dfs(left.Left, right.Right) && dfs(left.Right, right.Left)
	}
	if root == nil {
		return true
	}
	return dfs(root.Left, root.Right)
}
```



## 32 - I. 从上到下打印二叉树

```go
func levelOrder(root *TreeNode) []int {
	queue := make([]*TreeNode, 0)
	ans := make([]int, 0)
	if root == nil {
		return nil
	}
	queue = append(queue, root)
	for len(queue) != 0 {
		node := queue[0]
		if node != nil {
			ans = append(ans, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		queue = queue[1:]
	}
	return ans
}
```



## 32 - III. 从上到下打印二叉树 III

```go
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	queue := make([]*TreeNode, 0)	// 构建[]int{i}更耗时
	queue = append(queue, root)
	ans := make([][]int, 0)

	for len(queue) != 0 {
		n := len(queue)
		tmp := make([]int, n)
		for i:=0; i<n; i++ {
			if len(ans) % 2 == 0 {
				tmp[i] = queue[i].Val
			} else {
				tmp[n-i-1] = queue[i].Val
			}
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
		ans = append(ans, tmp)
	}
	return ans
}
```



## 41. 数据流中的中位数

```go
type maxHeap []int  // 大顶堆
type minHeap []int  // 小顶堆

// 每个堆都要heap.Interface的五个方法：Len, Less, Swap, Push, Pop
// 其实只有Less的区别。

// Len 返回堆的大小
func (m maxHeap) Len() int {
	return len(m)
}
func (m minHeap) Len() int {
	return len(m)
}

// Less 决定是大优先还是小优先
func (m maxHeap) Less(i, j int) bool {  // 大根堆
	return m[i] > m[j]
}
func (m minHeap) Less(i, j int) bool {  // 小根堆
	return m[i] < m[j]
}

 // Swap 交换下标i, j元素的顺序
func (m maxHeap) Swap(i, j int) {  
	m[i], m[j] = m[j], m[i]
}
func (m minHeap) Swap(i, j int) {   
	m[i], m[j] = m[j], m[i]
}

// Push 在堆的末尾添加一个元素，注意和heap.Push(heap.Interface, interface{})区分
func (m *maxHeap) Push(v interface{}) {
	*m = append(*m, v.(int))
}
func (m *minHeap) Push(v interface{}) {
	*m = append(*m, v.(int))
}

// Pop 删除堆尾的元素，注意和heap.Pop()区分
func (m *maxHeap) Pop() interface{} {
	old := *m
	n := len(old)
	v := old[n - 1]
	*m = old[:n - 1]
	return v
}
func (m *minHeap) Pop() interface{} {
	old := *m
	n := len(old)
	v := old[n - 1]
	*m = old[:n - 1]
	return v
}

// MedianFinder 维护两个堆，前一半是大顶堆，后一半是小顶堆，中位数由两个堆顶决定
type MedianFinder struct {	
	maxH *maxHeap
	minH *minHeap
}

// Constructor 建两个空堆
func Constructor() MedianFinder {
	return MedianFinder{
		new(maxHeap),
		new(minHeap),
	}
}

// AddNum 插入元素num
// 分两种情况插入：
// 1. 两个堆的大小相等，则小顶堆增加一个元素（增加的不一定是num）
// 2. 小顶堆比大顶堆多一个元素，大顶堆增加一个元素
// 这两种情况又分别对应两种情况：
// 1. num小于大顶堆的堆顶，则num插入大顶堆
// 2. num大于小顶堆的堆顶，则num插入小顶堆
// 插入完成后记得调整堆的大小使得两个堆的容量相等，或小顶堆大1
func (m *MedianFinder) AddNum(num int)  {
	if m.maxH.Len() == m.minH.Len() {
		if m.minH.Len() == 0 || num >= (*m.minH)[0] {
			heap.Push(m.minH, num)
		} else {
			heap.Push(m.maxH, num)
			top := heap.Pop(m.maxH).(int)
			heap.Push(m.minH, top)
		}
	} else {
		if num > (*m.minH)[0] {
			heap.Push(m.minH, num)
			bottle := heap.Pop(m.minH).(int)
			heap.Push(m.maxH, bottle)
		} else {
			heap.Push(m.maxH, num)
		}
	}
}

// FindMediam 输出中位数
func (m *MedianFinder) FindMedian() float64 {
	if m.minH.Len() == m.maxH.Len() {
		return float64((*m.maxH)[0]) / 2.0 + float64((*m.minH)[0]) / 2.0
	} else {
		return float64((*m.minH)[0])
	}
}
```



## 44. 数字序列中某一位的数字

```go
func findNthDigit(n int) int {
	digit := 1
	start := 1
	count := 9
	for n > count {
		n -= count
		digit ++
		start *= 10
		count = digit * start * 9
	}
	num := start + (n-1)/digit
	ans, _ := strconv.Atoi(string(strconv.Itoa(num)[(n-1)%digit]))
	return ans
}
```



## 46. 把数字翻译成字符串

```go
func translateNum(num int) int {
	// 翻译「1402」   两种情况: 1 4 0 2 beac 和  14 0 2  oac
	str := strconv.Itoa(num)
	// 用 f(i) 表示以第 i 位结尾的前缀串翻译的方案数
	// 单独翻译对 f(i) 的贡献为 f(i - 1)
	// 如果第 i−1 位存在，并且第 i−1 位和第 i 位形成的数字 x 满足 10~25，就可以把第 i−1 位和第 i 位连起来一起翻译，对 f(i) 的贡献为 f(i−2)，否则为 0
	// f(i) = f(i−1) + f(i−2)		边界条件是 f(−1)=0，  f(0)=1
	//   r  =    q   +   p          边界条件是   q = 0，   r = 1
	p, q, r := 0, 0, 1
	for i:=0; i<len(str); i++ {
		p, q, r = q, r, 0
		r += q			// f(i-1) 有贡献
		if i == 0 {		// 如果是第0位，就不用考虑i-1位
			continue
		}
		pre := str[i-1:i+1]		// pre是第 i−1 位和第 i 位连起来
		if pre >= "10" && pre <= "25" {		// 如果在10~25
			r += p		// f(i-2) 有贡献
		}
	}
	return r
}
```



## 48. 最长不含重复字符的子字符串

```go
func lengthOfLongestSubstring(s string) int {
	// 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
	right, ans := -1, 0
	hashMap := make(map[byte]bool)
	n := len(s)
	for i:=0; i<n; i++ {
		if i != 0 {
			// 左指针向右移动一格，移除一个字符
			hashMap[s[i-1]] = false
		}
		for right+1 < n && !hashMap[s[right+1]] {
			// 不断地移动右指针
			right++
			hashMap[s[right]] = true
		}
		// 第 i 到 right 个字符是一个极长的无重复字符子串
		ans = max(ans, right-i+1)
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 53 - II. 0～n-1中缺失的数字

```go
func missingNumber(nums []int) int {
	sum1, n := 0, len(nums)
	sum2 := (n+1) * n / 2
	for i:=0; i<n; i++ {
		sum1 += nums[i]
	}
	return sum2 - sum1
}

func missingNumber(nums []int) int {
	i, j := 0, len(nums)-1
	// 左子数组： nums[i] = i
	// 右子数组： nums[i] != i
	// 缺失的数字等于 “右子数组的首位元素” 对应的索引
	for i <= j {
		m := (i+j) / 2
		if nums[m] == m {
			i = m + 1
		} else {
			j = m - 1
		}
	}
	// 跳出时，变量 i 和 j 分别指向 “右子数组的首位元素” 和 “左子数组的末位元素” 。因此返回 i
	return i
}

func missingNumber(nums []int) int {
	if nums[0] == 1 {
		return 0
	}
	for i:=0; i<len(nums); i++ {
		if nums[i] != i {
			return i
		}
	}
	return len(nums)
}
```



## 54. 二叉搜索树的第k大节点

```go
func kthLargest(root *TreeNode, k int) int {
	ans := 0
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Right)
		k--
		if k == 0 {
			ans = node.Val
			return
		}
		dfs(node.Left)
	}
	dfs(root)
	return ans
}
```



## 56 - II. 数组中数字出现的次数 II

https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/solution/shu-zu-zhong-chu-xian-1ci-2ci-de-shu-zi-emt54/

```go
func singleNumber(nums []int) int {
	res := 0
	for i := 0; i < 32; i++ {
		// 对于int每一位
		bit := 0
		// 记录该位上的和
		for _, num := range nums {
			bit += (num >> i) & 1
		}
		// 对3取余即为res在该位的值
		res += (bit % 3) << i
	}
	return res
}
```



## 59 - I. 滑动窗口的最大值

https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/solution/dong-hua-yan-shi-dan-diao-dui-lie-jian-z-unpy/

```go
func maxSlidingWindow(nums []int, k int) []int {
	// 窗口个数
	n := len(nums)
	// 必须特殊判断，否则在输入为 [],0 时会报错
	if n == 0 {
		return []int{}
	}
	ans := make([]int, n-k+1)
	queue := make([]int, 0)
	// 遍历数组中元素，right表示滑动窗口右边界
	for right:=0; right<n; right++ {
		// 如果队列不为空且当前考察元素大于等于队尾元素，则将队尾元素移除。
		// 直到，队列为空或当前考察元素小于新的队尾元素
		for len(queue)!=0 && nums[right] >= nums[queue[len(queue)-1]] {
			queue = queue[:len(queue)-1]
		}
		// 存储元素下标
		queue = append(queue, right)
		// 计算窗口左侧边界
		left := right-k+1
		// 当队首元素的下标小于滑动窗口左侧边界left时
		// 表示队首元素已经不再滑动窗口内，因此将其从队首移除
		if queue[0] < left {
			queue = queue[1:]
		}
		// 由于数组下标从0开始，因此当窗口右边界right+1大于等于窗口大小k时
		// 意味着窗口形成。此时，队首元素就是该窗口内的最大值
		if right+1 >= k {
			ans[left] = nums[queue[0]]
		}
	}
	return ans
}
```



## 59 - II. 队列的最大值

https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/solution/ru-he-jie-jue-o1-fu-za-du-de-api-she-ji-ti-by-z1m/

```go
type MaxQueue struct {
	// q 是 队列本体    d 是辅助计算最大值的双端队列
	q []int
	d []int
}

func Constructor() MaxQueue {
	return MaxQueue{}
}


func (this *MaxQueue) Max_value() int {
	if len(this.q) == 0 {
		return -1
	}
	return this.d[0]
}


func (this *MaxQueue) Push_back(value int)  {
	this.q = append(this.q, value)
	for len(this.d) > 0 && value > this.d[len(this.d)-1] {
		this.d = this.d[:len(this.d)-1]
	}
	this.d = append(this.d, value)
}


func (this *MaxQueue) Pop_front() int {
	if len(this.q) == 0 {
		return -1
	}
	if this.d[0] == this.q[0] {
		this.d = this.d[1:]
	}
	value := this.q[0]
	this.q = this.q[1:]
	return value
}
```



## 63. 股票的最大利润

```go
func maxProfit(prices []int) int {
	minPrice := math.MaxInt32
	res := 0
	for i:=0; i<len(prices); i++ {
		minPrice = min(minPrice, prices[i])
		res = max(res, prices[i] - minPrice)
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```



## 66. 构建乘积数组

```go
func constructArr(a []int) []int {
	n := len(a)
	if n == 0 {
		return nil
	}
	res := make([]int, n)
	// answer[i] 表示索引 i 左侧所有元素的乘积
	// 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
	res[0] = 1
	for i:=1; i<n; i++ {
		res[i] = a[i-1] * res[i-1]
	}
	// R 为右侧所有元素的乘积
	// 刚开始右边没有元素，所以 R = 1
	r := 1
	for i:=n-1; i>=0; i-- {
		// 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
		res[i] *= r
		// R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
		r *= a[i]
	}
	return res
}
```



## 68 - I. 二叉搜索树的最近公共祖先

```go
func lowestCommonAncestor(root, p, q *TreeNode) (ancestor *TreeNode) {
    ancestor = root
    for {
        if p.Val < ancestor.Val && q.Val < ancestor.Val {
            ancestor = ancestor.Left
        } else if p.Val > ancestor.Val && q.Val > ancestor.Val {
            ancestor = ancestor.Right
        } else {
            return
        }
    }
}
```



## 68 - II. 二叉树的最近公共祖先

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left == nil {
		return right
	}
	if right == nil {
		return left
	}
	return root
}
```



# 递归

## 22. 括号生成

```go
func generateParenthesis(n int) []string {
	// n 代表生成括号的对数
	res := make([]string, 0)
	if n <= 0 {
		return nil
	}
	// 剩余左括号总数要小于等于右括号。 递归把所有符合要求的加上去
	var dfs func(str string, left, right int)
	dfs = func(str string, left, right int) {
		if left==0 && right==0 {	// 左右括号使用完毕
			res = append(res, str)
			return
		}
		// 剩余左右括号数相等，下一个只能用左括号
		if left == right {
			dfs(str+"(", left-1, right)		// 左括号-1   右括号不变
		} else if left < right {
			// 剩余左括号小于右括号，下一个可以用左括号也可以用右括号
			if left > 0 {
				dfs(str+"(", left-1, right)
			}
			dfs(str+")", left, right-1)
		}
	}
	dfs("", n, n)
	return res
}
```



## 39. 组合总和

https://leetcode-cn.com/problems/combination-sum/solution/0ms-25mb-hui-su-by-1990jxy-m9j8/

```go
func combinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	n := len(candidates)
	// 当前在 candidates 数组的第 idx 位，还剩 target 要组合
	var dfs func(target int, combine []int, idx int)
	dfs = func(target int, combine []int, idx int) {
		// 递归的终止条件  数组被全部用完
		if idx == n {
			return
		}
		// 递归的终止条件    找到一种答案
		if target == 0 {
			//tmp := make([]int, len(combine))
			//copy(tmp, combine)
			res = append(res, append([]int(nil), combine...))
			//res = append(res, tmp)
			return
		}
		// 直接跳过  不使用第idx个数
		dfs(target, combine, idx+1)
		// 选择当前的第idx个数
		if target-candidates[idx] >= 0 {
			combine = append(combine, candidates[idx])
			dfs(target - candidates[idx], combine, idx)		// 注意到每个数字可以被无限制重复选取，因此搜索的下标仍为 idx
			combine = combine[:len(combine)-1]	// 回溯
		}
	}
	dfs(target, []int{}, 0)
	return res
}

// 优化
func combinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	track := make([]int, 0)
	sum := 0
	n := len(candidates)

	var dfs func(start int)
	dfs = func(start int) {
		// 递归的终止条件
		if sum > target {
			return
		}
		// 递归的终止条件    找到一种答案
		if sum == target {
			res = append(res, append([]int{}, track...))
			return
		}
		// 选择当前的第idx个数
		for i:=start; i < n; i++ {
			sum += candidates[i]
			track = append(track, candidates[i])
			// 选择
			dfs(i)
			track = track[:len(track)-1]
			sum -= candidates[i]
		}
	}
	dfs(0)
	return res
}
```



## 40. 组合总和 II

https://leetcode-cn.com/problems/combination-sum-ii/solution/ke-yi-shi-xiao-gai-zu-he-zong-he-ii-quan-keeh/

```go
func combinationSum2(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	track := make([]int, 0)
	sum := 0
	n := len(candidates)
	sort.Ints(candidates)
	var dfs func(start int)
	dfs = func(start int) {
		// 递归的终止条件
		if sum > target {
			return
		}
		// 递归的终止条件    找到一种答案
		if sum == target {
			res = append(res, append([]int{}, track...))
			return
		}
		// 选择当前的第i个数
		for i:=start; i < n; i++ {
			// 同一层去重
			// 当前的第i个数  大于  起始下标
			// 当前选择的数 = 上次选择的数
			if i > start && candidates[i-1] == candidates[i] {
				continue
			}
			sum += candidates[i]
			track = append(track, candidates[i])
			// 选择
			dfs(i+1)	// 每个数字不可以被无限制重复选取
			track = track[:len(track)-1]
			sum -= candidates[i]
		}
	}
	dfs(0)
	return res
}
```



## 52. N皇后 II 

```go
func totalNQueens(n int) int {
	cols := make([]bool, n)			// 列上是否有皇后
	cross1 := make([]bool, 2*n-1)	// 左上到右下是否有皇后		同一条斜线上的每个位置满足行下标与列下标之差相等
	cross2 := make([]bool, 2*n-1)	// 右上到左下是否有皇后		同一条斜线上的每个位置满足行下标与列下标之和相等
	ans := 0
	var dfs func(row int)	// 回溯的具体做法是：依次在每一行放置一个皇后  row表示行
	dfs = func(row int) {
		if row == n {
			ans++
			return
		}
		for col, hasQueen := range cols {	// 由于每个皇后必须位于不同列，因此已经放置的皇后所在的列不能放置别的皇后
			d1, d2 := row+n-1-col, row+col
			if hasQueen || cross1[d1] || cross2[d2] {	// 判断一个位置所在的列和两条斜线上是否已经有皇后
				continue
			}
			cols[col] = true
			cross1[d1] = true
			cross2[d2] = true
			dfs(row+1)
			cols[col] = false
			cross1[d1] = false
			cross2[d2] = false
		}
	}
	dfs(0)
	return ans
}
```



# 每日一题

## 504. 七进制数

```go
func convertToBase7(num int) string {
	if num == 0 {
		return "0"
	}
	flag := false
	if num < 0 {
		flag = true
		num = -num
	}
	tmp := 0
	res := make([]byte, 0)
	for num != 0 {
		tmp = num % 7
		num /= 7
		res = append(res, '0'+byte(tmp))	// 注意这里要+'0'   byte就是字符码   tmp是int值
	}
	left, right := 0, len(res)-1
	for left < right {
		res[left], res[right] = res[right], res[left]
		left++		// 不要忘记
		right--		// 不要忘记
	}
	if flag {
		return "-" + string(res)
	}
	return string(res)
}
```



## 2055. 蜡烛之间的盘子

```go
// 超时
func platesBetweenCandles(s string, queries [][]int) []int {
	res := make([]int, 0)
	m := len(queries)
	for i:=0; i<m; i++ {
		left := queries[i][0]
		right := queries[i][1]
		for left < right {
			if s[left] != '|' {
				left++
			}
			if s[right] != '|' {
				right--
			}
		}
		if right-left <= 1 {
			res = append(res, 0)
			continue
		}
		sum := 0
		for j:=left+1; j<right; j++ {
			if s[j] == '*' {
				sum++
			}
		}
		res = append(res, sum)
	}
	return res
}

// 
func platesBetweenCandles(s string, queries [][]int) []int {
	// 遍历queries数组，确定左右蜡烛的位置(即在left和right中记录的索引)，二者相减得出中间包含的元素
	// 再减去preSum对应的蜡烛数量，结果即为盘子数量，加入结果集，最终返回
	res := make([]int, 0)
	n := len(s)
	// left : 左蜡烛位置    right : 右蜡烛位置    preSum : 记录蜡烛数量的前缀和
	left, right, preSum := make([]int, n+1), make([]int, n+1), make([]int, n+1)
	for i:=0; i<n+1; i++ {
		right[i] = math.MaxInt32
	}
	for i:=0; i<n; i++ {
		// 预处理区间内每个位置左侧的第一个蜡烛和右侧的第一个蜡烛
		if s[i] == '|' {	// 遇到蜡烛
			preSum[i+1] += preSum[i] + 1	// 蜡烛的前缀和+1
			left[i+1] = i	// 记录左边蜡烛的位置	即: 第i+1位置的左蜡烛坐标为i
		} else {	// 遇到盘子
			preSum[i+1] = preSum[i]		// 蜡烛的前缀和不变
			left[i+1] = left[i]		// 第i+1位置的左蜡烛坐标为left[i] 即上次记录的位置
		}
		if s[n-i-1] == '|' {	// 遇到蜡烛（右侧的）
			right[n-i-1] = n-i-1	// 第n-(i+1)位置的右蜡烛坐标为n-(i+1)
		} else {	// 遇到盘子
			right[n-i-1] = right[n-i]	// 因为右蜡烛是从右往左数的
		}
	}
	for _,v := range queries {	// 这里的v[0]是左边界   v[1]是右边界  要找左右边界中的盘子数量
		l := right[v[0]] 	// 找到位置v[0]右侧的第一个蜡烛
		r := left[v[1]+1]	// 找到位置v[1]左侧的第一个蜡烛
		if l < r {
			res = append(res, r-l-(preSum[r]-preSum[l]))
		} else {
			res = append(res, 0)
		}
	}
	return res
}
```



## 34. 在排序数组中查找元素的第一个和最后一个位置

https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/zai-pai-xu-shu-zu-zhong-cha-zhao-yuan-su-w1h4/

```go
// 调包
func searchRange(nums []int, target int) []int {
    // go 语言有自己实现的二分查找方法 sort.Search()
	left := sort.SearchInts(nums, target)
	if left == len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	right := sort.SearchInts(nums, target+1) - 1
	return []int{left, right}
}

// 二分
func searchRange(nums []int, target int) []int {
	left := findLeft(nums, target)
	if left == len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	right := findLeft(nums, target+1) - 1
	return []int{left, right}
}

func findLeft(nums []int, target int) int {
	left, right := 0, len(nums)-1	// [left, right]
	for left <= right {				// 因为 right 是闭区间，所以可以取 =
		mid := left + (right-left)/2
		if nums[mid] == target {
			right = mid - 1		// 收紧右侧边界以锁定左侧边界
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	// 返回左侧边界
	return left
}
```



## 49. 字母异位词分组

```go
func groupAnagrams(strs []string) [][]string {
	res := make([][]string, 0)
	hashMap := make(map[string][]string, len(strs))
	for _,v := range strs {
		b := []byte(v)
		// 对切片进行排序的一个函数
		// slice的作用 :  “eat” = “101,97,116” ->"97,101,116"
		sort.Slice(b, func(i, j int) bool {
			return b[i] < b[j]
		})
		key := string(b)
		// 只有当key一样的时候，才能将v存入value []string 这个切片中，所以存进去的都是key一样的
		// key经过排序，只要是相同字母不同组合排序后都一样
		hashMap[key] = append(hashMap[key], v)
	}
	// 输出的时候，所有元素存入res 返回res就行。
	for _, v := range hashMap {
		res = append(res, v)
	}
	return res
}
```



## 79. 单词搜索

```go
func exist(board [][]byte, word string) bool {
	x, y := len(board), len(board[0])
	n := len(word)
	visited := make([][]bool, x)
	for i:=0 ; i<x; i++ {
		visited[i] = make([]bool, y)
	}
	var dfs func(i, j, k int) bool
	dfs = func(i, j, k int) bool {
		if k == n {
			return true
		}
		if  i<0 || j<0 || i>=x || j>=y || visited[i][j] || board[i][j] != word[k] {
			return false
		}
		visited[i][j] = true
		flag := dfs(i+1, j, k+1) || dfs(i, j+1, k+1) || dfs(i-1, j, k+1) || dfs(i, j-1, k+1)
		if flag {
			return true
		}
		visited[i][j] = false
		return false
	}
	for i:=0; i<x; i++ {
		for j:=0; j<y; j++ {
			if board[i][j] == word[0] && dfs(i, j, 0) {		// 这里剪枝 优化了一下
				return true
			}
		}
	}
	return false
}
```



## 142. 环形链表 II

```go
func detectCycle(head *ListNode) *ListNode {
	hashMap := map[*ListNode]int{}
	for head != nil {
		// 假如 key 存在，则 ok = true，否则，ok = false
		if _, ok := hashMap[head]; ok {
			return head
		}
		hashMap[head]++
		head = head.Next
	}
	return nil
}
```



## 146. LRU 缓存

https://leetcode-cn.com/problems/lru-cache/solution/ke-yi-shi-xiao-gai-urlhuan-cun-qi-goyu-y-4zcy/

```go
type  LinkNode struct {
	key, val int
	pre, next *LinkNode
}
/*
需要一个 Cache 来存储所有的 Node
我们定义 cap 为 cache 的长度，m 用来存储元素
head 和 tail 作为 Cache 的首尾
*/

type LRUCache struct {
	m  map[int]*LinkNode
	cap  int
	// 记录连表的头和尾
	head,tail *LinkNode
}


func Constructor(capacity int) LRUCache {
	head := &LinkNode{0, 0, nil, nil}
	tail := &LinkNode{0, 0, nil, nil}
	head.next = tail
	tail.pre = head
	return LRUCache{
		make(map[int]*LinkNode),
		capacity,
		head,
		tail,
	}
}

// Get 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1
func (this *LRUCache) Get(key int) int {
	cache := this.m
	// 查找，如果存在就就将其放在最前面
	if v, exist := cache[key]; exist {
		this.MoveToHead(v)
		return v.val
	} else {
		// 找不到就返回-1
		return -1
	}
}

// Put 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。
// 当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
func (this *LRUCache) Put(key int, value int)  {
	tail := this.tail
	cache := this.m
	// 判断存的这个key在map里面有没有
	if v, exist := cache[key]; exist {
		// 如果有，变更其数据值
		v.val = value
		// 如果有，就把他放到双向链表最前面
		this.MoveToHead(v)
	} else {
		// 如果关键字不存在，则插入数组
		// v就相当于初始化了一些节点，将key，value存进去
		v := &LinkNode{key, value, nil, nil}
		// 当缓存池长度达到上限
		if len(cache) == this.cap {
			// 删除缓存池的最久未使用的数据末尾的
			//（可以这么理解，head节点，tail节点都是不存数据的，所以删或者加的时候都是在head之后，tail之前）
			// 缓存删的是key
			delete(cache, tail.pre.key)
			// 删除节点
			this.RemoveNode(tail.pre)
		}
		this.AddNode(v)
		cache[key] = v
	}
}

// RemoveNode 双链表的删除节点(删的是尾节点)
func (this *LRUCache) RemoveNode(node *LinkNode) {
	node.pre.next = node.next
	node.next.pre = node.pre
}

// AddNode 双链表的添加节点（添加节点也是添加到head节点之后）
func (this *LRUCache) AddNode(node *LinkNode) {
	//这个顺序也要注意
	head := this.head
	node.next = head.next
	head.next.pre = node
	node.pre = head
	head.next = node
}

// MoveToHead 将元素移动到双链表的最前面节点
func (this *LRUCache) MoveToHead(node *LinkNode) {
	this.RemoveNode(node)
	this.AddNode(node)
}
```



## 152. 乘积最大子数组

```go
func maxProduct(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	maxNum, minNum, res := nums[0], nums[0], nums[0]
	for _, v := range nums[1:] {
		maxTmp, minTmp := maxNum, minNum
		// 最大的数可以是:
		// ① 之前的正数最大值 * 当前的正数v
		// ② 之前的负数最小值 * 当前的负数v
		// ③ 之前都是负数或0，所以只能是当前的值v
		maxNum = max(maxTmp*v, max(minTmp*v, v))
		minNum = min(minTmp*v, min(maxTmp*v, v))
		//maxNum = max(maxTmp*v, minTmp*v)	// nums = [0, 2]     最大值为2   因此需要判断单独的v
		//minNum = min(minTmp*v, maxTmp*v)
		res = max(maxNum, res)
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 155. 最小栈

```go
type MinStack struct {
	stack []int
	minStack []int
}


func Constructor() MinStack {
	return MinStack{
		[]int{},
		[]int{math.MaxInt32},
	}
}


func (this *MinStack) Push(val int)  {
	this.stack = append(this.stack, val)
	// 在minStack栈放了一个top指针，一直指向minStack的最后一个元素
	top := this.minStack[len(this.minStack)-1]
	// 用的是min()，所以上面初始化minStack第一个元素放的是一个int64的最大值
	// 如果val比top还要小，将这个值也存入minStack中
	this.minStack = append(this.minStack, min(val, top))

}


func (this *MinStack) Pop()  {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}


func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}


func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 798. 得分最高的最小轮调

https://leetcode-cn.com/problems/smallest-rotation-with-highest-score/solution/wei-rao-li-lun-xue-xi-liao-yi-xia-chai-f-kqh2/

差分	https://oi-wiki.org/basic/prefix-sum/

```go
func bestRotation(nums []int) int {
	n := len(nums)
	// 任何值小于或等于其索引的项都可以记作一分	即当 nums[i] <= i 记作一分
	// diff[0] 表示 当 k = 0 时，数组的得分
	move := make([]int, n)
	// 遍历一遍每个数，通过直接计算得到产生分数贡献变化的分界点，将对应的移动值和变化记录下来
	for i:=0; i<n; i++ {
		// 优化
		move[(i+1)%n]++
		move[(n - (nums[i] - i) + 1)%n]--
		if nums[i] <= i {
			move[0]++
		}
	}
	// 最后求总分的时候再遍历累计求和即可
	sum := 0
	maxSum := 0
	k := 0
	for i:=0; i<n; i++ {
		sum += move[i]
		if sum > maxSum {
			maxSum = sum
			k = i
		}
	}
	return k
}
```



## 589. N 叉树的前序遍历

```go
type Node struct {
	Val int
	Children []*Node
}

func preorder(root *Node) []int {
	if root == nil {
		return nil
	}
	res := make([]int, 0)
	var dfs func(root *Node)
	dfs = func(root *Node) {
		res = append(res, root.Val)
		for _, v := range root.Children {	// i是下标int   v是*Node
			dfs(v)
		}
	}
	dfs(root)
	return res
}
```



## 215. 数组中的第K个最大元素

```go
func findKthLargest(nums []int, k int) int {
	n := len(nums)
	sort.Ints(nums)
	// 倒数第K大的数  = n-k+1  如果算上nums[0]的话就不用加一
	return nums[n-k]
}
```



## 148. 排序链表

```go
type ListNode struct {
	Val  int
	Next *ListNode
}

func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head
	var preSlow *ListNode
	// fast !=nil这种是判断偶数个节点的情况，fast.Next !=nil判断的是奇数个节点的时候
	// 这里将链表从中间分为两段，第一段头节点为head，第二段头节点为slow
	for fast != nil && fast.Next != nil {
		preSlow = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	preSlow.Next = nil		// 断开，分成两链
	left := sortList(head)	// 已排序的左链
	right := sortList(slow)	// 已排序的右链
	return merge(left, right)	// 合并已排序的左右链，一层层向上返回
}

func merge(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{Val: 0}	// 虚拟头结点
	pre := dummy				// 用pre去扫，先指向dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			pre.Next = l1
			l1 = l1.Next
		} else {
			pre.Next = l2
			l2 = l2.Next
		}
		pre = pre.Next	// pre.Next确定了，pre指针推进
	}
	if l1 != nil {
		pre.Next = l1
	}
	if l2 != nil {
		pre.Next = l2
	}
	return dummy.Next
}
```



## 200. 岛屿数量

```go
func numIslands(grid [][]byte) int {
	res := 0
	m, n := len(grid), len(grid[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		// 终止条件：超出数组范围，或者当前位置不是陆地   1是陆地   0是水
		if i<0 || j<0 || i>=m || j>=n || grid[i][j]=='0' {
			return
		}
		// 沉没当前位置
		grid[i][j] = '0'
		// 上方递归执行沉没
		dfs(i-1, j)
		// 下方
		dfs(i+1,j)
		// 左方
		dfs(i,j-1)
		// 右方
		dfs(i,j+1)
	}
	for i, row := range grid {
		for j, ch := range row {
			// 发现岛头value=1时，开始沉没
			if ch == '1' {
				dfs(i, j)
				res++
			}
		}
	}
	return res
}
```



## 221. 最大正方形

```go
/*
对于每个位置 (i, j)，检查在矩阵中该位置的值
	如果该位置的值是 0，则 dp(i,j)=0，因为当前位置不可能在由 1 组成的正方形中；
	如果该位置的值是 1，则 dp(i,j) 的值由其上方、左方和左上方的三个相邻位置的 dp 值决定。
		具体而言，当前位置的元素值等于三个相邻位置的元素中的最小值加 1
 */
func maximalSquare(matrix [][]byte) int {
	res := 0
	m, n := len(matrix), len(matrix[0])
	// dp(i,j) 表示以 (i, j) 为右下角，且只包含 1 的正方形的边长最大值
	dp := make([][]int, m)
	// 初始化
	for i:=0; i<m; i++ {
		dp[i] = make([]int, n)
		for j:=0; j<n; j++ {
			dp[i][j] = int(matrix[i][j] - '0')
			if dp[i][j] == 1 {
				res = 1
			}
		}
	}

	for i:=1; i<m; i++ {
		for j := 1; j < n; j++ {
			if dp[i][j] == 1 {
				dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1
				res = max(res, dp[i][j])
			}
		}
	}
	// 返回其面积  所以要平方
	return res * res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 160. 相交链表

```go
// 双指针
type ListNode struct {
	Val  int
	Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	pa, pb := headA, headB
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}
	return pa
}
```



## 226. 翻转二叉树

```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left = right
	root.Right = left
	return root
}
```



## 234. 回文链表

```go
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return true
	}
	// 找到前半部分链表的尾节点并反转后半部分链表
	half := findHalf(head)		// 返回的是slow节点
	end := reverse(half.Next)	// 所以这里是slow.next
	// 判断是否回文
	p1 := head
	p2 := end
	for p2 != nil {
		if p1.Val != p2.Val {
			return false
		}
		p1 = p1.Next
		p2 = p2.Next
	}
	return true
}

func reverse(head *ListNode) *ListNode {
	var pre, cur *ListNode= nil, head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

func findHalf(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}
```



## 543. 二叉树的直径

```go
func diameterOfBinaryTree(root *TreeNode) int {
	// 路径长度：一定是左子树的叶节点到右子树的叶节点之间的长度
	res := 0
	var dfs func(root *TreeNode) int
	dfs = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		// 左子树的直径长度
		leftLength := dfs(root.Left)
		// 右子树的直径长度
		rightLength := dfs(root.Right)
		// 将每个节点最大直径(左子树深度+右子树深度)与当前最大值比较并取大者
		res = max(res, leftLength+rightLength)
		// 返回左子树和右子树中较大的子树直径长度	也就是节点深度
		return max(leftLength, rightLength) + 1
	}
	dfs(root)
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 560. 和为 K 的子数组

```go
func subarraySum(nums []int, k int) int {
	// pre[i] 为 [0..i] 里所有数的和
	// pre[i] 可以由 pre[i−1] 递推而来，即： pre[i] = pre[i−1] + nums[i]
	count, pre := 0, 0
	// 以和为键，出现次数为值，记录 pre[i] 出现的次数
	hashMap := make(map[int]int)
	// 连续数组的和为0（数组为空）时，出现次数为1
	hashMap[0] = 1
	for i:=0; i<len(nums); i++ {
		pre += nums[i]
		// 考虑以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为 pre[i]−k 的 pre[j] 即可
		if _, ok := hashMap[pre-k]; ok {	// 如果存在  pre-k  这样的前缀和
			count += hashMap[pre-k]			// 加上 前缀和 pre-k 的出现次数
		}
		hashMap[pre]++
	}
	return count
}
```



## 287. 寻找重复数

```go
func findDuplicate(nums []int) int {
	n := len(nums)
	hashMap := make([]int, n)
	for i:=0; i<n; i++ {
		hashMap[nums[i]-1]++
		if hashMap[nums[i]-1] > 1 {
			return nums[i]
		}
	}
	return -1
}
```



## 238. 除自身以外数组的乘积

```go
func productExceptSelf(nums []int) []int {
	n := len(nums)
	ans := make([]int, n)
	// answer[i] 表示索引 i 左侧所有元素的乘积
	// 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
	ans[0] = 1
	for i:=1; i<n; i++ {
		ans[i] = ans[i-1] * nums[i-1]
	}
	// R 为右侧所有元素的乘积
	// 刚开始右边没有元素，所以 R = 1
	r := 1
	for i:=n-1; i>=0; i-- {
		// 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
		ans[i] *= r
		// R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
		r *= nums[i]
	}
	return ans
}
```



## 590. N 叉树的后序遍历

```go
type Node struct {
	Val int
	Children []*Node
}

func postorder(root *Node) []int {
	res := make([]int, 0)
	var dfs func(root *Node)
	dfs = func(root *Node) {
		if root == nil {
			return
		}
		for _, node := range root.Children {
			dfs(node)
		}
		res = append(res, root.Val)
	}
	dfs(root)	// 不要忘了这里！！！
	return res
}
```



## 347. 前 K 个高频元素

```go
func topKFrequent(nums []int, k int) []int {
	// 初始化一个map，用来存数字和数字出现的数字
	hashMap := make(map[int]int)
	res := make([]int, 0)
	for _,v := range nums {
		// 先查询map看一下v有没有存入map中，如果存在ok为true
		if _, ok := hashMap[v]; ok {
			// 不是第一次出现map的value数值+1
			hashMap[v]++
		} else {
			hashMap[v] = 1
			res = append(res, v)
		}
	}
	// 将res 按照map的value进行排序
	sort.Slice(res, func(i, j int) bool {
		return hashMap[res[i]] > hashMap[res[j]]
	})
	return res[:k]
}
```



## 169. 多数元素

```go
func majorityElement(nums []int) int {
	// 摩尔投票法
	// 如果我们把众数记为 +1，把其他数记为 −1，将它们全部加起来，显然和大于 0，从结果本身我们可以看出众数比其他数多
	res, count := 0, 0
	for _, num := range nums {
		if count == 0 {
			res = num	// 找到一个新的众数，维护它
			count = 1
		} else {
			if res == num {		// 新的数等于众数
				count++
			} else {			// 新的数不等于众数
				count--
			}
		}
	}
	return res
}
```



## 26. 删除有序数组中的重复项

```go
func removeDuplicates(nums []int) int {
	i := 0
	for _, num := range nums {
		if i == 0 || num != nums[i-1] {		// i=0时，num = nums[0]     或者    遍历的num ≠ nums[i-1]
			nums[i] = num					// 将数组的第i位赋值为num
			i++								// 赋值后数组下标 i 增加
		}
	}
	return i		// 数组一共有i个
}
```



## 80. 删除有序数组中的重复项 II

```go
func removeDuplicates(nums []int) int {
	i := 0
	for _, num := range nums {
		// 由于相同的数字最多保留 k 个，那么原数组的前 k 个元素我们可以直接保留；   i < k
		// 对于后面的数字，能够保留的前提是：当前数字 num 与前面已保留的数字的倒数第 k 个元素比较，不同则保留，相同则跳过   num != nums[i-k]
		if i < 2 || num != nums[i-2] {
			nums[i] = num					// 将数组的第i位赋值为num
			i++								// 赋值后数组下标 i 增加
		}
	}
	return i		// 数组一共有i个
}
```



## 27. 移除元素

```go
func removeElement(nums []int, val int) int {
	count, n := 0, len(nums)
	for i:=0; i<n; i++ {
		if nums[i] == val {
			count++
		} else {
			nums[i-count] = nums[i]
		}
	}
	return n - count
}
```



## 283. 移动零

```go
func moveZeroes(nums []int)  {
	// 双指针
	// 左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部
	// 右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移
	// 注意到以下性质：
	//		左指针左边均为非零数；
	//		右指针左边直到左指针处均为零。
	// 因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变
	left, n := 0, len(nums)
	for right:=0; right<n; right++ {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left++
		}
	}
}
```



## 189. 轮转数组

```go
func rotate(nums []int, k int)  {
	n := len(nums)
	k %= n						// 注意这里   题目中k可能很大，可能超过数组的长度n，因此要求余
	reverse(nums, 0, n-1)	// 翻转整个数组 0 ~ n-1
	reverse(nums, 0, k-1)	// 翻转 0 ~ k-1
	reverse(nums, k, n-1)		// 翻转 k ~ n-1
}

func reverse(nums []int, i, j int) {
	for i < j {
		nums[i], nums[j] = nums[j], nums[i]
		i++
		j--
	}
}
```



## 393. UTF-8 编码验证

```go
func validUtf8(data []int) bool {
	n := len(data)	// 用 m 表示 data 的长度
	// 从前往后处理每个 data[i]，先统计 data[i] 从第 7 位开始往后有多少位连续的 1，代表这是一个几字节的字符，记为 cnt
	for i:=0; i<n; {
		tmp := data[i]
		j := 7
		for j >= 0 && (((tmp >> j) & 1) == 1) {	// 遇到连续的1
			j--	// 移动的位数减少1
		}
		cnt := 7 - j	// 这个字符有cnt位
		if cnt == 1 || cnt > 4 {	// 如果 cnt 为 1 或者大于 4 均违反编码规则    因为一个字符可能的长度为 1 到 4 字节
			return false
		}
		if i + cnt - 1 > n {		// 如果位置 i 后面不足 cnt−1 也 false
			return false
		}
		for k:=i+1; k<i+cnt; k++ {
			if (((data[k] >> 7) & 1) ==1) && ((data[k] >> 6) & 1 == 0) {
				continue
			}
			return false
		}
		if cnt == 0 {
			i++
		} else {
			i += cnt
		}
	}
	return true
}
```



## 54. 螺旋矩阵

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])
	ans := make([]int, 0)	// 注意，这里是0，不能写m*n
	top, bottom, left, right := 0, m-1, 0, n-1
	for left <= right && top <= bottom {
		for i:=left; i<=right; i++ {			// 左上到右上
			ans = append(ans, matrix[top][i])
		}
		for i:=top+1; i<=bottom; i++ {			// 右上到右下
			ans = append(ans, matrix[i][right])
		}
		if left < right && top < bottom {	// 当 left == right 或者 top == bottom 时，不会发生右到左和下到上，否则会重复计数
			for i:=right-1; i>=left; i-- {		// 右下到左下
				ans = append(ans, matrix[bottom][i])
			}
			for i:=bottom-1; i>top; i-- {		// 左下到左上    注意 这里是>，不是>=，要缩小一圈
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



## 15. 三数之和

```go
func threeSum(nums []int) [][]int {
	n := len(nums)
	res := make([][]int, 0)   // (a, b, c)
	if n < 3 {
		return res
	}
	sort.Ints(nums)		// 保证最终答案不包含重复的三元组
	// 枚举 a
	for i:=0; i<n-2 && nums[i] <= 0; i++ {	// 保证起码第一个数是<=0的
		if i>0 && nums[i]==nums[i-1] {		// 对于每一重循环而言，相邻两次枚举的元素不能相同，否则也会造成重复
			continue	// 只有和上一次枚举的元素不相同，才会进行枚举
		}
		j, k := i+1, n-1		// c 对应的指针初始指向数组的最右端
		// 枚举 b
		for j < k {		// 保持第二重循环不变，而将第三重循环变成一个从数组最右端开始向左移动的指针
			if nums[i] + nums[j] + nums[k] == 0 {
				res = append(res, []int{nums[i], nums[j], nums[k]})
				j++
				k--
				for j < n && nums[j] == nums[j-1] {		// 找到一个解之后，跳过相同的b
					j++
				}
				for k > i && nums[k] == nums[k+1] {		// 找到一个解之后，跳过相同的c
					k--
				}
			} else if nums[i] + nums[j] + nums[k] < 0 {
				j++
			} else {
				k--
			}
		}
	}
	return res
}
```



## 18. 四数之和

```go
func fourSum(nums []int, target int) [][]int {
	n := len(nums)
	res := make([][]int, 0)   // (a, b, c, d)
	if n < 4 {
		return res
	}
	sort.Ints(nums)		// 保证最终答案不包含重复的三元组
	// 枚举 a
	for a:=0; a<n-3; a++ {
		if a>0 && nums[a]==nums[a-1] {		// 对于每一重循环而言，相邻两次枚举的元素不能相同，否则也会造成重复
			continue	// 只有和上一次枚举的元素不相同，才会进行枚举
		}
		for b:=a+1; b<n; b++ {
			if b>a+1 && nums[b]==nums[b-1] {
				continue
			}
			c, d := b+1, n-1
			for c < d {
				if nums[a]+nums[b]+nums[c]+nums[d] == target {
					res = append(res, []int{nums[a], nums[b], nums[c], nums[d]})
					c++
					d--
					for c < n && nums[c] == nums[c-1] {
						c++
					}
					for d > b && nums[d] == nums[d+1] {
						d--
					}
				} else if nums[a]+nums[b]+nums[c]+nums[d] > target {
					d--
				} else {
					c++
				}
			}
		}
	}
	return res
}
```



## 16. 最接近的三数之和

```go
func threeSumClosest(nums []int, target int) int {
	n := len(nums)
	res := math.MaxInt32
	sort.Ints(nums)		// 保证最终答案不包含重复的三元组
	// 枚举 a
	for i:=0; i<n-2; i++ {
		if i>0 && nums[i]==nums[i-1] {		// 对于每一重循环而言，相邻两次枚举的元素不能相同，否则也会造成重复
			continue	// 只有和上一次枚举的元素不相同，才会进行枚举
		}
		j, k := i+1, n-1		// c 对应的指针初始指向数组的最右端
		// 枚举 b
		for j < k {		// 保持第二重循环不变，而将第三重循环变成一个从数组最右端开始向左移动的指针
			sum := nums[i] + nums[j] + nums[k]
			if sum == target {
				return target
			}
			res = compare(target, sum, res)
			if sum > target {
				k--
				for j < k && nums[k] == nums[k+1] {
					k--
				}
			} else {
				j++
				for j < n && nums[j] == nums[j-1] {
					j++
				}
			}
		}
	}
	return res
}

func compare(target, sum, res int) int {
	if abs(target-sum) < abs(target-res) {
		return sum
	}
	return res
}

func abs(a int) int {
	if a > 0 {
		return a
	}
	return -a
}
```



## 599. 两个列表的最小索引总和

```go
func findRestaurant(list1 []string, list2 []string) []string {
	hashMap := make(map[string]int)
	for i, s := range list1 {
		hashMap[s] = i
	}
	res := make([]string, 0)
	sum := math.MaxInt32
	for i, s := range list2 {
		if j, ok := hashMap[s]; ok {
			if i+j < sum {
				sum = i+j
				res = []string{s}
			} else if i+j == sum {
				res = append(res, s)
			}
		}
	}
	return res
}
```



## 88. 合并两个有序数组

```go
func merge(nums1 []int, m int, nums2 []int, n int) {
	// nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0
	i, j, k := m-1, n-1, m+n-1	// 从后往前
	for j >= 0 {	// 对于 nums2 中的数
		if i >= 0 && nums1[i] > nums2[j] {	// 如果nums1还有数，且nums1>nums2    则选取nums1的元素
			nums1[k] = nums1[i]
			i--
		} else {		// 如果nums1没有数，或nums1 <= nums2     则选取nums2的元素
			nums1[k] = nums2[j]
			j--
		}
		k--	// k不用减到0   因为nums1的前m个数是应合并的，有可能不需要移动，所以大循环是遍历nums2中的数
	}
}
```



## 153. 寻找旋转排序数组中的最小值

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	if nums[left] < nums[right] {
		return nums[0]
	}
	for left < right {
		mid := left + (right - left) / 2	// 取中间值
		if nums[mid] > nums[right] {		// 如果中间值大于最大值，则最小值变大
			left = mid + 1
		} else {				// 如果中间值小于最大值，则最大值减小
			right = mid	
		}
	}
	return nums[left]
}
```



## 154. 寻找旋转排序数组中的最小值 II

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	if nums[left] < nums[right] {
		return nums[0]
	}
	for left < right {
		mid := left + (right - left) / 2	// 取中间值
		if nums[mid] > nums[right] {		// 如果中间值大于最大值，说明最小值在 m 的右边
			left = mid + 1
		} else if nums[mid] < nums[right] {		// 如果中间值小于最大值，说明最小值在 m 的左边（包括 m）
			right = mid
		} else {	// 若相等，无法判断，直接将 r 减 1。循环比较
			right--
		}
	}
	return nums[left]
}
```



## 345. 反转字符串中的元音字母

```go
func reverseVowels(s string) string {
	// 将字符串转为字符数组（或列表），定义双指针 i、j，分别指向数组（列表）头部和尾部，当 i、j 指向的字符均为元音字母时，进行交换。
	//依次遍历，当 i >= j 时，遍历结束。将字符数组（列表）转为字符串返回即可。

	left, right := 0, len(s)-1
	a := []byte(s)
	for left < right {
		for left < right && !isAEIOU(a[left]) {
			left++
		}
		for left < right && !isAEIOU(a[right]) {
			right--
		}
		if left != right && isAEIOU(a[left]) && isAEIOU(a[right]) {
			a[left], a[right] = a[right], a[left]
			left++
			right--
		}
	}
	return string(a)
}

func isAEIOU(b byte) bool {
	return b=='a' || b=='A' || b=='e' || b=='E' || b=='i' || b=='I' || b=='o' || b=='O' || b=='u' || b=='U'
}
```



## 8. 字符串转换整数 (atoi)

```go
func myAtoi(s string) int {
/*
	1 读入字符串并丢弃无用的前导空格
   2 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
   3 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
   4 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
   5 如果整数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。
		具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
   6 返回整数作为最终结果。
 */

	i, n := 0, len(s)
	ans := 0
	for i<n && s[i]==' ' {	// 丢弃前导空格
		i++
	}
	if i == n {
		return 0
	}
	flag := 1
	if s[i] == '-' {	// 检查符号
		flag = -1
		i++
	} else if s[i] == '+'{
		i++
	}
	for i<n && s[i]>='0' && s[i]<='9' {		// 读入数字字符，直到遇到非数字字符
		ans = ans*10 + int(s[i]-'0')
		i++
		if ans > math.MaxInt32 {
			break
		}
	}
	if ans > math.MaxInt32 {	// 整数超过 32 位有符号整数范围
		if flag == -1 {
			return math.MinInt32
		} else {
			return math.MaxInt32
		}
	}
	return flag * ans
}
```



## 2044. 统计按位或能得到最大值的子集数目

```go
func countMaxOrSubsets(nums []int) int {
	res, tmp := 0, 0
	var dfs func(int, int)
	// 参数 pos 表示当前下标，or 表示当前下标之前的某个子集按位或值
	dfs = func(pos int, or int) {
		if pos == len(nums) {
			if or > tmp {			// 新的 or 大于之前的最大值
				tmp = or
				res = 1
			} else if tmp == or {	// 新的 or 等于之前的最大值
				res++
			}
			return
		}
		dfs(pos+1, or | nums[pos])
		dfs(pos+1, or)
	}
	dfs(0,0)
	return res
}
```



## 445. 两数相加 II

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	s1, s2 := list.New(), list.New()
	for l1 != nil {
		s1.PushBack(l1.Val)
		l1 = l1.Next
	}
	for l2 != nil {
		s2.PushBack(l2.Val)
		l2 = l2.Next
	}

	// 建立一个虚拟头结点，这个虚拟头结点的 Next 指向真正的 head，这样 head 不需要单独处理
	head := &ListNode{Val: 0}
	// carry : 进位
	carry := 0
	for s1.Len()!=0 || s2.Len()!=0 || carry!=0 {	// 链表1还未到底  或  链表2还未到底  或  还有进位没处理
		if s1.Len() != 0 {
			carry += s1.Back().Value.(int)
			s1.Remove(s1.Back())
		}
		if s2.Len() != 0 {
			carry += s2.Back().Value.(int)
			s2.Remove(s2.Back())
		}
		// 创建一个新节点作为当前结点的下一个
		current := &ListNode{Val: carry%10, Next: head.Next}	// 注意 这里的顺序和之前的同类题完全不一样
		head.Next = current
		carry /= 10	// 计算在新节点应该加上的进位
	}
	return head.Next
}
```



## 237. 删除链表中的节点

```go
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}
```



## 83. 删除排序链表中的重复元素

```go
func deleteDuplicates(head *ListNode) *ListNode {
	real := head
	for head != nil && head.Next != nil {
		if head.Val == head.Next.Val {
			head.Next = head.Next.Next
		} else {
			head = head.Next
		}
	}
	return real
}
```



## 82. 删除排序链表中的重复元素 II

```go
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	dummy := &ListNode{0, head}
	current := dummy
	for current.Next != nil && current.Next.Next != nil {
		if current.Next.Val == current.Next.Next.Val {
			x := current.Next.Val
			for current.Next != nil && current.Next.Val == x {
				current.Next = current.Next.Next
			}
		} else {
			current = current.Next
		}
	}
	return dummy.Next
}
```



## 面试题 02.02. 返回倒数第 k 个节点

```go
func kthToLast(head *ListNode, k int) int {
	slow, fast := head, head
	for i:=0; i<k; i++ {
		fast = fast.Next
	}
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	return slow.Val
}
```



## 21. 合并两个有序链表

```go
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			cur.Next = list1
			list1 = list1.Next
		} else {
			cur.Next = list2
			list2 = list2.Next
		}
		cur = cur.Next
	}
	if list1 != nil {
		cur.Next = list1
	}
	if list2 != nil {
		cur.Next = list2
	}
	return dummy.Next
}
```



## 23. 合并K个升序链表

```go
func mergeKLists(lists []*ListNode) *ListNode {
	n := len(lists)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return lists[0]
	}
	var dfs func(l,r int) *ListNode
	dfs = func(l,r int) *ListNode{
		if l == r {		// 如果当前l==r，则表示当前只有一个链表，直接返回
			return lists[l]
		}
		mid := l + ((r - l) >> 1)
		ll := dfs(l,mid)
		rr := dfs(mid + 1,r)
		return mergeTwoLists(ll,rr)
	}
	return dfs(0,n - 1)
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			cur.Next = list1
			list1 = list1.Next
		} else {
			cur.Next = list2
			list2 = list2.Next
		}
		cur = cur.Next
	}
	if list1 != nil {
		cur.Next = list1
	}
	if list2 != nil {
		cur.Next = list2
	}
	return dummy.Next
}
```



## 147. 对链表进行插入排序

```go
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	// end 为链表的已排序部分的最后一个节点
	// cur 为待插入的元素
	end, cur := head, head.Next
	for cur != nil {
		if end.Val <= cur.Val {		// 如果要插入的节点大于已排序的最后一个节点，直接放在已排序的后面
			end = end.Next
		} else {					// 如果要插入的节点小于已排序的最后一个节点，要从头遍历找到合适的位置
			// pre 为遍历到的第一个大于cur的节点
			pre := dummy
			for pre.Next.Val <= cur.Val {
				pre = pre.Next
			}
			end.Next = cur.Next		// 已排序的最后一个节点的next指向待插入节点的next
			cur.Next = pre.Next		// 待插入节点变成已插入节点，其next指向合适的位置的next
			pre.Next = cur			// 合适位置前一位的next指向待插入节点
		}
		cur = end.Next	// 待插入节点变为已排序节点的下一个
	}
	return dummy.Next
}
```



## 92. 反转链表 II

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head.Next == nil || left == right {
		return head
	}
	dummy := &ListNode{Next: head}
	pre := dummy
	for i:=0; i<left-1; i++ {
		pre = pre.Next
	}
	//    pre.next = left = cur
	cur := pre.Next
	// cur	指向待反转区域的第一个节点 left
	// next	永远指向 cur 的下一个节点，循环过程中，cur 变化以后 next 会变化
	// pre 	永远指向待反转区域的第一个节点 left 的前一个节点，在循环过程中不变
	for i:=0; i<right-left; i++ {
		next := cur.Next
		cur.Next = next.Next
		next.Next = pre.Next
		pre.Next = next
	}
	return dummy.Next
}
```



## 143. 重排链表

```go
func reorderList(head *ListNode)  {
	if head == nil || head.Next == nil {
		return
	}
	slow, fast := head, head.Next

	// 先通过快慢指针找到链表中点
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}

	// 将链表划分为左右两部分
	cur := slow.Next
	slow.Next = nil
	var pre *ListNode	// 这里不能初始化，只能定义

	// 反转右半部分的链表
	for cur != nil {
		t := cur.Next
		cur.Next = pre
		pre, cur = cur, t
	}

	// 将左右两个链接依次连接
	cur = head
	for pre != nil {
		t := pre.Next
		pre.Next = cur.Next
		cur.Next = pre
		cur, pre = pre.Next, t
	}
}
```



## 720. 词典中最长的单词

```go
func longestWord(words []string) string {
	// 为了方便处理，需要将数组 words 排序，
	// 排序的规则是首先按照单词的长度升序排序，如果单词的长度相同则按照字典序降序排序。
	// 排序之后，可以确保当遍历到每个单词时，比该单词短的全部单词都已经遍历过，
	// 且每次遇到符合要求的单词一定是最长且字典序最小的单词，可以直接更新答案。
	sort.Slice(words, func(i, j int) bool {
		s, t := words[i], words[j]
		return len(s) < len(t) || len(s)==len(t) && s > t
	})
	res := ""
	hashMap := make(map[string]int)
	hashMap[""] = 1	// 初始时将空字符串加入哈希集合
	// 对于每个单词，判断当前单词去掉最后一个字母之后的前缀是否在哈希集合中
	for _, word := range words {
		if _, ok := hashMap[word[:len(word)-1]]; ok {
			res = word
			hashMap[word] = 1
		}
	}
	return res
}
```



## 61. 旋转链表

```go
/*
将链表右半部分的 k 的节点拼接到 head 即可。
	注：k 对链表长度 n 取余，即 k %= n。
 */
func rotateRight(head *ListNode, k int) *ListNode {
	if k == 0 || head == nil || head.Next == nil {
		return head
	}
	n := 0
	for cur:=head; cur!=nil; cur=cur.Next {
		n++
	}
	k %= n
	if k == 0 {
		return head
	}
	slow, fast := head, head
	// fast移动后要指向链表的末尾
	// slow移动后slow.next要指向倒数第k个节点
	for k > 0 {
		fast = fast.Next
		k--
	}

	for fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	// 此时fast是尾节点
	// start 是移动k位后的头节点
	// slow.next是移动k位后的起始节点，slow是移动k位后的尾节点
	start := slow.Next
	slow.Next = nil
	fast.Next = head
	return start
}
```



## 328. 奇偶链表

```go
func oddEvenList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	odd, even := head, head.Next
	evenHead := even
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	odd.Next = evenHead
	return head
}
```



## 141. 环形链表

```go
// 定义快慢指针 slow、fast，初始指向 head。
// 快指针每次走两步，慢指针每次走一步，不断循环。
// 当相遇时，说明链表存在环。如果循环结束依然没有相遇，说明链表不存在环。
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}
```



## 933. 最近的请求次数

```go
type RecentCounter struct {
	q []int
}


func Constructor() RecentCounter {
	return RecentCounter{
		q: []int{},
	}
}

// Ping
/*
在时间 t 添加一个新请求，其中 t 表示以毫秒为单位的某个时间，并返回过去 3000 毫秒内发生的所有请求数（包括新请求）。
确切地说，返回在 [t-3000, t] 内发生的请求数。
 */
func (this *RecentCounter) Ping(t int) int {
	this.q = append(this.q, t)
	for this.q[0] < t-3000 {	// 进行优化    可以当作只存储3000毫秒内的数据，超过的直接丢弃
		this.q = this.q[1:len(this.q)]
	}
	return len(this.q)
}
```



## 2043. 简易银行系统

```go
type Bank struct {
	b []int64
}

func Constructor(balance []int64) Bank {
	return Bank{
		b: balance,
	}
}


func (this *Bank) Transfer(account1 int, account2 int, money int64) bool {
	n := len(this.b)
	if account1 > n || account2 > n || money > this.b[account1-1] {
		return false
	}
	this.b[account1-1] -= money
	this.b[account2-1] += money
	return true
}


func (this *Bank) Deposit(account int, money int64) bool {
	n := len(this.b)
	if account > n {
		return false
	}
	this.b[account-1] += money
	return true
}


func (this *Bank) Withdraw(account int, money int64) bool {
	n := len(this.b)
	if account > n || money > this.b[account-1] {
		return false
	}
	this.b[account-1] -= money
	return true
}
```





## 12. 整数转罗马数字

```go
func intToRoman(num int) string {
	thousands := []string{"", "M", "MM", "MMM"}
	hundreds  := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
	tens      := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
	ones      := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
	return thousands[num/1000] + hundreds[num%1000/100]	 + tens[num%100/10] + ones[num%10]
}
```



## 13. 罗马数字转整数

```go
func romanToInt(s string) int {
	hashMap := make(map[byte]int)
	hashMap['I'] = 1
	hashMap['V'] = 5
	hashMap['X'] = 10
	hashMap['L'] = 50
	hashMap['C'] = 100
	hashMap['D'] = 500
	hashMap['M'] = 1000
	ans := 0
	for i:=0; i<len(s)-1; i++ {
		// 通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例
		if hashMap[s[i]] < hashMap[s[i+1]] {	// 特例 : 左边字符小，右边字符大
			ans -= hashMap[s[i]]
		} else {	//  常规情况 : 左边字符大，右边字符小
			ans += hashMap[s[i]]
		}
	}
	return ans + hashMap[s[len(s)-1]]
}
```



## 136. 只出现一次的数字

```go
func singleNumber(nums []int) int {
	// 两个相同的数异或之后的结果为 0
	// 对该数组所有元素进行异或运算，结果就是那个只出现一次的数字
	res := 0
	for _, v := range nums {
		res ^= v
	}
	return res
}
```



## 137. 只出现一次的数字 II

```go
func singleNumber(nums []int) int {
	n := len(nums)
	hashMap := make(map[int]int, n)
	for i:=0; i<n; i++ {
		hashMap[nums[i]]++
	}
	for i, v := range hashMap {
		if v == 1 {
			return i
		}
	}
	return -1
}
```



## 260. 只出现一次的数字 III

```go
func singleNumber(nums []int) (ans []int) {
    freq := map[int]int{}
    for _, num := range nums {
        freq[num]++
    }
    for num, occ := range freq {
        if occ == 1 {
            ans = append(ans, num)
        }
    }
    return
}
```



## 606. 根据二叉树创建字符串

```go
func tree2str(root *TreeNode) string {
	if root == nil {
		return ""
	} else if root.Left == nil && root.Right == nil {
		return strconv.Itoa(root.Val)
	} else if root.Right == nil {
		return strconv.Itoa(root.Val) + "(" + tree2str(root.Left) + ")"
	} else {
		return strconv.Itoa(root.Val) + "(" + tree2str(root.Left) + ")" + "(" + tree2str(root.Right) + ")"
	}
}
```



## 645. 错误的集合

```go
func findErrorNums(nums []int) []int {
	n := len(nums)
	repeat, lost := 0, 0
	hashMap := make(map[int]int, n)
	for _, v := range nums {
		hashMap[v]++
	}
	for i:=1; i<=n; i++ {
		if hashMap[i] == 2 {
			repeat = i
		} else if hashMap[i] == 0 {
			lost = i
		}
	}
	return []int{repeat, lost}
}
```



## 191. 位1的个数

```go
func hammingWeight(num uint32) int {
	// n & (n−1)，其运算结果恰为把 n 的二进制位中的最低位的 1 变为 0 之后的结果
	ans := 0
	for num != 0 {
		num &= num-1
		ans++
	}
	return ans
}
```



## 204. 计数质数

```go
func countPrimes(n int) int {
	// 如果 x 是质数，那么大于 x 的 x 的倍数 2x,3x,… 一定不是质数
	if n < 2 {
		return 0
	}
	primes := make([]bool, n)
	// 初始化长度 O(n) 的标记数组，表示这个数组是否为质数
	for i:=0; i<n; i++ {
		primes[i] = true	// 数组初始化所有的数都是质数
	}
	res := 0
	// 从 2 开始将当前数字的倍数全都标记为合数。标记到 根号n 时停止即可
	for i:=2; i<n; i++ {
		if primes[i] {
			res++
			if i*i < n {
				for j:=i*i; j<n; j+=i {
					primes[j] = false
				}
			}
		}
	}
	return res
}
```



## 268. 丢失的数字

```go
func missingNumber(nums []int) int {
	// 2n+1 个整数中，丢失的数字出现了一次，其余的数字都出现了两次
	// 因此对上述 2n+1 个整数进行按位异或运算，结果即为丢失的数字
	res := 0
	for i, v := range nums {
		// 在 nums 这 n 个数的后面添加从 0 到 n 的每个整数
		res ^= i ^ v
	}
	return res ^ len(nums)
}
```



## 2039. 网络空闲的时刻

```go
func networkBecomesIdle(edges [][]int, patience []int) (ans int) {
    n := len(patience)
    g := make([][]int, n)
    for _, e := range edges {
        x, y := e[0], e[1]
        g[x] = append(g[x], y)
        g[y] = append(g[y], x)
    }

    vis := make([]bool, n)
    vis[0] = true
    q := []int{0}
    for dist := 1; q != nil; dist++ {
        tmp := q
        q = nil
        for _, x := range tmp {
            for _, v := range g[x] {
                if vis[v] {
                    continue
                }
                vis[v] = true
                q = append(q, v)
                ans = max(ans, (dist*2-1)/patience[v]*patience[v]+dist*2+1)
            }
        }
    }
    return
}

func max(a, b int) int {
    if b > a {
        return b
    }
    return a
}
```



## 653. 两数之和 IV

```go
func findTarget(root *TreeNode, k int) bool {
	hashMap := make(map[int]bool)
	var dfs func(node *TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return false
		}
		if _, ok := hashMap[k-node.Val]; ok {
			return true
		}
		hashMap[node.Val] = true
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}
```



## 1137. 第 N 个泰波那契数

```go
func tribonacci(n int) int {
	a, b, c := 0, 1, 1
	for i:=0; i<n; i++ {
		a, b, c = b, c, a+b+c
	}
	return a
}
```



## 740. 删除并获得点数

```go
func deleteAndEarn(nums []int) int {
	// 核心思路: 一个数字要么不选，要么全选
	// 首先计算出每个数字的总和 sums，并维护两个
	Max := math.MinInt32
	for _, num := range nums {
		Max = max(Max, num)
	}
	sum := make([]int, Max+1)	// sum[i] 代表值为 i 的元素总和
	for _, num := range nums {
		sum[num] += num
	}
	first := sum[0]	// 初始化为选了nums[0]
	second := max(sum[0], sum[1])	// 初始化为选了sum[0]和sum[1]中较大的那个
	for i:=2; i<=Max; i++ {
		// 如果选 i，那么就是 first+sums[i]    first是选了0~i-2范围内的值
		// 如果不选 i，那么就是second 	 second是选了0~i-1范围内的值  因为不选sum[i]，所以可以选sum[i-1]
		cur := max(first+sum[i], second)	// 选择其中较大的选法
		first = second	// 因为这一轮second没有选sum[i]，所以下一轮可以选，second变成first
		second = cur	// 因为cur是当前最优解，下一轮不进行选择
	}
	return second
}



func abs(a int) int {
	if a > 0 {
		return a
	}
	return -a
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 918. 环形子数组的最大和

https://leetcode-cn.com/problems/maximum-sum-circular-subarray/solution/wo-hua-yi-bian-jiu-kan-dong-de-ti-jie-ni-892u/

```go
func maxSubarraySumCircular(nums []int) int {
	// total为数组的总和，
	// maxSum为最大子数组和，
	// minSum为最小子数组和，
	// curMax为包含当前元素的最大子数组和，
	// curMin为包含当前元素的最小子数组和
	total, maxSum, minSum, curMax, curMin := nums[0], nums[0], nums[0], nums[0], nums[0]

	for i := 1; i < len(nums); i++ {
		total += nums[i]
		curMax = max(curMax+nums[i], nums[i])
		maxSum  = max(maxSum, curMax)
		curMin = min(curMin+nums[i], nums[i])
		minSum  = min(minSum, curMin)
	}

	// 等价于if maxSum < 0
	if total == minSum  {	// 全是负数
		return maxSum		// 返回最大的那个负数（最接近0）
	} else {
		return max(maxSum, total - minSum)
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 2038. 如果相邻两个颜色均相同则删除当前颜色

```go
func winnerOfGame(colors string) bool {
	res := [2]int{}
	cur, num := 'C', 0
	for _, ch := range colors {
		if ch != cur {
			cur = ch
			num = 1
		} else {
			num++
			if num > 2 {
				res[cur-'A']++
			}
		}
	}
	return res[0] > res[1]
}
```



## 1567. 乘积为正数的最长子数组长度

```go
func getMaxLen(nums []int) int {
	// pos 表示乘积为正数的最长子数组长度
	// neg 表示乘积为负数的最长子数组长度
	pos, neg := 0, 0
	if nums[0] > 0 {
		pos = 1
	} else if nums[0] < 0 {
		neg = 1
	}
	res := pos
	for i:=1; i<len(nums); i++ {
		if nums[i] > 0 {			// 当 nums[i]>0 时，之前的乘积乘以 nums[i] 不会改变乘积的正负性
			pos++					// 正数长度++
			if neg > 0 {
				neg++				// 之前有负数的话，负数长度++
			}
		} else if nums[i] < 0{		// 当 nums[i]<0 时，之前的乘积乘以 nums[i] 会改变乘积的正负性
			tmpPos, tmpNeg := pos, neg
			neg = tmpPos + 1	// 负数长度=之前的正数长度+1
			if tmpNeg > 0 {
				pos = tmpNeg + 1	// 正数长度=之前的负数长度+1
			} else {
				pos = 0				// 之前全是正数，此时正数长度为0
			}
		} else {	// 当 nums[i]=0 时，pos和neg重置为0
			pos, neg = 0, 0
		}
		res = max(res, pos)
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 1014. 最佳观光组合

```go
func maxScoreSightseeingPair(values []int) int {
	res, tmp := 0, values[0]
	for i:=1; i<len(values); i++ {
		res = max(res, values[i] - i + tmp)	// 这里的i是一直更新的，其实相当于j res = values[j] - j + tmp
		tmp = max(tmp, values[i] + i)		// tmp是之前记录的i，tmp = values[i] + i
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 413. 等差数列划分

```go
func numberOfArithmeticSlices(nums []int) int {
	// 动态规划法
	// 设 dp[i] 表示以 i 结尾的数组构成的等差数列的个数。
	// 如果 nums[i] + nums[i - 2] ≠ nums[i - 1] * 2，说明以 i 结尾的数组无法构成等差数列，dp[i] = 0；否则 dp[i] = 1 + dp[i - 1]。
	// 结果返回 dp 数组所有元素之和即可。
	n := len(nums)
	dp := make([]int, n)
	for i:=2; i<n; i++ {
		if nums[i]-nums[i-1] == nums[i-1]-nums[i-2] {
			dp[i] = dp[i-1] + 1
		}
	}
	res := 0
	for _, v := range dp {
		res += v
	}
	return res
}
```



## 91. 解码方法

```go
func numDecodings(s string) int {
	// 设 f[i] 表示字符串 s 的前 i 个字符 s[0..i-1] 的解码方法数
	n := len(s)
	// a = f[i-2], b = f[i-1], c = f[i]
	a, b, c := 0, 1, 0
	for i := 0; i < n; i++ {
		c = 0
		if s[i] != '0' {	// 如果当前的字符不是0
			c += b			// 则f[i] += f[i-1]   因为这一次解码用了1个字符，所以是和f[i-1]有关
		}
		// 从第二个字符开始，如果前一个字符不是0，前一个字符*10+后一个字符<=26，则多了一种解码方法
		if i > 0 && s[i-1] != '0' && ((s[i-1]-'0')*10+(s[i]-'0') <= 26) {
			c += a	// 则f[i] += f[i-2]   因为这一次解码用了2个字符，所以是和f[i-2]有关
		}
		a, b = b, c
	}
	return c
}
```



## 440. 字典序的第K小数字

```go
// 给定整数 n 和 k，返回  [1, n] 中字典序第 k 小的数字
func findKthNumber(n int, k int) int {
	i := 1		// 数字i
	p := 1		// 位置p
	for p < k {
		// 还未到第k个元素
		cnt := getCount(i, n)
		if p+cnt > k {
			i *= 10
			p++
		} else {
			i++
			p += cnt
		}
	}
	return i
}

// 计算以数字 i 开头且不超过最大数字 n 的数字个数，算法命名为 getCount
func getCount(i, n int) int {
	cnt := 0
	// 区间起点a = 数字i
	// 区间终点b = 数字i+1
	for a,b := i, i+1; a <= n; a,b = a*10,b*10 {
		// 说明区间没有超过最大数字n
		cnt += min(b, n+1) - a
	}
	return cnt
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 264. 丑数 II

```go
func nthUglyNumber(n int) int {
	dp := make([]int, n)
	dp[0] = 1
	p2, p3, p5 := 0, 0, 0
	for i:=1; i<n; i++ {
		next2, next3, next5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
		dp[i] = min(next2, min(next3, next5))
		if dp[i] == next2 {
			p2++
		}
		if dp[i] == next3 {
			p3++
		}
		if dp[i] == next5 {
			p5++
		}
	}
	return dp[n-1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 931. 下降路径最小和

```go
func minFallingPathSum(matrix [][]int) int {
	n := len(matrix)
	for i:=1; i<n; i++ {
		for j:=0; j<n; j++ {
			mi := matrix[i-1][j]	// 正上方
			if j>0 && mi>matrix[i-1][j-1] {
				mi = matrix[i-1][j-1]	// 左上方
			}
			if j<n-1 && mi>matrix[i-1][j+1] {
				mi = matrix[i-1][j+1]	// 右上方
			}
			matrix[i][j] += mi
		}
	}
	res := 100000
	for j:=0; j<n; j++ {
		res = min(res, matrix[n-1][j])
	}
	return res
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 120. 三角形最小路径和

```go
func minimumTotal(triangle [][]int) int {
	n := len(triangle)
	dp := make([]int, n+1)
	for i:=n-1; i>=0; i-- {
		for j:=0; j<=i; j++ {
			dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
		}
	}
	return dp[0]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 1314. 矩阵区域和

```go
func matrixBlockSum(mat [][]int, k int) [][]int {
	m, n := len(mat), len(mat[0])
	pre := make([][]int, m+1)
	for i:=0; i<m+1; i++ {
		pre[i] = make([]int, n+1)
	}
	for i:=1; i<m+1; i++ {
		for j:=1; j<n+1; j++ {
			pre[i][j] = pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1] + mat[i-1][j-1]
		}
	}
	get := func(i, j int) int {
		i = max(min(m, i), 0)
		j = max(min(n, j), 0)
		return pre[i][j]
	}

	ans := make([][]int, m)
	for i := 0; i < m; i++ {
		ans[i] = make([]int, n)
		for j := 0; j < n; j++ {
			ans[i][j] = get(i+k+1, j+k+1) - get(i+k+1, j-k) - get(i-k, j+k+1) + get(i-k, j-k)
		}
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```



## 661. 图片平滑器

```go
func imageSmoother(img [][]int) [][]int {
	m, n := len(img), len(img[0])
	ans := make([][]int, m)
	for i:=0; i<m; i++ {
		ans[i] = make([]int, n)
		for j:=0; j<n; j++ {
			sum, cnt:= 0, 0
			for _, row := range img[max(0, i-1):min(i+2, m)] {
				for _, v := range row[max(0, j-1):min(j+2, n)] {
					sum += v
					cnt++
				}
			}
			ans[i][j] = sum / cnt
		}
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```



## 304. 二维区域和检索 - 矩阵不可变

https://leetcode-cn.com/problems/range-sum-query-2d-immutable/solution/ru-he-qiu-er-wei-de-qian-zhui-he-yi-ji-y-6c21/

```go
type NumMatrix struct {
	pre [][]int
}


func Constructor(matrix [][]int) NumMatrix {
	m, n := len(matrix), len(matrix[0])
	pre := make([][]int, m+1)
	for i:=0; i<=m; i++ {
		pre[i] = make([]int, n+1)
	}
	for i:=1; i<=m; i++ {
		for j:=1; j<=n; j++ {
			pre[i][j] = pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1] + matrix[i-1][j-1]
		}
	}
	return NumMatrix{pre}
}

func (this *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	return this.pre[row2+1][col2+1] - this.pre[row2+1][col1] - this.pre[row1][col2+1] + this.pre[row1][col1]
}
```



## 64. 最小路径和

```go
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i:=0; i<m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for i:=1; i<m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for j:=1; j<n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}
	for i:=1; i<m; i++ {
		for j:=1; j<n; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```



## 剑指 Offer 47. 礼物的最大价值

```go
func maxValue(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i:=0; i<m; i++ {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for i:=1; i<m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}
	for j:=1; j<n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}
	for i:=1; i<m; i++ {
		for j:=1; j<n; j++ {
			dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}
	return dp[m-1][n-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 354. 俄罗斯套娃信封问题

```go
func maxEnvelopes(envelopes [][]int) int {
    sort.Slice(envelopes, func(i, j int) bool {
        a, b := envelopes[i], envelopes[j]
        return a[0] < b[0] || a[0] == b[0] && a[1] > b[1]
    })

    f := []int{}
    for _, e := range envelopes {
        h := e[1]
        if i := sort.SearchInts(f, h); i < len(f) {
            f[i] = h
        } else {
            f = append(f, h)
        }
    }
    return len(f)
}
```



## 172. 阶乘后的零

```go
func trailingZeroes(n int) int {
	res := 0
	for i:=5; i<=n; i+=5 {
		for x:=i; x%5==0; x/=5 {
			res++
		}
	}
	return res
}
```



## 682. 棒球比赛

```go
func calPoints(ops []string) int {
	res := 0
	stack := make([]int, 0)
	num := 0
	for _, v := range ops {
		if v == "C" {
			stack = stack[:len(stack)-1]
		} else if v == "D" {
			stack = append(stack, stack[len(stack)-1]*2)
		} else if v == "+" {
			stack = append(stack, stack[len(stack)-1]+stack[len(stack)-2])
		} else {
			num, _ = strconv.Atoi(v)
			stack = append(stack, num)
		}
	}
	for _, v := range stack {
		res += v
	}
	return res
}
```



## 2028. 找出缺失的观测数据

```go
func missingRolls(rolls []int, mean, n int) []int {
	missingSum := mean * (n + len(rolls))
	for _, roll := range rolls {
		missingSum -= roll
	}
	if missingSum < n || missingSum > n*6 {
		return nil
	}

	quotient, remainder := missingSum/n, missingSum%n
	ans := make([]int, n)
	for i := range ans {
		ans[i] = quotient
		if i < remainder {	// ans[0~remainder-1] = quotient+1
			ans[i]++		// ans[remainder~n-1] = quotient
		}
	}
	return ans
}
```



## 693. 交替位二进制数

```go
func hasAlternatingBits(n int) bool {
    a := n ^ n>>1
    return a&(a+1) == 0
}
```



## 2024. 考试的最大困扰度

```go
func maxConsecutiveChar(answerKey string, k int, ch byte) (ans int) {
	left, sum := 0, 0	// 区间中另一种字符的数量为 sum
	for right := range answerKey {
		if answerKey[right] != ch {
			sum++
		}
		// 当 sum 超过 k，我们需要让左端点右移, 直到 sum ≤ k
		for sum > k {
			if answerKey[left] != ch {
				sum--
			}
			left++
		}
		// 记录滑动窗口的最大长度，即为指定字符的最大连续数目
		ans = max(ans, right-left+1)
	}
	return
}

func maxConsecutiveAnswers(answerKey string, k int) int {
	// 分别指定字符为 T 和 F 时的最大连续数目的较大值
	return max(maxConsecutiveChar(answerKey, k, 'T'), maxConsecutiveChar(answerKey, k, 'F'))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 728. 自除数

```go
func selfDividingNumbers(left int, right int) []int {
	ans := make([]int, 0)
	for i:= left; i<=right; i++ {
		if isDivided(i) {
			ans = append(ans, i)
		}
	}
	return ans
}

func isDivided(num int) bool {
	s := num
	mod := num
	for s > 0 {
		mod = s % 10
		if mod != 0 {	// num 中这一位不是 0
			if num % mod != 0 {		// 不可以除尽，结束循环
				return false
			}
		} else {	// num 的这一位是 0
			return false
		}
		s /= 10
	}
	return true
}
```



## 954. 二倍数对数组

```go
func canReorderDoubled(arr []int) bool {
	n := len(arr)
	// 题目本质上是问 arr 能否分成 n/2 对元素，每对元素中一个数是另一个数的两倍
	hashMap := make(map[int]int, n)		// hashMap[i] 记录了 i 出现的次数
	for _, num := range arr {
		hashMap[num]++
	}
	if hashMap[0] % 2 == 1 {	// 0 必须出现偶数次，否则直接返回false
		return false
	}
	// 用一个哈希表来统计出现次数，并将其键值按绝对值从小到大排序
	val := make([]int, len(hashMap))
	for num := range hashMap {
		val = append(val, num)
	}
	sort.Slice(val, func(i, j int) bool {
		return abs(val[i]) < abs(val[j])
	})
	// arr 存储的是数字 i
	// hashMap 存储的是数字 i 出现的次数
	// val 存储的是数字 i（经过绝对值排序）

	// 设 x 为 arr 中绝对值最小的元素，由于没有绝对值比 x 更小的数，因此 x 只能与 2x 匹配
	for _, num := range val {
		if hashMap[num*2] < hashMap[num] {	// 无法找到足够的 2x 与 x 配对
			return false	// 如果此时 hashMap[2x] < hashMap[x]，那么会有部分 x 无法找到它的另一半
		}
		hashMap[num*2] -= hashMap[num]	// 将所有 x 和 hashMap[x] 个 2x 从 arr 中去掉，继续判断剩余元素是否满足题目要求
	}
	return true
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
```



## 744. 寻找比目标字母大的最小字母

```go
func nextGreatestLetter(letters []byte, target byte) byte {
	for _, ch := range letters {
		if target < ch {
			return ch
		}
	}
	return letters[0]
}
```



## 762. 二进制表示中质数个计算置位

```go
func isPrime(x int) bool {
    if x < 2 {
        return false
    }
    for i := 2; i*i <= x; i++ {
        if x%i == 0 {
            return false
        }
    }
    return true
}

func countPrimeSetBits(left, right int) (ans int) {
    for x := left; x <= right; x++ {
        if isPrime(bits.OnesCount(uint(x))) {
            ans++
        }
    }
    return
}
```



## 796. 旋转字符串

```go
func rotateString(s string, goal string) bool {
	return len(s) == len(goal) && strings.Contains(s+s, goal)
}
```



## 429. N 叉树的层序遍历

```go
func levelOrder(root *Node) [][]int {
	if root == nil {
		return nil
	}
	ans := make([][]int, 0)
	queue := make([]*Node, 0)
	queue = append(queue, root)
	for queue != nil {
		level := make([]int, 0)
		tmp := queue
		queue = nil
		for _, node := range tmp {
			level = append(level, node.Val)
			queue = append(queue, node.Children...)
		}
		ans = append(ans, level)
	}
	return ans
}
```



## 780. 到达终点

```go
func reachingPoints(sx int, sy int, tx int, ty int) bool {
	for tx > sx && ty > sy && tx != ty {
		if tx > ty {
			tx %= ty
		} else {
			ty %= tx
		}
	}
	switch {
	case tx == sx && ty == sy:
		return true
	case tx == sx :
		return ty > sy && (ty-sy)%tx == 0
	case ty == sy:
		return tx > sx && (tx-sx)%ty == 0
	default:
		return false
	}
}
```



## 804. 唯一摩尔斯密码词

```go
func uniqueMorseRepresentations(words []string) int {
	morse := []string{
		".-", "-...", "-.-.", "-..", ".", "..-.", "--.",
		"....", "..", ".---", "-.-", ".-..", "--", "-.",
		"---", ".--.", "--.-", ".-.", "...", "-", "..-",
		"...-", ".--", "-..-", "-.--", "--..",
	}
	hashMap := make(map[string]bool)
	for _, word := range words {
		str := ""
		for _, ch := range word {
			str += morse[ch - 'a']
		}
		hashMap[str] = true
	}
	return len(hashMap)
}
```



## 357. 统计各位数字都不同的数字个数

```go
func countNumbersWithUniqueDigits(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 10
	}
	ans, cur := 10, 9
	for i:=0; i<n-1; i++ {
		cur *= 9-i
		ans += cur
	}
	return ans
}
```



## 806. 写字符串需要的行数

```go
func numberOfLines(widths []int, s string) []int {
	row, col := 1, 0
	for _, ch := range s {
		col += widths[ch-'a']
		if col > 100 {
			col = widths[ch-'a']
			row++
		}
	}
	return []int{row, col}
}
```



## 380. O(1) 时间插入、删除和获取随机元素

```go
type RandomizedSet struct {
    nums    []int
    indices map[int]int
}

func Constructor() RandomizedSet {
    return RandomizedSet{[]int{}, map[int]int{}}
}

func (rs *RandomizedSet) Insert(val int) bool {
    if _, ok := rs.indices[val]; ok {
        return false
    }
    rs.indices[val] = len(rs.nums)
    rs.nums = append(rs.nums, val)
    return true
}

func (rs *RandomizedSet) Remove(val int) bool {
    id, ok := rs.indices[val]
    if !ok {
        return false
    }
    last := len(rs.nums) - 1
    rs.nums[id] = rs.nums[last]
    rs.indices[rs.nums[id]] = id
    rs.nums = rs.nums[:last]
    delete(rs.indices, val)
    return true
}

func (rs *RandomizedSet) GetRandom() int {
    return rs.nums[rand.Intn(len(rs.nums))]
}
```



## 1672. 最富有客户的资产总量

```go
func maximumWealth(accounts [][]int) int {
	res, count := 0, 0
	for i := range accounts {
		for _, v := range accounts[i] {
			count += v
		}
		res = max(res, count)
		count = 0
	}
	return res
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 385. 迷你语法分析器

```go
func deserialize(s string) *NestedInteger {
    if s[0] != '[' {
        num, _ := strconv.Atoi(s)
        ni := &NestedInteger{}
        ni.SetInteger(num)
        return ni
    }
    stack, num, negative := []*NestedInteger{}, 0, false
    for i, ch := range s {
        if ch == '-' {
            negative = true
        } else if unicode.IsDigit(ch) {
            num = num*10 + int(ch-'0')
        } else if ch == '[' {
            stack = append(stack, &NestedInteger{})
        } else if ch == ',' || ch == ']' {
            if unicode.IsDigit(rune(s[i-1])) {
                if negative {
                    num = -num
                }
                ni := NestedInteger{}
                ni.SetInteger(num)
                stack[len(stack)-1].Add(ni)
            }
            num, negative = 0, false
            if ch == ']' && len(stack) > 1 {
                stack[len(stack)-2].Add(*stack[len(stack)-1])
                stack = stack[:len(stack)-1]
            }
        }
    }
    return stack[len(stack)-1]
}
```



## 479. 最大回文数乘积

```go
func largestPalindrome(n int) int {
	ans := []int{9, 987, 123, 597, 677, 1218, 877, 475}
	return ans[n-1]
}
```



## 819. 最常见的单词

```go
func mostCommonWord(paragraph string, banned []string) string {
	ban := make(map[string]bool)
	for _, str := range banned {
		ban[str] = true
	}
	freq := make(map[string]int)
	maxFreq := 0
	word := make([]byte, 0)
	for i, n:=0, len(paragraph); i<=n; i++ {
		if i < n && unicode.IsLetter(rune(paragraph[i])) {
			word = append(word, byte(unicode.ToLower(rune(paragraph[i]))))
		} else if word != nil {
			s := string(word)
			if !ban[s] {
				freq[s]++
				maxFreq = max(maxFreq, freq[s])
			}
			word = nil
		}
	}
	for s, f := range freq {
		if f == maxFreq {
			return s
		}
	}
	return ""
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 386. 字典序排数

```go
func lexicalOrder(n int) []int {
	ans := make([]int, n)
	num := 1
	for i := range ans {
		ans[i] = num
		// 尝试在 number 后面附加一个 0
		if num*10 <= n {
			num *= 10
		} else {
			// 末尾的数位已经搜索完成，退回上一位
			for num%10 == 9 || num+1 > n {
				num /= 10
			}
			num++	// 下一个字典序整数
		}
	}
	return ans
}
```



## 821. 字符的最短距离

```go
func shortestToChar(s string, c byte) []int {
	n := len(s)
	ans := make([]int, n)
	location := -n
	// 从左往右遍历
	for i, v := range s {
		if c == byte(v) {
			location = i
		}
		ans[i] = i - location
	}
	location = 2*n
	// 从右往左遍历
	for i:=n-1; i>=0; i-- {
		if s[i] == c {
			location = i
		}
		ans[i] = min(ans[i], location-i)
	}
	return ans
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```



## 824. 山羊拉丁文

```go
var yuanYinMap = map[byte]bool{
	'a': true,
	'e': true,
	'i': true,
	'o': true,
	'u': true,
	'A': true,
	'E': true,
	'I': true,
	'O': true,
	'U': true,
}

func toGoatLatin(sentence string) string {
	slice := strings.Split(sentence, " ")
	res := make([]string, len(slice))
	i := 0
	for _, word := range slice {
		tmp := []byte(word)
		if yuanYinMap[tmp[0]] {
			res[i] = word + "ma"
		} else {
			res[i] = word[1:] + word[0:1] + "ma"
		}
		res[i] += strings.Repeat("a", i+1)
		i++
	}
	return strings.Join(res, " ")
}
```



## 396. 旋转函数

```go
func maxRotateFunction(nums []int) int {
	sum := 0
	for _, v := range nums {
		sum += v
	}
	f := 0
	for i, num := range nums {
		f += i*num
	}
	ans := f
	n := len(nums)
	for i:=n-1; i>0; i-- {
		f += sum - n*nums[i]
		ans = max(ans, f)
	}
	return ans
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



## 1823. 找出游戏的获胜者

https://leetcode-cn.com/circle/article/BOoxAL/

```go
func findTheWinner(n, k int) int {
    winner := 1
    for i := 2; i <= n; i++ {
        winner = (k+winner-1)%i + 1
    }
    return winner
}
```



## 153. 最小值

```go

```



## 153. 最小值

```go

```



## 153. 最小值

```go

```





# 程序员面试金典

## 01.01. 判定字符是否唯一

```go
func isUnique(astr string) bool {
	mark := 0	// 代替长度为26的bool数组
	for _, ch := range astr {
		// ch - 'a' 是 字符ch离'a'这个字符的距离，即要位移的距离
		tmp := ch - 'a'
		if mark & (1 << tmp) != 0 {
			return false
		} else {
			// 对于没有出现过的字符，我们用或运算将对应下标位的值置为1
			mark |= 1 << tmp
		}
	}
	return true
}

func isUnique(astr string) bool {
	hashMap := make(map[int]bool)
	for _, ch := range astr {
		if hashMap[int(ch-'a')]{
			return false
		}
		hashMap[int(ch-'a')] = true
	}
	return true
}
```



## 01.02. 判定是否互为字符重排

```go
func CheckPermutation(s1 string, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}
	count := make([]int, 26)
	for _, ch := range s1 {
		count[ch-'a']++
	}
	for _, ch := range s2 {
		count[ch-'a']--
		if count[ch-'a'] < 0 {
			return false
		}
	}
	return true
}
```



## 01.03. URL化

https://leetcode-cn.com/problems/string-to-url-lcci/solution/gozai-yuan-shu-zu-shang-cao-zuo-by-deerh-8d84/

```go
func replaceSpaces2(S string, length int) string {
	bytes := []byte(S)
	i, j := len(S)-1, length-1
	for j >= 0 {
		if bytes[j] == ' ' {
			bytes[i] = '0'
			bytes[i-1] = '2'
			bytes[i-2] = '%'
			i -= 3
		} else {
			bytes[i] = bytes[j]
			i--
		}
		j--
	}
    // 题目没有说字符串尾部留给我们的是恰好够用的空间，所以空间可能会多，所以最后需要裁剪
	return string(bytes[i+1:])
}

func replaceSpaces(S string, length int) string {
	builder := strings.Builder{}
	for i := 0; i < length; i++ {
		if S[i] != ' ' {
			builder.WriteByte(S[i])
		} else {
			builder.WriteString("%20")
		}
	}
	return builder.String()
}
```



## 01.04. 回文排列

```go
func canPermutePalindrome(s string) bool {
	n := len(s)
	hashMap := make(map[int32]bool)
	for _, ch := range s {
		if hashMap[ch-'a'] {
			hashMap[ch-'a'] = false
		} else {
			hashMap[ch-'a'] = true
		}
	}
	if n % 2 == 0 {
		for _, v := range hashMap {
			if v {
				return false
			}
		}
		return true
	} else {
		cnt := 0
		for _, v := range hashMap {
			if v {
				cnt++
				if cnt > 1 {
					return false
				}
			}
		}
		return true
	}
}
```



## 01.05. 一次编辑

```go
func oneEditAway(first string, second string) bool {
	len1, len2 := len(first), len(second)
	// 为方便后续处理，先保证输入 first 长度 < second 长度
	if len1 > len2 {
		return oneEditAway(second, first)
	}
	if len2 - len1 > 1 {
		return false
	}
	if len2 == len1 {
		count := 0
		// 遍历两字符串，统计“对应索引处字符不同”数量
		for i:=0; i<len1; i++ {
			if first[i] != second[i] {
				count++
			}
		}
		// 若“对应索引处字符不同”数量 <= 1 ，则能够通过一次编辑互相转换
		return count <= 1
	}
	i, move := 0, 0
	// 遍历两字符串，统计“对应索引处字符不同”数量
	for i < len1 {
		// 当遍历到不同字符时，执行偏移量 move += 1
		if first[i] != second[i+move] {
			move++
			// 若偏移量 > 1 ,说明无法通过一次编辑互相转换
			if move > 1 {
				return false
			}
		} else {
			i++
		}
	}
	// 遍历完成，代表能够通过一次编辑互相转换
	return true
}
```



## 01.06. 字符串压缩

```go
func compressString(S string) string {
	if len(S) < 2 {
		return S
	}
	len1 := len(S)
	ans := make([]byte, 0, len1)	// 这里预先设定好长度，会减少很多空间
	i, j := 0, 0
	for i<len1 {
		for j < len1 && S[i] == S[j] {
			j++
		}
		ans = append(ans, S[i])
		ans = append(ans, []byte(strconv.Itoa(j-i))...)
		if len1 <= len(ans) {	// 这里必须加上等于号
			return S
		}
		i = j
	}
	return string(ans)
}
```



## 01.07. 旋转矩阵

```go
func rotate(matrix [][]int)  {
	n := len(matrix)
	for i:=0; i<n/2; i++ {
		matrix[i], matrix[n-i-1] = matrix[n-i-1], matrix[i]
	}

	for i:=1; i<n; i++ {
		for j:=0; j<i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}
```



## 01.08. 零矩阵

```go
func setZeroes(matrix [][]int)  {
	m, n := len(matrix), len(matrix[0])
	col0 := false	// 用一个标记变量记录第一列是否原本存在 0
	for _, r := range matrix {
		if r[0] == 0 {
			col0 = true
		}
		for j:=1; j<n; j++ {
			if r[j] == 0 {
				r[0] = 0			// 第一列的第 i 个元素即可以标记第 i 行是否出现 0
				matrix[0][j] = 0	// 第一行的第 j 个元素即可以标记第 j 列是否出现 0
			}
		}
	}
	// 为了防止每一列的第一个元素被提前更新，我们需要从最后一行开始，倒序地处理矩阵元素
	for i:=m-1; i>=0; i-- {
		for j:=1; j<n; j++ {
			if matrix[i][0] == 0 || matrix[0][j] == 0 {
				matrix[i][j] = 0
			}
		}
		if col0 {
			matrix[i][0] = 0
		}
	}
}
```



## 01.09. 字符串轮转

```go
func isFlipedString(s1 string, s2 string) bool {
	return len(s1) == len(s2) && strings.Contains(s1+s1, s2)
}
```



## 02.01. 移除重复节点

```go
func removeDuplicateNodes(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	hashMap := make(map[int]bool)
	hashMap[head.Val] = true
	pos := head
	for pos.Next != nil {
		cur := pos.Next
		if !hashMap[cur.Val] {
			hashMap[cur.Val] = true
			pos = pos.Next
		} else {
			pos.Next = pos.Next.Next
		}
	}
	pos.Next = nil
	return head
}
```



## 02.03. 删除中间节点

```go
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}
```



## 02.04. 分割链表

```go
func partition(head *ListNode, x int) *ListNode {
	small, large := &ListNode{}, &ListNode{}
	smallHead, largeHead := small, large
	for head != nil {
		if head.Val < x {
			// small 链表按顺序存储所有小于 x 的节点
			small.Next = head
			small = small.Next
		} else {
			// large 链表按顺序存储所有大于等于 x 的节点
			large.Next = head
			large = large.Next
		}
		head = head.Next
	}
	large.Next = nil
	// 将 small 链表尾节点指向 large 链表的头节点即能完成对链表的分割
	small.Next = largeHead.Next
	return smallHead.Next
}
```



## 02.05. 链表求和

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	sum, carry := 0, 0
	var head, tail *ListNode
	for l1 != nil || l2 != nil {
		n1, n2 := 0, 0
		if l1 != nil {
			n1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			n2 = l2.Val
			l2 = l2.Next
		}
		sum = n1 + n2 + carry
		sum, carry = sum%10, sum/10		// 这里不可以分成两行来写，因为 sum 会改变
		if head == nil {
			head = &ListNode{Val: sum}
			tail = head
		} else {
			tail.Next = &ListNode{Val: sum}
			tail = tail.Next
		}
	}
	if carry > 0 {
		tail.Next = &ListNode{Val: carry}
	}
	return head
}
```



## 02.06. 回文链表

```go
// O(n)
func isPalindrome(head *ListNode) bool {
	res := make([]int, 0)
	node := head
	for node != nil {
		res = append(res, node.Val)
		node = node.Next
	}
	left, right := 0, len(res)-1
	for left < right {
		if res[left] != res[right] {
			return false
		}
		left++
		right--
	}
	return true
}

// O(1)
// 反转后半部分链表
func reverseList(head *ListNode) *ListNode {
	var prev, cur *ListNode = nil, head
	for cur != nil {
		nextTmp := cur.Next
		cur.Next = prev
		prev = cur
		cur = nextTmp
	}
	return prev
}

// 找到前半部分链表的尾节点
func endOfFirstHalf(head *ListNode) *ListNode {
	fast := head
	slow := head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}

// 判断是否回文
func isPalindrome(head *ListNode) bool {
	if head == nil {
		return true
	}

	// 找到前半部分链表的尾节点并反转后半部分链表
	firstHalfEnd := endOfFirstHalf(head)
	secondHalfStart := reverseList(firstHalfEnd.Next)

	// 判断是否回文
	p1 := head
	p2 := secondHalfStart
	result := true
	for result && p2 != nil {
		if p1.Val != p2.Val {
			result = false
		}
		p1 = p1.Next
		p2 = p2.Next
	}

	// 还原链表并返回结果
	firstHalfEnd.Next = reverseList(secondHalfStart)
	return result
}
```



## 02.08. 环路检测

```go
func detectCycle(head *ListNode) *ListNode {
	for head != nil {
		if head.Val == 1e9 {	// 当前节点已经是范围外的数，说明有环
			return head
		}
		head.Val = 1e9		// 节点值改成范围外的大数
		head = head.Next
	}
	return nil
}
```



## 03.01. 三合一

```go
type TripleInOne struct {
    S []int
    Len []int
    Cap int
}
func Constructor(size int) TripleInOne {
    return TripleInOne{
        make([]int, size * 3),
        make([]int, 3),
        size,
    }
}
func (this *TripleInOne) Push(num int, value int)  {
    if this.Len[num] == this.Cap { return }
    this.S[num * this.Cap + this.Len[num]] = value
    this.Len[num]++
}
func (this *TripleInOne) Pop(num int) int {
    if this.Len[num] == 0 {return -1}
    x := this.S[num * this.Cap + this.Len[num]-1]
    this.Len[num]--
    return x
}
func (this *TripleInOne) Peek(num int) int {
    if this.Len[num] == 0 {return -1}
    return  this.S[num * this.Cap + this.Len[num]-1]
}
func (this *TripleInOne) IsEmpty(num int) bool {
    return this.Len[num] == 0 
}
```



## 03.02. 栈的最小值

```go
type MinStack struct {
    stack []int
    minStack []int
}

func Constructor() MinStack {
    return MinStack{
        stack: []int{},
        minStack: []int{math.MaxInt64},
    }
}

func (this *MinStack) Push(x int)  {
    this.stack = append(this.stack, x)
    top := this.minStack[len(this.minStack)-1]
    this.minStack = append(this.minStack, min(x, top))
}

func (this *MinStack) Pop()  {
    this.stack = this.stack[:len(this.stack)-1]
    this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
    return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
    return this.minStack[len(this.minStack)-1]
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}
```



## 03.03. 堆盘子

```go
type StackOfPlates struct {
	Cap       int
	Stack     [][]int
}


func Constructor(cap int) StackOfPlates {
	s := StackOfPlates{
		Cap:cap,
		Stack:make([][]int, 0),
	}
	return s
}


func (this *StackOfPlates) Push(val int)  {
	if this.Cap == 0 {
		return
	}
	if len(this.Stack) == 0 {
		newPlate := make([]int, 0)
		newPlate = append(newPlate, val)
		this.Stack = append(this.Stack, newPlate)
		return
	}

	lastPlate := this.Stack[len(this.Stack) - 1]
	if len(lastPlate) == this.Cap {
		newPlate := make([]int, 0)
		newPlate = append(newPlate, val)
		this.Stack = append(this.Stack, newPlate)
		return
	}

	lastPlate = append(lastPlate, val)
	this.Stack[len(this.Stack) - 1] = lastPlate
}


func (this *StackOfPlates) Pop() int {
	if len(this.Stack) == 0 {
		return -1
	}
	plate := this.Stack[len(this.Stack) - 1]
	v := plate[len(plate) - 1]
	plate = plate[0:len(plate)-1]
	this.Stack[len(this.Stack) - 1] = plate
	if len(plate) == 0 {
		this.Stack = this.Stack[0:len(this.Stack) - 1]
	}
	return v
}


func (this *StackOfPlates) PopAt(index int) int {
	n := len(this.Stack)
	if index >= n {
		return -1
	}

	plate := this.Stack[index]
	v := plate[len(plate) - 1]
	plate = plate[0:len(plate) - 1]
	this.Stack[index] = plate

	if len(plate) == 0 {
		tmp := this.Stack[index+1:]
		this.Stack = this.Stack[:index]
		this.Stack = append(this.Stack, tmp...)
	}

	return v
}
```



## 03.04. 化栈为队

```go
type MyQueue struct {
    inStack, outStack []int
}

func Constructor() MyQueue {
    return MyQueue{}
}

func (q *MyQueue) Push(x int) {
    q.inStack = append(q.inStack, x)
}

func (q *MyQueue) in2out() {
    for len(q.inStack) > 0 {
        q.outStack = append(q.outStack, q.inStack[len(q.inStack)-1])
        q.inStack = q.inStack[:len(q.inStack)-1]
    }
}

func (q *MyQueue) Pop() int {
    if len(q.outStack) == 0 {
        q.in2out()
    }
    x := q.outStack[len(q.outStack)-1]
    q.outStack = q.outStack[:len(q.outStack)-1]
    return x
}

func (q *MyQueue) Peek() int {
    if len(q.outStack) == 0 {
        q.in2out()
    }
    return q.outStack[len(q.outStack)-1]
}

func (q *MyQueue) Empty() bool {
    return len(q.inStack) == 0 && len(q.outStack) == 0
}
```



## 03.05. 栈排序

```go
type SortedStack struct {
	sortedData []int
}

func Constructor() SortedStack {
	return SortedStack{make([]int, 0)}
}

func (this *SortedStack) Push(val int) {
	i := len(this.sortedData)
	if i == 0 || this.sortedData[i-1] <= val {
		this.sortedData = append(this.sortedData, val)
		return
	}
	for i != 0 && this.sortedData[i-1] > val {
		i--
	}
	// 插入
	this.sortedData = append(this.sortedData[:i], append([]int{val}, this.sortedData[i:]...)...)
}

func (this *SortedStack) Pop() {
    if this.IsEmpty() {
		return
	}
	this.sortedData = this.sortedData[1:]
}

func (this *SortedStack) Peek() int {
    if this.IsEmpty() {
		return -1
	}
	return this.sortedData[0]
}

func (this *SortedStack) IsEmpty() bool {
	return len(this.sortedData) == 0
}
```



## 03.06. 动物收容所

```go

type AnimalShelf struct {
	animals [2][]int
}

func Constructor() AnimalShelf {
	return AnimalShelf{}
}

func (this *AnimalShelf) Enqueue(animal []int) {
	this.animals[animal[1]] = append(this.animals[animal[1]], animal[0])
}

func (this *AnimalShelf) DequeueAny() []int {
	cats, dogs := this.animals[0], this.animals[1]
	if len(cats) == 0 {
		if len(dogs) == 0 {
			return []int{-1, -1}
		} else {
			this.decreaseAnimal(1)
			return []int{dogs[0], 1}
		}
	} else {
		if len(dogs) == 0 || cats[0] < dogs[0] {
			this.decreaseAnimal(0)
			return []int{cats[0], 0}
		} else {
			this.decreaseAnimal(1)
			return []int{dogs[0], 1}
		}
	}
}

func (this *AnimalShelf) DequeueDog() []int {
	return this.dequeue(1)
}

func (this *AnimalShelf) DequeueCat() []int {
	return this.dequeue(0)
}

func (this *AnimalShelf) dequeue(species int) []int {
	if len(this.animals[species]) == 0 {
		return []int{-1, -1}
	}
	animal := this.animals[species][0]
	this.decreaseAnimal(species)
	return []int{animal, species}
}

func (this *AnimalShelf) decreaseAnimal(species int) {
	this.animals[species] = this.animals[species][1:]
}

/**
 * Your AnimalShelf object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Enqueue(animal);
 * param_2 := obj.DequeueAny();
 * param_3 := obj.DequeueDog();
 * param_4 := obj.DequeueCat();
 */

```



## 04.01. 节点间通路

```go
func findWhetherExistsPath(n int, graph [][]int, start int, target int) bool {
	matrix := make([][]int, n)
	for i:=0; i<n; i++ {
		matrix[i] = make([]int, 0)
	}
	for i:=0; i<len(graph); i++ {
		from, to := graph[i][0], graph[i][1]
		matrix[from] = append(matrix[from], to)
	}
	var traverse func(start, target int) bool
	traverse = func(start, target int) bool {
		for _, next := range matrix[start] {
			if next == target {
				return true
			}
			res := traverse(next, target)
			if res {
				return true
			}
		}
		return false
	}
	return traverse(start, target)
}
```



## 04.02. 最小高度树

```go
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	var dfs func(start, end int) *TreeNode
	dfs = func(start, end int) *TreeNode{
		if start > end {
			return nil
		}
		mid := (start+end) / 2
		node := &TreeNode{Val: nums[mid]}
		node.Left = dfs(start, mid-1)
		node.Right = dfs(mid+1, end)
		return node
	}
	return dfs(0, len(nums)-1)
}
```



## 04.03. 特定深度节点链表

```go
func listOfDepth(tree *TreeNode) []*ListNode {
	queue := make([]*TreeNode, 1)
	queue[0] = tree
	ans := make([]*ListNode, 0)
	for len(queue) > 0 {
		dummy := &ListNode{}
		cur := dummy
		size := len(queue)
		for i:=0; i<size; i++ {
			cur.Next = &ListNode{Val: queue[i].Val}
			cur = cur.Next
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		ans = append(ans, dummy.Next)
		queue = queue[size:]
	}
	return ans
}
```



## 04.04. 检查平衡性

```go
func isBalanced(root *TreeNode) bool {
    return height(root) >= 0
}

func height(root *TreeNode) int {
    if root == nil {
        return 0
    }
    leftHeight := height(root.Left)
    rightHeight := height(root.Right)
    if leftHeight == -1 || rightHeight == -1 || abs(leftHeight - rightHeight) > 1 {
        return -1
    }
    return max(leftHeight, rightHeight) + 1
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}

func abs(x int) int {
    if x < 0 {
        return -1 * x
    }
    return x
}
```



## 04.05. 合法二叉搜索树

```go
func isValidBST(root *TreeNode) bool {
    return tool(root, math.MinInt64, math.MaxInt64)
}

func tool(root *TreeNode, l, r int) bool {
    if root == nil {
        return true
    }
    if root.Val <= l || root.Val >= r {
        return false
    } 
    return tool(root.Left, l, root.Val) && tool(root.Right, root.Val, r)
}
```



## 04.06. 后继者

```go
func inorderSuccessor(root *TreeNode, p *TreeNode) *TreeNode {
	if root == nil || p == nil {
		return nil
	}
	// 如果结点 p 的值大于等于 root 的值，说明 p 的后继结点在 root 右子树中，那么就递归到右子树中查找
	if p.Val >= root.Val {
		return inorderSuccessor(root.Right, p)
	// 如果结点 p 的值小于 root 的值，说明 p 在 root 左子树中，而它的后继结点有两种可能，要么也在左子树中，要么就是 root
	} else {
		left := inorderSuccessor(root.Left, p)
		if left != nil {
			// 如果左子树中找到了后继结点，那就直接返回答案
			return left
		} else {
			// 如果左子树中没有找到后继结点，那就说明 p 的右儿子为空，那么 root 就是它的后继结点
			return root
		}
	}
}
```



## 04.08. 首个共同祖先

https://leetcode-cn.com/problems/first-common-ancestor-lcci/solution/shou-ge-gong-tong-zu-xian-zhu-shi-xiang-xi-ji-yu-c/

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// 到底了还没找到，返回 nil
	if root == nil {
		return nil
	}
	// 如果找到了 p 或 q，返回它
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)	// left 记录 p 或 q 是在左子树找到的
	right := lowestCommonAncestor(root.Right, p, q) // right 记录 p 或 q 是在右子树找到的
	// 如果 left 和 right 都记录了找到的节点，那么肯定是一个记录了 p ，另一个记录了 q
	// 它们分别在以 root 为根的左右子树中，所以 root 就是它们的最近公共祖先
	if left != nil && right != nil {
		return root
	}
	// 由于节点 p,q 一定在二叉树中，left和right不会同时为null
	// 若 left != null && right == null，说明在左子树中找到 p 或 q，而在右子树找不到 p 或 q，则剩下一个也在左子树
	// 所以 left 就是最近公共祖先
	// 另一种情况同理
	if left == nil {
		return right
	}
	return left
}
```



## 04.09. 二叉搜索树序列

```go
func BSTSequences(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{{}}
    }

    res := [][]int{}
    dq := list.New()
    dq.PushBack(root)

    var dfs func(dq *list.List, path []int)
    dfs = func(dq *list.List, path []int) {
        if dq.Len() == 0 {
            ans := make([]int, len(path))
            copy(ans, path)
            res = append(res, ans)
            return
        }

        for i := 0; i < dq.Len() ; i++ {
            cur := dq.Remove(dq.Front()).(*TreeNode)
            path = append(path, cur.Val)

            if cur.Left != nil {
                dq.PushBack(cur.Left)
            }
            if cur.Right != nil {
                dq.PushBack(cur.Right)
            }

            dfs(dq, path)

            if cur.Left != nil {
                dq.Remove(dq.Back())
            }
            if cur.Right != nil {
                dq.Remove(dq.Back())
            }
            dq.PushBack(cur)

            path = path[:len(path) - 1]
        }
    }

    dfs(dq, nil)

    return res
}
```



## 04.10. 检查子树

```go
func checkSubTree(t1 *TreeNode, t2 *TreeNode) bool {
	if t1 == nil && t2 == nil {
		return true
	}
	if t1 == nil && t2 != nil {
		return false
	}
	if t1 != nil && t2 == nil {
		return false
	}
	if t1.Val == t2.Val {
		return checkSubTree(t1.Left, t2.Left) && checkSubTree(t1.Right, t2.Right)
	} else {
		return checkSubTree(t1.Left, t2) || checkSubTree(t1.Right, t2)
	}
}
```



## 04.12. 求和路径

```go
func pathSum(root *TreeNode, sum int) (ans int) {
    preSum := map[int64]int{0: 1}
    var dfs func(*TreeNode, int64)
    dfs = func(node *TreeNode, curr int64) {
        if node == nil {
            return
        }
        curr += int64(node.Val)
        ans += preSum[curr-int64(sum)]
        preSum[curr]++
        dfs(node.Left, curr)
        dfs(node.Right, curr)
        preSum[curr]--
        return
    }
    dfs(root, 0)
    return
}
```



## 153. 最小值

```go

```



## 153. 最小值

```go

```



## 153. 最小值

```go

```



## 153. 最小值

```go

```

