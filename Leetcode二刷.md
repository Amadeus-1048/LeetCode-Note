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

初始时，cur指向虚拟头结点，然后进行如下三步：

<img src="https://code-thinking.cdn.bcebos.com/pics/24.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B91.png" alt="24.两两交换链表中的节点1" style="zoom:50%;" />

操作之后，链表如下：

<img src="https://code-thinking.cdn.bcebos.com/pics/24.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B92.png" alt="24.两两交换链表中的节点2" style="zoom:50%;" />

```go
还是要画一下图，记住三个步骤

head.Next.Next = head	对结点指向哪里进行修改
pre = head				用变量pre表示head结点（即pre和head同时表示一个结点）
```



## 19. 删除链表的倒数第N个节点

答案

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	fast, slow := dummy, dummy
	for i:=0; i<=n; i++ {
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



分析

```go
双指针的经典应用
定义fast指针和slow指针，初始值为虚拟头结点
fast首先走n + 1步 ，为什么是n+1呢，因为只有这样同时移动的时候slow才能指向删除节点的上一个节点（方便做删除操作）
fast和slow同时移动，直到fast指向末尾(NULL)
删除slow指向的下一个节点
```



## 160. 相交链表

答案

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
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



分析

```go
双指针
如果有相交，那么相交时两个指针走的步数相等，重合在相交点
如果没有，两个指针会走完两轮，同时指向null，此时相等，退出循环
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
		fast = fast.Next.Next
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





# 哈希表

## 242.有效的字母异位词

答案

```go
func isAnagram(s string, t string) bool {
	record := make([]int, 26)
	for i:=0; i<len(s); i++ {
		record[s[i]-'a']++
	}
	for i:=0; i<len(t); i++ {
		record[t[i]-'a']--
	}
	for i:=0; i<26; i++ {
		if record[i] != 0 {
			return false
		}
	}
	return true
}
```



分析

```go
把字符映射到数组也就是哈希表的索引下标上
因为字符a到字符z的ASCII是26个连续的数值
所以字符a映射为下标0，相应的字符z映射为下标25
```



## 349.两个数组的交集

答案

```go
func intersection(nums1 []int, nums2 []int) []int {
	m := make(map[int]int)
	for _, v := range nums1 {
		m[v] = 1	// 注意是赋值为1，不是++，因为交集中每个数字只出现一次
	}
	res := make([]int, 0)
	for _, v := range nums2 {
		if count, ok := m[v]; ok && count>0 {
			res = append(res, v)
			m[v]--	// 避免交集中出现重复的数字
		}
	}
	return res
}
```



分析

```go

```



## 202. 快乐数

答案

```go
func isHappy(n int) bool {
    m := make(map[int]bool)
    for n != 1 && !m[n] {
        n, m[n] = getSum(n), true
    }
    return n == 1
}

func getSum(n int) int {
    sum := 0
    for n > 0 {
        sum += (n % 10) * (n % 10)
        n = n / 10
    }
    return sum
}
```



分析

```go

```



## 1. 两数之和

答案

```go
func twoSum(nums []int, target int) []int {
	hashMap := make(map[int]int)
	for i, v := range nums {
    // 这样处理是为了防止出现重复的数
		if j, ok := hashMap[target-v]; ok {
			return []int{i,j}
		}
		hashMap[v] = i
	}
	return []int{}
}
```



分析

```go
什么时候使用哈希法：当我们需要查询一个元素是否出现过，或者一个元素是否在集合里的时候，就要第一时间想到哈希法

不仅要知道元素有没有遍历过，还要知道这个元素对应的下标
所以需要使用 key value 结构来存放
key来存元素，value来存下标
```



## 454. 两数相加II

答案

```go
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	m := make(map[int]int)
	count := 0
  // 遍历A和B数组，统计两个数组元素之和、出现的次数，放到map中
	for _, v1 := range nums1 {
		for _, v2 := range nums2 {
			m[v1+v2]++
		}
	}
  // 遍历C和D数组，找到如果 0-(c+d) 在map中出现过的话，就用count把map中key对应的value也就是出现次数统计出来
	for _, v3 := range nums3 {
		for _, v4 := range nums4 {
			count += m[-v3-v4]
		}
	}
	return count
}
```



分析

```go
创建的map中，key放a和b两数之和，value 放a和b两数之和出现的次数
```



## 383. 赎金信

答案

```go
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
```



分析

```go
因为题目所只有小写字母，那可以采用空间换取时间的哈希策略，用一个长度为26的数组还记录magazine里字母出现的次数
```



## 15. 三数之和

答案

```go
func threeSum(nums []int) [][]int  {
	sort.Ints(nums)		// 排序是为了去重
	res := [][]int{}
	length := len(nums)
	for i:=0; i<length-2; i++ {
		n1 := nums[i]	// 先选第一个数 n1
		if n1 > 0 {		// 第一个数就大于0，则不可能三个数和为0
			break	// 接下来都不用再试了
		}
		if i>0 && nums[i]==nums[i-1] {	// 避免重复的答案 比如 -1 -1 0 1
			continue	// 只是跳过当前的i
		}
		l, r := i+1, length-1
		for l < r {
			n2, n3 := nums[l], nums[r]
			if n1+n2+n3 == 0 {
				res = append(res, []int{n1, n2, n3})
				for l<r && n2==nums[l] {
					l++
				}
				for l<r && n3==nums[r] {
					r--
				}
			} else if n1+n2+n3 < 0 {
				l++
			} else {
				r--
			}
		}
	}
	return res
}
```



分析

```go
这道题目使用 双指针法 要比 哈希法 高效一些
还要注意去重
```



## 18. 四数之和

答案

```go
func fourSum(nums []int, target int) [][]int {
	if len(nums) < 4 {
		return nil
	}
	sort.Ints(nums)
	var res [][]int
	for i := 0; i < len(nums)-3; i++ {
		n1 := nums[i]
		// if n1 > target { // 不能这样写,因为可能是负数
		// 	break
		// }
		if i > 0 && n1 == nums[i-1] {
			continue
		}
		for j := i + 1; j < len(nums)-2; j++ {
			n2 := nums[j]
			if j > i+1 && n2 == nums[j-1] {
				continue
			}
			l := j + 1
			r := len(nums) - 1
			for l < r {
				n3 := nums[l]
				n4 := nums[r]
				sum := n1 + n2 + n3 + n4
				if sum < target {
					l++
				} else if sum > target {
					r--
				} else {
					res = append(res, []int{n1, n2, n3, n4})
					for l < r && n3 == nums[l] { // 去重	这个for循环会至少进入一次（n3和自己作比较）
						l++
					}
					for l < r && n4 == nums[r] { // 去重	这个for循环会至少进入一次（n4和自己作比较）
						r--
					}
				}
			}
		}
	}
	return res
}
```



分析

```go
四数之和，和15.三数之和是一个思路，都是使用双指针法, 基本解法就是在15.三数之和的基础上再套一层for循环

四数之和这道题目 target是任意值

对于15.三数之和双指针法就是将原本暴力O(n^3)的解法，降为O(n^2)的解法
四数之和的双指针解法就是将原本暴力O(n^4)的解法，降为O(n^3)的解法
```



# 字符串

## 344. 反转字符串

答案

```go
func reverseString(s []byte)  {
	left := 0
	right := len(s)-1
	for left<right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}
```



分析

```go
对于字符串，我们定义两个指针（也可以说是索引下标），一个从字符串前面，一个从字符串后面，两个指针同时向中间移动，并交换元素
```



## 541. 反转字符串II

答案

```go
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
```



分析

```go

```



## 剑指 Offer 05. 替换空格

答案

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



分析

```go
很多数组填充类的问题，都可以先预先给数组扩容带填充后的大小，然后在从后向前进行操作。

这么做有两个好处：
	不用申请新数组。
	从后向前填充元素，避免了从前向后填充元素时，每次添加元素都要将添加元素之后的所有元素向后移动的问题。
```



## 151.反转字符串中的单词

答案

```go
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
	reverse151(b, 0, len(b)-1)
	//3.反转单个单词  i单词开始位置，j单词结束位置
	i := 0
	for i < len(b) {
		j := i
		for ; j < len(b) && b[j] != ' '; j++ {
		}
		reverse151(b, i, j-1)
		i = j
		i++
	}
	return string(b)
}

func reverse151(b []byte, left, right int) {
	for left < right {
		b[left], b[right] = b[right], b[left]
		left++
		right--
	}
}
```



分析

```go
将整个字符串都反转过来，那么单词的顺序指定是倒序了，只不过单词本身也倒序了，那么再把单词反转一下，单词不就正过来了。

所以解题思路如下：
	移除多余空格
	将整个字符串反转
	将每个单词反转
```



## 剑指 Offer 58 - II. 左旋转字符串

答案

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



分析

```go
可以通过局部反转+整体反转 达到左旋转的目的。

具体步骤为：
	反转区间为前n的子串
	反转区间为n到末尾的子串
	反转整个字符串

最后就可以达到左旋n的目的，而不用定义新的字符串，完全在本串上操作
```







# 树

## 二叉树递归遍历

答案

### 144. 前序遍历

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



### 94. 中序遍历

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



### 145. 后序遍历

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



## 102. 二叉树的层序遍历

答案

```go
func levelOrder(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue)>0 {
        tmp := []int{}
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
			tmp = append(tmp, node.Val)	//将值加入本层切片中
		}
		ans = append(ans, tmp)	//放入结果集
	}
	return ans
}
```



分析

```go

```



## 107. 二叉树的层序遍历II

答案

```go
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
```



分析

```go

```



## 199. 二叉树的右视图

答案

```go
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
```



分析

```go
每次返回每层的最后一个字段即可
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



## 429. N 叉树的层序遍历

答案

```go
type Node struct {
	Val      int
	Children []*Node
}

func levelOrder(root *Node) [][]int {
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
```



分析

```go

```



## 515.在每个树行中找最大值

答案

```go
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
```



分析

```go

```



## 116. 填充每个节点的下一个右侧节点指针

答案

```go
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
```



分析

```go

```



## 117. 填充每个节点的下一个右侧节点指针 II

答案

```go
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
```



分析

```go

```



## 104. 二叉树的最大深度

答案

```go
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
```



分析

```go

```



## 111. 二叉树的最小深度

答案

```go
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
```



分析

```go
相对于 104.二叉树的最大深度 ，本题还也可以使用层序遍历的方式来解决，思路是一样的。

需要注意的是，只有当左右孩子都为空的时候，才说明遍历的最低点了。如果其中一个孩子为空则不是最低点
```



## 226. 翻转二叉树

答案

```go
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
```



分析

```go
只要把每一个节点的左右孩子翻转一下，就可以达到整体翻转的效果

这道题目使用前序遍历、后序遍历、层序遍历都可以，唯独中序遍历不方便，因为中序遍历会把某些节点的左右孩子翻转了两次
```



## 101. 对称二叉树

答案

```go
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
```



分析

```go
通过队列来判断根节点的左子树和右子树的内侧和外侧是否相等
```



## 222. 完全二叉树的节点个数

答案

```go
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
```



分析

```go

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





## 110. 平衡二叉树

答案

```go
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



分析

```go
二叉树节点的深度：指从根节点到该节点的最长简单路径边的条数。
二叉树节点的高度：指从该节点到叶子节点的最长简单路径边的条数。

因为求深度可以从上到下去查 所以需要前序遍历（中左右），而高度只能从下到上去查，所以只能后序遍历（左右中）
```



## 257. 二叉树的所有路径

答案

```go
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
```



分析

```go

```



## 404. 左叶子之和

答案

```go
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
```



分析

```go
左叶子的明确定义：
节点A的左孩子不为空，且左孩子的左右孩子都为空（说明是叶子节点），那么A节点的左孩子为左叶子节点
```



## 513. 找树左下角的值

答案

```go
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
			if i == 0 {	// 记录每一行第一个元素
				ans = node.Val
			}
		}
	}
	return ans
}
```



分析

```go
在树的最后一行找到最左边的值
```



## 112. 路径总和

答案

```go
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
```



分析

```go

```



## 113. 路径总和II

答案

```go
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
```



分析

```go
要遍历整个树，找到所有路径，所以需要回溯
```



## 106. 从中序与后序遍历序列构造二叉树

答案

```go
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

```



分析

```go
如何根据两个顺序构造一个唯一的二叉树：
以后序数组的最后一个元素为切割点，先切中序数组，根据中序数组，反过来再切后序数组。
一层一层切下去，每次后序数组最后一个元素就是节点元素。
```



## 105. 从前序与中序遍历序列构造二叉树

答案

```go
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
```



分析

```go
前序和中序可以唯一确定一棵二叉树。

后序和中序可以唯一确定一棵二叉树。

前序和后序不能唯一确定一棵二叉树！因为没有中序遍历无法确定左右部分，也就是无法分割
```



## 654. 最大二叉树

答案

```go
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
```



分析

```go

```



## 617. 合并二叉树

答案

```go
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
```



分析

```go

```



## 700. 二叉搜索树中的搜索

答案

```go
// 递归
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

// 迭代
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
```



分析

```go
因为二叉搜索树的特殊性，也就是节点的有序性，可以不使用辅助栈或者队列就可以写出迭代法
```



## 98. 验证二叉搜索树

答案

```go
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
```



分析

```go
在中序遍历下，输出的二叉搜索树节点的数值是有序序列
有了这个特性，验证二叉搜索树，就相当于变成了判断一个序列是不是递增的了
```



## 530. 二叉搜索树的最小绝对差

答案

```go
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
```



分析

```go
最直观的想法，就是把二叉搜索树转换成有序数组，然后遍历一遍数组，就统计出来最小差值了
```



## 501.二叉搜索树中的众数

答案

```go
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
```



分析

```go

```



## 236. 二叉树的最近公共祖先

答案

```go
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
```



分析

```go
遇到这个题目首先想的是要是能自底向上查找就好了，这样就可以找到公共祖先了
二叉树回溯的过程就是从低到上，后序遍历（左右中）就是天然的回溯过程，可以根据左右子树的返回值，来处理中节点的逻辑
如果找到一个节点，发现左子树出现结点p，右子树出现节点q，或者 左子树出现结点q，右子树出现节点p，那么该节点就是节点p和q的最近公共祖先
```



## 235. 二叉搜索树的最近公共祖先

答案

```go
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
```



分析

```go
因为是有序树，所有 如果 中间节点是 q 和 p 的公共祖先，那么 中节点的数组 一定是在 [p, q]区间的

当我们从上向下去递归遍历，第一次遇到 cur节点是数值在[p, q]区间中，那么cur就是 p和q的最近公共祖先
```



## 701. 二叉搜索树中的插入操作

答案

```go
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
```



分析

```go
只要遍历二叉搜索树，找到空节点 插入元素就可以
```



## 450.删除二叉搜索树中的节点

答案

```go
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
```



分析

```go
只要遍历二叉搜索树，找到空节点 插入元素就可以
```



## 669. 修剪二叉搜索树

答案

```go
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return nil
	}
	// 如果该节点值小于最小值，则该节点更换为该节点的经过递归处理的右节点值，继续遍历
	if root.Val < low {
		return trimBST(root.Right, low, high)
	}
	// 如果该节点的值大于最大值，则该节点更换为该节点的经过递归处理的左节点值，继续遍历
	if root.Val > high {
		return trimBST(root.Left, low, high)
	}
	// 该节点的值在low和high之间，则处理符合条件的左右子树
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}
```



分析

```go

```



## 108. 将有序数组转换为二叉搜索树

答案

```go
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
```



分析

```go
如果根据数组构造一棵二叉树，本质就是寻找分割点，分割点作为当前节点，然后递归左区间和右区间

有序数组构造二叉搜索树，分割点就是数组中间位置的节点。

本题要构造二叉树，依然用递归函数的返回值来构造中节点的左右孩子
```



## 538. 把二叉搜索树转换为累加树

答案

```go
func convertBST(root *TreeNode) *TreeNode {
	sum := 0
	var rightMLeft func(root *TreeNode)
	rightMLeft = func(root *TreeNode) {
		if root == nil { // 终止条件，遇到空节点就返回
			return
		}
		rightMLeft(root.Right) // 先遍历右边
		tmp := sum             // 暂存总和值
		sum += root.Val        // 将总和值变更
		root.Val += tmp        // 更新节点值
		rightMLeft(root.Left)  // 遍历左节点
		return
	}
	rightMLeft(root)
	return root
}
```



分析

```go

```





# 回溯

## 77. 组合

答案

```go
func combine(n int, k int) [][]int {
	res := [][]int{}
	var backtrace func(start int, trace []int)
	backtrace = func(start int, trace []int) {
		if len(trace) == k {
			tmp := make([]int, k)
			copy(tmp, trace)
			res = append(res, tmp)
		}
		if len(trace)+n-start+1 < k { // 剪枝优化
			return
		}
		for i := start; i <= n; i++ { // 选择本层集合中元素，控制树的横向遍历
			trace = append(trace, i)     // 处理节点
			backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
			trace = trace[:len(trace)-1] // 回溯，撤销处理结果
		}
	}
	backtrace(1, []int{})
	return res
}
```



分析

```go

```



## 216. 组合总和 III

答案

```go
func combinationSum3(k int, n int) [][]int {
	res := [][]int{}
	var backtrace func(start int, trace []int)
	backtrace = func(start int, trace []int) {
		if len(trace) == k {
			sum := 0
			tmp := make([]int, k)
			for i, v := range trace {
				sum += v
				tmp[i] = v
			}
			if sum == n {
				res = append(res, tmp)
			}
			return
		}
		if start > n { // 剪枝优化
			return
		}
		for i := start; i <= 9; i++ { // 选择本层集合中元素，控制树的横向遍历
			trace = append(trace, i)     // 处理节点
			backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
			trace = trace[:len(trace)-1] // 回溯，撤销处理结果
		}
	}
	backtrace(1, []int{})
	return res
}
```



分析

```go

```



## 17. 电话号码的字母组合

答案

```go
func letterCombinations(digits string) []string {
	digitsMap := [10]string{
		"",     // 0
		"",     // 1
		"abc",  // 2
		"def",  // 3
		"ghi",  // 4
		"jkl",  // 5
		"mno",  // 6
		"pqrs", // 7
		"tuv",  // 8
		"wxyz", // 9
	}

	res := []string{}
	length := len(digits)
	if length <= 0 || length > 4 {
		return res
	}
	var backtrace func(s string, index int)
	backtrace = func(s string, index int) {
		if len(s) == length {
			res = append(res, s)
			return
		}
		num := digits[index] - '0' // 将index指向的数字转为int
		letter := digitsMap[num]   // 取数字对应的字符集
		for i := 0; i < len(letter); i++ {
			s += string(letter[i])
			backtrace(s, index+1)
			s = s[:len(s)-1]
		}
	}
	backtrace("", 0)
	return res
}
```



分析

```go

```





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
    
剪枝优化的思路：
    对candidates排序之后，如果下一层的sum（就是本层的 sum + candidates[i]）已经大于target，就可以结束本轮for循环的遍历。
```





## 40. 组合总和 II

答案

```go
func combinationSum2(candidates []int, target int) [][]int {
	res := [][]int{}
	trace := []int{}
	sort.Ints(candidates)
	var backtrace func(start, sum int)
	backtrace = func(start, sum int) {
		if sum == target {
			tmp := make([]int, len(trace))
			copy(tmp, trace)
			res = append(res, tmp)
			return
		}
		if sum > target {
			return
		}

		for i:=start; i<len(candidates); i++ {
			if i>start && candidates[i]==candidates[i-1] {
				continue
			}
			trace = append(trace, candidates[i])
			sum += candidates[i]
			backtrace(i+1, sum)
			trace = trace[:len(trace)-1]
			sum -= candidates[i]
		}
	}
	backtrace(0, 0)
	return res
}
```



分析

```
本题的难点在于：集合（数组candidates）有重复元素，但还不能有重复的组合
即要去重的是同一树层上的“使用过”。同一树枝上的都是一个组合里的元素，不用去重
加一个bool型数组used，用来记录同一树枝上的元素是否使用过
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
由于栈结构的特殊性，非常适合做对称匹配类的题目
```



## 1047. 删除字符串中的所有相邻重复项

```go
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
```



分析

```go
栈用来存放遍历过的元素
当遍历当前元素时，去栈里看一下是不是遍历过相同数值的相邻元素，然后再去做对应的消除操作
```



## 150. 逆波兰表达式求值

```go
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
```



分析

```go

```



## 239. 滑动窗口最大值

```go
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

// 如果push的元素value大于队列尾的数值，那么就将队列尾的元素弹出，直到value小于等于队列尾元素的数值为止（或队列为空）
func (m *MyQueue) Push(val int) {
	for !m.Empty() && val > m.Back() {
		m.queue = m.queue[:len(m.queue)-1]
	}
	m.queue = append(m.queue, val)
}

// 如果窗口移除的元素value等于单调队列头元素，那么队列弹出元素，否则不用任何操作
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
```



分析

```go
维护元素单调递减的队列就叫做单调队列，即单调递减或单调递增的队列
```



## 347. 前 K 个高频元素

```go
func topKFrequent(nums []int, k int) []int {
	// 初始化一个map，用来存数字和数字出现的次数
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
	sort.Slice(res, func(i, j int) bool {	 //利用O(nlogn)排序
		return hashMap[res[i]] > hashMap[res[j]]
	})
	return res[:k]
}
```



分析

```go

```



## 71. 简化路径

```go
func simplifyPath(path string) string {
    stack := []string{}
    for _, name := range strings.Split(path, "/") {
        // 对于「空字符串」以及「一个点」，我们实际上无需对它们进行处理
        // 遇到「两个点」时，需要将目录切换到上一级，因此只要栈不为空，就弹出栈顶的目录
        if name == ".." {
            if len(stack) > 0 {
                stack = stack[:len(stack)-1]
            }
        } else if name != "" && name != "." { 
            stack = append(stack, name) // 遇到「目录名」时，就把它放入栈
        }
    }
    // 将从栈底到栈顶的字符串用 / 进行连接，再在最前面加上 / 表示根目录
    return "/" + strings.Join(stack, "/")
}
```



分析

```go
将给定的字符串 path 根据 / 分割成一个由若干字符串组成的列表，记为 names。
根据题目中规定的「规范路径的下述格式」，names 中包含的字符串只能为以下几种：
	空字符串。例如当出现多个连续的 /，就会分割出空字符串；
	一个点 .；
	两个点 ..；
	只包含英文字母、数字或 _ 的目录名
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





