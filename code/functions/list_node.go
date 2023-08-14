package functions

type ListNode struct {
	Val  int
	Next *ListNode
}

// 203. 移除链表元素
func removeElements(head *ListNode, val int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	cur := dummy
	for cur.Next != nil {
		if cur.Next.Val == val {
			cur.Next = cur.Next.Next
		} else {
			cur = cur.Next
		}
	}
	return dummy.Next
}

// 707. 设计链表
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

// 206. 反转链表
func reverseList(head *ListNode) *ListNode {
	cur := head
	var pre *ListNode // 不能用pre := &ListNode{}    输出结果会在最后多一个0
	for cur != nil {
		tmp := cur.Next
		cur.Next = pre
		pre = cur
		cur = tmp
	}
	return pre // 是返回pre，不是cur，因为最后cur是nil
}

// 24. 两两交换链表中的节点
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

// 19. 删除链表的倒数第N个节点
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{}
	dummy.Next = head
	fast, slow := dummy, dummy
	for i := 0; i <= n; i++ {
		fast = fast.Next
	}
	for fast != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}

// 160. 相交链表
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

// 141. 环形链表
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

// 142. 环形链表II
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next // 任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍
		if slow == fast {     // 找到重合的节点，说明在环中
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
