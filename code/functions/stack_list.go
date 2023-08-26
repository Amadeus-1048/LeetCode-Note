package functions

import "strconv"

// 20. 有效的括号
func isValid(s string) bool {
	hash := map[byte]byte{')': '(', ']': '[', '}': '{'}
	stack := []byte{} // 注意是string中的每个字符是byte类型
	if s == "" {
		return true
	}
	for i := 0; i < len(s); i++ {
		if s[i] == '(' || s[i] == '{' || s[i] == '[' {
			stack = append(stack, s[i]) // 注意是string中的每个字符是byte类型
		} else if len(stack) > 0 && stack[len(stack)-1] == hash[s[i]] {
			stack = stack[:len(stack)-1]
		} else {
			return false
		}
	}
	return len(stack) == 0
}

// 1047. 删除字符串中的所有相邻重复项
func removeDuplicates(s string) string {
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		// 栈不空 且 与栈顶元素相等
		if len(stack) > 0 && stack[len(stack)-1] == s[i] {
			// 弹出栈顶元素 并 忽略当前元素(s[i])
			stack = stack[:len(stack)-1]
		} else {
			stack = append(stack, s[i])
		}
	}
	return string(stack)
}

// 150. 逆波兰表达式求值
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

// 239. 滑动窗口最大值
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
