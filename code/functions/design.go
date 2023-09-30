package functions

import "container/list"

type LRUNode struct {
	key, value int
}

type LRUCache struct {
	capacity int
	cache    map[int]*list.Element // Element 用于代表双链表的元素
	LRUList  *list.List            // List 用于表示双链表
}

func LRUCacheConstructor(capacity int) LRUCache {
	return LRUCache{
		capacity: capacity,
		cache:    map[int]*list.Element{},
		LRUList:  list.New(), // 通过 container/list 包的 New() 函数初始化 list
	}
}

func (this *LRUCache) Get(key int) int {
	element := this.cache[key] // 获得 valve
	if element == nil {        // 关键字 key 不在缓存中
		return -1
	}
	this.LRUList.MoveToFront(element) // 刷新缓存使用时间	将元素 e 移动到链表的开头
	return element.Value.(LRUNode).value
}

func (this *LRUCache) Put(key int, value int) {
	element := this.cache[key]
	if element != nil { // 关键字 key 已经存在，则变更其数据值 value
		element.Value = LRUNode{key: key, value: value}
		this.LRUList.MoveToFront(element) // 刷新缓存使用时间	将元素 e 移动到链表的开头
		return
	}
	// 如果不存在，则向缓存中插入该组 key-value
	this.cache[key] = this.LRUList.PushFront(LRUNode{key: key, value: value}) //将包含了值v的元素e插入到链表的开头并返回e
	if len(this.cache) > this.capacity {                                      // 如果插入操作导致关键字数量超过 capacity ，则应该逐出最久未使用的关键字
		delete(this.cache, this.LRUList.Remove(this.LRUList.Back()).(LRUNode).key)
	}
}

// 54. 螺旋矩阵
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])
	ans := make([]int, 0)
	top, bottom, left, right := 0, m-1, 0, n-1
	for left <= right && top <= bottom {
		for i := left; i <= right; i++ { // 左上方到右
			ans = append(ans, matrix[top][i])
		}
		top++
		for i := top; i <= bottom; i++ { // 右上方到下
			ans = append(ans, matrix[i][right])
		}
		right--
		// 这里的判断条件必须是&&，不能是||
		if left <= right && top <= bottom { // 当 left > right 或者 top > bottom 时，不会发生右到左和下到上，否则会重复计数
			for i := right; i >= left; i-- { // 右下方到左
				ans = append(ans, matrix[bottom][i])
			}
			bottom--
			for i := bottom; i >= top; i-- { // 左下方到上
				ans = append(ans, matrix[i][left])
			}
			left++
		}
	}
	return ans
}
