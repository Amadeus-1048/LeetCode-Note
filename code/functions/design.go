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
