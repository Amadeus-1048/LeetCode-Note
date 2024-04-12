package main

import (
	"fmt"
	"sync"
)

const (
	MAX     = 20 // 打印多少值
	GoCount = 4  // 几个协程
)

func main() {
	fmt.Println(solution2(MAX, GoCount))
}

func solution2(max, goCount int) *[]int {
	result := make([]int, 0, max)
	wgLine := make([]*sync.WaitGroup, goCount) // 控制不同 goroutine 的执行顺序
	wg := &sync.WaitGroup{}                    // 等待所有 goroutine 的完成

	// 循环创建 goCount 个 goroutine
	// 每个 goroutine 都有一个自己的 WaitGroup（selfWg）和一个指向下一个 goroutine 的 WaitGroup（nextWg）
	for i := 0; i < goCount; i++ {
		wgLine[i] = &sync.WaitGroup{}
		wgLine[i].Add(1)
	}

	count := 1
	wg.Add(goCount)
	for i := 0; i < goCount; i++ { // 对于每个 goroutine
		go func(max int, selfWg, nextWg *sync.WaitGroup) {
			for {
				selfWg.Wait() // 在开始时等待自己的 WaitGroup（selfWg）
				if count > max {
					wg.Done()     // 表示完成
					selfWg.Add(1) // 重新加一个等待计数到 selfWg
					nextWg.Done() // 触发下一个 goroutine 的 WaitGroup （nextWg.Done()），然后退出
					return
				}
				//println(count)
				result = append(result, count)
				count++
				selfWg.Add(1) // 当前 goroutine 重新为自己的 WaitGroup 加一（selfWg.Add(1)）
				nextWg.Done() // 触发下一个 goroutine 的 WaitGroup （nextWg.Done()）
			}
		}(max, wgLine[i], wgLine[(i+1)%goCount])

		if i == 0 { // 手动触发第一个 goroutine
			wgLine[goCount-1].Done() // 第0个goroutine是由最后一个goroutine触发的
		}
	}
	wg.Wait()
	return &result
}
