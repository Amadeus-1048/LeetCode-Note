package main

import (
	"fmt"
	"sync"
)

// go 通过goroutine 和 channel 实现交替打印并发控制
func main() {
	// 设定要起的协程数量
	var goroutineNum = 3
	// 最大打印整数
	var maxPrintNum = 30
	// 设定等待协程
	waitGoroutine := new(sync.WaitGroup)
	waitGoroutine.Add(goroutineNum)
	// 第一层管道，主要目的是把最后一协程生成的管道信号重新传递给第一个协程
	firstChannel := make(chan int)
	// 临时channel
	var temp chan int
	// 循环启动协程
	for i := 0; i < goroutineNum; i++ {
		// 每次循环把上一个goroutine里面的channel给带入到下一层
		if i == 0 {
			// 第一次是从主函数main生成的第一层channel
			temp = PrintNumber(firstChannel, i+1, maxPrintNum, waitGoroutine)
		} else {
			temp = PrintNumber(temp, i+1, maxPrintNum, waitGoroutine)
		}
	}
	// 第一层管道先增加一个量，从0开始计算
	firstChannel <- 0
	// 这里最终接受到的是 最后一个协程生成的 nextChannel
	for v := range temp {
		firstChannel <- v
	}
	close(firstChannel)
	waitGoroutine.Wait()
}

// PrintNumber 打印数字
// preChan 上一个协程生成的信号管道
// nowGoroutineFlag 当前协程标志,没啥用，就是为了看清打印的时候是哪个协程再打印
// maxNum 整个程序打印的最大数字
// wg 等待组，为了优雅退出
func PrintNumber(preChan chan int, nowGoroutineFlag int, maxNum int, wg *sync.WaitGroup) chan int {
	wg.Done()
	// 生成一个新的channel 用于给下一个goroutine传递信号
	nextChannel := make(chan int)
	// 把上一个goroutine的channel带入到新的协程里
	go func(preChan chan int) {
		// 上一个协程没有塞入数据之前这里是阻塞的
		for v := range preChan {
			// 如果上一个协程的channel发送了信号,这里将解除阻塞

			if v > maxNum {
				// 不再继续生成新的数字
				break
			} else {
				// 当前要打印的数字
				nowNum := v + 1
				// 打印当前协程标识和数字
				fmt.Printf("当前协程为第 %d 协程,当前数字为：%d \n", nowGoroutineFlag, nowNum)
				// 往下一个协程需要用到的channel里塞入信号
				nextChannel <- nowNum
			}
		}
		// 根据go的管道关闭原则，尽可能的在发送方关闭管道
		// 完成当前协程所有任务后，关闭管道
		close(nextChannel)
	}(preChan)
	return nextChannel
}
