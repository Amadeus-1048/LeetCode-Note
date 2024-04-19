package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	inputs := bufio.NewScanner(os.Stdin) // 不用放在循环内部
	buf := make([]byte, 64*1024)         // 创建一个更大的缓冲区
	inputs.Buffer(buf, 1024*1024)        // 设置缓冲区和最大 token 大小
	for inputs.Scan() {                  //每次读入一行
		data := strings.Split(inputs.Text(), " ") //通过空格将他们分割，并存入一个字符串切片
		var sum int
		for _, v := range data {
			val, _ := strconv.Atoi(v) //将字符串转换为int
			sum += val
		}
		fmt.Println("sum = ", sum)
		fmt.Println("data = ", data) // data 是 []string
		fmt.Println("data[0] = ", data[0])
	}
}
