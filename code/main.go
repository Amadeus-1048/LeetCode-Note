package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	n := 0
	_, _ = fmt.Scan(&n)
	inputs := bufio.NewScanner(os.Stdin)
	inputs.Scan()
	data := strings.Split(inputs.Text(), " ")
	a := make([]int, 0)
	max := 0
	for i := range data {
		tmp, _ := strconv.Atoi(data[i])
		if i == 0 {
			max = tmp
		} else {
			if tmp > max {
				max = tmp
			}
		}
		a = append(a, tmp)
	}
	if n == 0 {
		return
	}
	if n == 1 {
		fmt.Printf("%d ", a[0]*2)
		return
	}
	// strings.Builder的0值可以直接使用
	var builder strings.Builder

	for i := 0; i < n; i++ {
		num := 0
		tmp := a[i] * 2
		if tmp > max {
			num = tmp
		} else {
			num = max
		}
		// 向builder中写入字符/字符串
		builder.WriteString(strconv.Itoa(num))
		builder.WriteString(" ")
	}
	// String() 方法获得拼接的字符串
	fmt.Printf("%s", builder.String())
}
