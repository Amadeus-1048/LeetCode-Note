package main

import (
	"fmt"
)

// sql  查找所有连续且仅出现两次的数字

func main() {
	a, b := 0, 0
	fmt.Scan(&a, &b)
	add := 0
	res := 0
	count := 1
	for a > 0 || b > 0 || add > 0 {
		x := a % 10
		y := b % 10
		tmp := x + y + add
		if tmp == 1 { // tmp=1, x, y, add 中有一个是1
			add = 0

		} else if tmp == 2 { // tmp=2, x, y, add 中有两个是1，要进位
			add = 1
			tmp = 0
		} else if tmp == 3 { // tmp=3, x, y, add 都是1，要进位
			add = 1
			tmp = 1
		}
		res = res + tmp*count
		count *= 10
		a /= 10
		b /= 10
	}
	fmt.Println(res)
}
