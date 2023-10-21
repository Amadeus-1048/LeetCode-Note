package main

import "fmt"

func main() {
	n := 0
	fmt.Scan(&n)
	a := make([]int, n+1)
	for i := 1; i <= n; i++ {
		fmt.Scan(&a[i])
	}

	if n == 1 {
		fmt.Println(1)
		return
	}

	b := make([]int, n+1)
	used := make(map[int]bool)
	// 要求：
	// ai+bi 是i的倍数
	// 1 <= bi <= 10^9
	// b中没有元素相等
	for i := n; i >= 1; i-- {
		beishu := 0
		value := 0
		if a[i] <= i {
			beishu = 1
		} else {
			beishu = a[i] / i
			if a[i]%i == 0 {
				beishu++
			}
		}

		value = beishu*i - a[i]
		//fmt.Println("i:", i)
		//fmt.Println("beishu:", beishu)
		//fmt.Println("value:", value)
		for value <= 0 {
			beishu++
			value = beishu*i - a[i]
		}
		_, ok := used[value]
		// 这个value在b中用过了
		for ok == true {
			beishu++
			value = beishu*i - a[i]
			_, ok = used[value]
		}
		used[value] = true
		b[i] = value
	}
	fmt.Printf("%d", b[1])
	for i := 2; i <= n; i++ {
		fmt.Printf(" %d", b[i])

	}
}
