package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	var s string
	inputs := bufio.NewScanner(os.Stdin)
	inputs.Scan()
	s = inputs.Text()
	res := 0
	length := len(s)
	if length == 1 {
		fmt.Println(0)
		return
	}
	locationXiao, locationMi := make([]int, 0), make([]int, 0)
	for i := 0; i < length-3; i++ {
		if s[i] == 'x' {
			if s[i+1] == 'i' && s[i+2] == 'a' && s[i+3] == 'o' {
				locationXiao = append(locationXiao, i)
			}
		}
	}
	for i := 0; i < length-1; i++ {
		if s[i] == 'm' && s[i+1] == 'i' {
			locationMi = append(locationMi, i)
		}
	}
	for i := 0; i < len(locationXiao); i++ {
		for j := 0; j < len(locationMi); j++ {
			if locationXiao[i] < locationMi[j] {
				res++
			}
		}
	}
	fmt.Println(res)
}
