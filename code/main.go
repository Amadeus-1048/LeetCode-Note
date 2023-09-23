package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	//fmt.Println("hello world, amadeus")
	scoreMap := map[string]float64{"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
	inputs := bufio.NewScanner(os.Stdin)
	for inputs.Scan() {
		var avg float64
		flag := true
		data := strings.Split(inputs.Text(), " ")
		for _, v := range data {
			score, ok := scoreMap[v]
			if ok == false {
				fmt.Println("Unknown")
				flag = false
				break
			}
			avg += score
		}
		if flag {
			avg = avg / float64(len(data))
			fmt.Printf("%.2f\n", avg) // 保留两位小数
		}
	}
}
