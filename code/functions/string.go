package functions

// 344. 反转字符串
func reverseString(s []byte) {
	left := 0
	right := len(s) - 1
	for left < right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}

// 541. 反转字符串II
func reverseStr(s string, k int) string {
	ss := []byte(s)
	length := len(ss)
	for i := 0; i < length; i += 2 * k {
		if i+k <= length {
			reverse(ss[i : i+k])
		} else {
			reverse(ss[i:length])
		}
	}
	return string(ss)
}

func reverse(s []byte) {
	left := 0
	right := len(s) - 1
	for left < right {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
}
