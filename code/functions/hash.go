package functions

// 242.有效的字母异位词
func isAnagram(s string, t string) bool {
	record := make([]int, 26)
	for i := 0; i < len(s); i++ {
		record[s[i]-'a']++
	}
	for i := 0; i < len(t); i++ {
		record[t[i]-'a']--
	}
	for i := 0; i < 26; i++ {
		if record[i] != 0 {
			return false
		}
	}
	return true
}

// 349. 两个数组的交集
func intersection(nums1 []int, nums2 []int) []int {
	m := make(map[int]int)
	for _, v := range nums1 {
		m[v] = 1 // 注意是赋值为1，不是++，因为交集中每个数字只出现一次
	}
	res := make([]int, 0)
	for _, v := range nums2 {
		if count, ok := m[v]; ok && count > 0 {
			res = append(res, v)
			m[v]-- // 避免交集中出现重复的数字
		}
	}
	return res
}
