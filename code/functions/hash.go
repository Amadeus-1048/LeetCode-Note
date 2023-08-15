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

// 202. 快乐数
func isHappy(n int) bool {
	m := make(map[int]bool)
	for n != 1 && !m[n] {
		n, m[n] = getSum(n), true
	}
	return n == 1
}

func getSum(n int) int {
	sum := 0
	for n > 0 {
		sum += (n % 10) * (n % 10)
		n = n / 10
	}
	return sum
}

// 1. 两数之和
func twoSum(nums []int, target int) []int {
	hashMap := make(map[int]int)
	for i, v := range nums {
		// 这样处理是为了防止出现重复的数
		if j, ok := hashMap[target-v]; ok {
			return []int{i, j}
		}
		hashMap[v] = i
	}
	return []int{}
}

// 454. 两数相加II
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	m := make(map[int]int)
	count := 0
	// 遍历A和B数组，统计两个数组元素之和、出现的次数，放到map中
	for _, v1 := range nums1 {
		for _, v2 := range nums2 {
			m[v1+v2]++
		}
	}
	// 遍历C和D数组，找到如果 0-(c+d) 在map中出现过的话，就用count把map中key对应的value也就是出现次数统计出来
	for _, v3 := range nums3 {
		for _, v4 := range nums4 {
			count += m[-v3-v4]
		}
	}
	return count
}

// 383. 赎金信
func canConstruct(ransomNote string, magazine string) bool {
	record := make([]int, 26)
	for _, v := range magazine {
		record[v-'a']++
	}
	for _, v := range ransomNote {
		record[v-'a']--
		if record[v-'a'] < 0 {
			return false
		}
	}
	return true
}
