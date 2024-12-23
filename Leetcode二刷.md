# 总结

## 数组

- 二分法
  - **循环不变量原则**，在循环中坚持对区间的定义
- 双指针法
  - 双指针法（快慢指针法）：通过**一个快指针和一个慢指针**在一个for循环下完成两个for循环的工作
- 滑动窗口
  - 滑动窗口的精妙之处在于根据当前子序列和大小的情况，**不断调节子序列的起始位置**。从而将O(n^2^)的暴力解法降为O(n)
  - 实现滑动窗口，主要确定如下三点：
    窗口内是什么？
    如何移动窗口的起始位置？
    如何移动窗口的结束位置？
  - 只用一个for循环，那么**循环的索引**一定是表示**滑动窗口的终止位置**
- 模拟
  - **循环不变量原则**

## 链表

- 虚拟头节点
  - 链表的一大问题就是操作当前节点必须要找前一个节点才能操作。
    因为头结点没有前一个节点了，**每次对应头结点的情况都要单独处理**，所以使用虚拟头结点的技巧，就可以解决这个问题
- 双指针法

## 哈希表

- 数组作为哈希表
  - 数组就是简单的哈希表，但是数组的大小是受限的。如果题目包含小写字母，那么使用数组来做哈希最合适不过
- 双指针法

## 栈与队列

- 括号匹配问题

- 逆波兰式

- 栈实现队列、队列实现栈

- 单调队列
  
  - 单调队列里的值是单调递减(增)的。如果发现队尾元素小于要加入的元素，则将队尾元素出队，直到队尾元素大于等于新元素时，再让新元素入队，目的就是维护一个单调递减的队列。

- 单调栈
  
  - 遍历数组，数组中的值为待入栈元素，待入栈元素入栈时会先跟栈顶元素进行对比，如果小于等于该值则入栈，如果大于则将栈顶元素出栈，新的元素入栈。

## 树

- 遍历方式
  - 递归三要素，以及前中后序的递归写法。
  - 层次遍历
- 树的属性
  - 求普通二叉树的属性，一般是后序，一般要通过递归函数的返回值做计算
  - 求二叉搜索树的属性，一定是中序，不然浪费了有序性
- 树的构造
  - 涉及到二叉树的构造，无论普通二叉树还是二叉搜索树一定前序，都是先构造中节点
- 二叉搜索树
  - 二叉搜索树的中序遍历是有序数组

## 回溯

- 回溯模板
  
  - ```c++
    void backtracking(参数) {
        if (终止条件) {
            存放结果;
            return;
        }
    
        for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
            处理节点;
            backtracking(路径，选择列表); // 递归
            回溯，撤销处理结果
        }
    }
    ```

- **回溯是递归的副产品，只要有递归就会有回溯**
  
  - 所以回溯法也经常和二叉树遍历，深度优先搜索混在一起，因为这两种方式都是用了递归。
  - **回溯法就是暴力搜索**，并不是什么高效的算法，最多再剪枝一下

- 什么时候需要返回值
  
  - 如果目的是找到一个符合的条件（就在树的叶子节点上）立刻就返回，相当于找从根节点到叶子节点一条唯一路径，这个时候才需要返回值。

- 组合问题
  
  - 是收集叶子节点的结果
  - 需要使用startIndex
  - **for循环横向遍历，递归纵向遍历，回溯不断调整结果集**
  - 在for循环上做**剪枝**操作是回溯法剪枝的常见套路
  - “树枝去重”和“树层去重”

- 排列问题
  
  - 不需要使用startIndex
  - 排列是有序的，也就是说 [1,2] 和 [2,1] 是两个集合，这和之前分析的子集以及组合所不同的地方

- 切割问题
  
  - 用求解**组合问题**的思路来解决 切割问题

- 子集问题
  
  - 在树形结构中**子集问题**是要**收集所有节点**的结果，所以每次递归后直接添加进结果。而组合问题是收集叶子节点的结果

- 棋盘问题
  
  - 棋盘的**宽度**就是**for循环的长度**，棋盘的**高度**就是递归的**深度**，这样就可以套进回溯法的模板里了

## 贪心

- **找出局部最优并可以推出全局最优，就是贪心**
- 贪心无套路，也没有框架

## 动态规划

- 动规五部曲分别为：
  
  1. 确定dp数组（dp table）以及下标的含义
  2. 确定递推公式
  3. dp数组如何初始化
  4. 确定遍历顺序
  5. 举例推导dp数组

- 背包问题
  
  - ![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210117171307407-20230310133624872.png)
  - 背包递推公式
    - 能否能装满背包（或者最多装多少）：dp[j] = max(dp[j], dp[j - nums[i]] + nums[i])
    - 装满背包有几种方法：dp[j] += dp[j - nums[i]]
    - 背包装满最大价值：dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    - 装满背包所有物品的最小个数：dp[j] = min(dp[j], dp[j - coins[i]] + 1)
  - 遍历顺序
    - 01背包：**二维**dp数组01背包先遍历物品还是先遍历背包都是可以的，且第二层for循环是从小到大遍历
    - 01背包：**一维**dp数组01背包只能**先遍历物品再遍历背包容量**，且第二层for循环是**从大到小**遍历
    - 纯完全背包的遍历顺序：一维dp数组实现，先遍历物品还是先遍历背包都是可以的，且第二层for循环是从小到大遍历
    - 完全背包（题目变化）：**求组合数就是外层for循环遍历物品，内层for遍历背包**
    - 完全背包（题目变化）：**求排列数就是外层for遍历背包，内层for循环遍历物品**

- 打劫问题

- 子序列问题

# 数组

## [704. 二分查找](https://leetcode.cn/problems/binary-search/)

给定一个 `n` 个元素有序的（升序）整型数组 `nums` 和一个目标值 `target`  ，写一个函数搜索 `nums` 中的 `target`，如果目标值存在返回下标，否则返回 `-1`。

答案

```go
func search(nums []int, target int) int {
    left := 0
    right := len(nums)    // 如果是左闭右闭，那么 right = len -1
    for left < right {
        mid := left + (right-left)/2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid // 因为是左闭右开区间，所以这里不能是high = mid - 1，mid-1有可能是答案
        }
    }
    return -1
}
```

分析

```go
数组为有序数组，同时题目还强调数组中无重复元素，则可以用二分法

第一种二分法：[left, right]
for left <= right，因为left在区间成立时可能等于right
取左半边时，right = mid-1， 因为已经确定了nums[mid]不会是target

第二种二分法：[left, right)
for left < right，因为left在区间成立时不可能等于right
取左半边时，right = mid， 因为已经确定了nums[mid]不会是target，又因为右区间是开区间，所以不会取到mid
```

## [27. 移除元素](https://leetcode.cn/problems/remove-element/)

给你一个数组 `nums` 和一个值 `val`，你需要 **[原地](https://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)** 移除所有数值等于 `val` 的元素。元素的顺序可能发生改变。然后返回 `nums` 中与 `val` 不同的元素的数量。

假设 `nums` 中不等于 `val` 的元素数量为 `k`，要通过此题，您需要执行以下操作：

- 更改 `nums` 数组，使 `nums` 的前 `k` 个元素包含不等于 `val` 的元素。`nums` 的其余元素和 `nums` 的大小并不重要。
- 返回 `k`。

答案

```go
func removeElement(nums []int, val int) int {
    length := len(nums)
    slow, fast := 0, 0
    for fast < length {
        if nums[fast] != val {
            nums[slow] = nums[fast]
            slow++
            fast++
        } else {
            fast++
        }
    }
    return slow
}
```

分析

```go
双指针法（快慢指针法）： 通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。

定义快慢指针：
    快指针：寻找新数组的元素 ，新数组就是不含有目标元素的数组
    慢指针：指向更新 新数组下标的位置
```

## [977. 有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)

给你一个按 **非递减顺序** 排序的整数数组 `nums`，返回 **每个数字的平方** 组成的新数组，要求也按 **非递减顺序** 排序。

答案

```go
func sortedSquares(nums []int) []int {
    n := len(nums)
    i, j, k := 0, n-1, n-1    // i、j分别指向原数组的起、止位置，k指向新数组的最后
    ans := make([]int, n)
    for k >= 0 {
        left, right := nums[i]*nums[i], nums[j]*nums[j]
        if left < right {
            ans[k] = right
            j--
        } else {
            ans[k] = left
            i++
        }
        k--
    }
    return ans
}
```

分析

```go
数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。
此时可以考虑双指针法了，i指向起始位置，j指向终止位置
定义一个新数组result，和A数组一样的大小，让k指向result数组终止位置
```

## [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

给定一个含有 `n` 个正整数的数组和一个正整数 `target` **。**

找出该数组中满足其总和大于等于 `target` 的长度最小的 **子数组**

 `[numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度。如果不存在符合条件的子数组，返回 `0` 。

答案

```go
func minSubArrayLen(target int, nums []int) int {
    i, sum := 0, 0  // 子序列的起始位置
    length := len(nums)
    res := length + 1
    for j := 0; j < length; j++ {    // 调节子序列的终止位置
        sum += nums[j]
        for sum >= target {
            res = min(res, j-i+1)    // 比较并更新答案
            sum -= nums[i]    // 调节子序列的起始位置
            i++
        }
    }
    if res == length+1 {
        return 0
    }
    return res
}

func min(a, b int) int {
  if a < b {
    return a
  }
  return b
}
```

分析

```go
滑动窗口 : 不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果

在本题中实现滑动窗口，主要确定如下三点：
窗口内是什么？
如何移动窗口的起始位置？
如何移动窗口的结束位置？

只用一个for循环，那么这个循环的索引，一定是表示滑动窗口的终止位置
```

## [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

给定一个字符串 `s` ，请你找出其中不含有重复字符的最长子串的长度。

答案

```go
func lengthOfLongestSubstring(s string) int {
    ans, left, right := 1, 0, 1 // 滑动窗口
    hash := make(map[byte]bool)
    n := len(s)
    if n <= 1 {
        return n
    }
    hash[s[0]] = true
    for right < n {    // 调节滑动窗口的终止位置
        if !hash[s[right]] { // 没有遇到重复的字符，则s[right]存入map,计算长度，right推进
            hash[s[right]] = true
            ans = max(ans, right-left+1)
            right++
        } else { // 遇到重复字符，需要在map中去掉s[left]，left推进
            for hash[s[right]] { // 不断推进left，直到遇到和s[right]相同的字符
                delete(hash, s[left])
                left++
            }
            hash[s[right]] = true // 因为上面把s[right]给delete了，需要重新放到map里
            right++
        }
    }
    return ans
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
滑动窗口 : 不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果

在本题中实现滑动窗口，主要确定如下三点：
窗口内是什么？
如何移动窗口的起始位置？
如何移动窗口的结束位置？

只用一个for循环，那么这个循环的索引，一定是表示滑动窗口的终止位置
```

## [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

答案

```go
func searchTwisted(nums []int, target int) int {
    n := len(nums)
    if n == 1 {
        if nums[0] == target {
            return 0
        } else {
            return -1
        }
    }
    left, right, mid := 0, n-1, 0
    for left <= right {
        mid = (left + right) / 2 // 二分法
        if nums[mid] == target { // 判断是否找到target
            return mid
        }
        if nums[0] <= nums[mid] { // 0~mid是有序的    这里必须加=号
            if nums[0] <= target && target < nums[mid] { // target在有序的0~mid范围内，进行查找
                right = mid - 1
            } else { // target不在0~mid范围内，在无序的mid+1~n-1范围内重新查找
                left = mid + 1
            }
        } else { // 0~mid是无序的，mid~n是有序的
            if nums[mid] < target && target <= nums[n-1] { // target在有序的mid~n-1范围内，进行查找
                left = mid + 1
            } else { // target不在mid~n-1范围内，在无序的0~mid-1范围内重新查找
                right = mid - 1
            }
        }
    }
    return -1
}
```

分析

```go
将数组一分为二，其中一定有一个是有序的，另一个可能是有序，也可能是部分有序。
此时有序部分用二分法查找。无序部分再一分为二，其中一个一定有序，另一个可能有序，可能无序。就这样循环.
```

## [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0` ，应忽略。`nums2` 的长度为 `n` 。

答案

```go
func merge(nums1 []int, m int, nums2 []int, n int) {
    i, j := m-1, n-1 // nums1 的初始长度为 m + n
    tail := m + n - 1    // 从后往前放置元素
    for i >= 0 || j >= 0 {    // 只要有一个大于等于0，就表示还没合并完
        if i < 0 {    // nums1全部用完，直接用nums2的
            nums1[tail] = nums2[j]
            j--
        } else if j < 0 {    // nums2全部用完，直接用nums1的
            nums1[tail] = nums1[i]
            i--
        } else if nums1[i] <= nums2[j] {
            nums1[tail] = nums2[j]
            j--
        } else {
            nums1[tail] = nums1[i]
            i--
        }
        tail--
    }
}
```

分析

```go

```

## [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

答案

```go
func trap(height []int) int {
    left, right := 0, len(height)-1
    maxLeft, maxRight := 0, 0
    ans := 0
    // 双指针法，左右指针代表着要处理的雨水位置，最后一定会汇合
    for left <= right { // 注意，这里可能 left==right
        // 对于位置left而言，它左边最大值一定是maxLeft，右边最大值“大于等于”maxRight
        if maxLeft < maxRight { // 如果maxLeft < maxRight，那么无论右边将来会不会出现更大的maxRight，都不影响这个结果
            ans += max(0, maxLeft-height[left])
            maxLeft = max(maxLeft, height[left])
            left++
        } else { // 反之，去处理right下标
            ans += max(0, maxRight-height[right])
            maxRight = max(maxRight, height[right])
            right--
        }
    }
    return ans
}
```

分析

```go
每个位置能储存的雨水量为左边最高柱子的高度 和 右边最高柱子的高度中较小的那个 减去该位置的柱子高度
即: drop[i] = min(maxLeft, maxRight) - height[i]
```

## [69. x 的平方根](https://leetcode.cn/problems/sqrtx/)

给你一个非负整数 `x` ，计算并返回 `x` 的 **算术平方根** 。

由于返回类型是整数，结果只保留 **整数部分** ，小数部分将被 **舍去 。**

答案

```go
func mySqrt(x int) int {
    res := 0 // x 平方根的整数部分 res 是满足 k^2 ≤ x 的最大 k 值
    mid := 0
    left, right := 0, x
    if x <= 1 {
        return x
    } else {
        for left <= right { // 进行二分查找
            mid = left + (right-left)/2
            if mid*mid <= x { // 比较中间元素的平方与 x 的大小关系
                res = mid
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return res
}
```

分析

```go

```

## [283. 移动零](https://leetcode.cn/problems/move-zeroes/)

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

答案

```go
func moveZeroes(nums []int) {
    // 左指针指向当前已经处理好的序列的尾部+1
    // 右指针指向待处理序列的头部（也就是每一轮遍历到哪里）
    left, right, n := 0, 0, len(nums)
    for right < n {
        if nums[right] != 0 {    // 每次右指针指向非零数，则将左右指针对应的数交换
            nums[left], nums[right] = nums[right], nums[left]
            left++
        }
        right++
    }
}
```

分析

```go
白话：每轮遍历右指针都向后移动一位，如果遇到了非0数，则将其放到前面的左指针处

左指针指向当前已经处理好的序列的尾部+1
右指针指向待处理序列的头部
右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移
```

## [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

答案

```go
func maxArea(height []int) int {
    left, right := 0, len(height)-1 // 左右指针分别指向数组的左右两端
    ans := 0
    for left < right {
        // 容纳的水量 = 两个指针指向的数字中较小值 ∗ 指针之间的距离
        area := min(height[left], height[right]) * (right - left)
        ans = max(ans, area)
        // 指针之间的距离一定会开始逐渐减小，要想找到更大的答案，必须要找到更大的较小值，所以必须移动值较小的指针
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    return ans
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
容纳的水量 = 两个指针指向的数字中较小值 ∗ 指针之间的距离

如果我们移动数字较大的那个指针，那么前者「两个指针指向的数字中较小值」不会增加，后者「指针之间的距离」会减小，那么这个乘积会减小。
因此，我们移动数字较大的那个指针是不合理的。因此，我们移动 数字较小的那个指针。

每次将 对应的数字较小的那个指针 往 另一个指针 的方向移动一个位置，就表示我们认为 这个指针不可能再作为容器的边界了
```

## [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

答案

```go
func findAnagrams(s, p string) (ans []int) {
    sLen, pLen := len(s), len(p)
    if sLen < pLen {
        return
    }

    var sCount, pCount [26]int    
      // 在字符串 s 中构造一个长度为与字符串 p 的长度相同的滑动窗口
    for i, _ := range p {
        sCount[s[i]-'a']++
        pCount[p[i]-'a']++
    }
    if sCount == pCount {    // 窗口中每种字母的数量与字符串 p 中每种字母的数量相同
        ans = append(ans, 0)
    }

    for i, ch := range s[:sLen-pLen] {    // 虽然是从0开始遍历，但0其实已经处理过了，所以实际是从1开始
        sCount[ch-'a']--    // 第i位字母的数量减少1，ch=s[i]
        sCount[s[i+pLen]-'a']++    // 第si+pLen位字母的数量增加1
        if sCount == pCount {
            ans = append(ans, i+1)
        }
    }
    return
}
```

分析

```go

```

## [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**注意：**

- 对于 `t` 中重复字符，我们寻找的子字符串中该字符数量必须不少于 `t` 中该字符数量。
- 如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

答案

```go
func minWindow(s string, t string) string {
    var res string
    count := math.MaxInt32
    hashMap := make(map[byte]int)
    l, r := 0, 0
    for i := 0; i < len(t); i++ {
        hashMap[t[i]]++    // 哈希表记录t数组中字符出现的次数
    }
    for r < len(s) {
        hashMap[s[r]]--
        for check(hashMap) {
            if r-l+1 < count {
                count = r - l + 1
                res = s[l : r+1]
            }
            hashMap[s[l]]++    // l向右移动，字符出现的次数要恢复
            l++
        }
        r++
    }
    return res
}

// true 表示窗口内已完全覆盖t字符串
func check(hashMap map[byte]int) bool {
    for _, v := range hashMap {
        if v > 0 {
            return false
        }
    }
    return true
}
```

分析

```go
只要hashMap有值 >0，说明窗口内尚未覆盖t字符串所有字母，窗口需要扩大，r指针需要右移

只要hashMap的值都 <=0，说明窗口内已完全覆盖t字符串，窗口内的字符串可能就是答案，但窗口存在可缩小的可能性，开始尝试l右移缩小窗口
```

## [189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

答案

```go
func reverse(a []int) {
    for i, j := 0, len(a)-1; i < j; i,j=i+1,j-1 {
        a[i], a[j] = a[j], a[i]
    }
}

func rotate(nums []int, k int) {
    k %= len(nums)    // k 可能 大于 len(nums)
    reverse(nums)
    reverse(nums[:k])
    reverse(nums[k:])
}
```

分析

```go
先将所有元素翻转，这样尾部的 k mod n 个元素就被移至数组头部，
然后我们再翻转 [0,k mod n−1] 区间的元素  和  [k mod n,n−1] 区间的元素即能得到最后的答案
```

## [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在  **32 位** 整数范围内。

请不要使用除法，且在 `O(n)` 时间复杂度内完成此题。

答案

```go
func productExceptSelf(nums []int) []int {
    length := len(nums)
    answer := make([]int, length)

    // answer[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
    answer[0] = 1
    for i := 1; i < length; i++ {
        answer[i] = nums[i-1] * answer[i-1]
    }

    // R 为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 R = 1
    R := 1
    for i := length - 1; i >= 0; i-- {
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        answer[i] = answer[i] * R
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i]
    }
    return answer
}
```

分析

```go

```

## [41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/)

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

答案

```go
func firstMissingPositive(nums []int) int {
    n := len(nums)
    for i := 0; i < n; i++ {
          // 对于遍历到的数 x=nums[i]，如果 x∈[1,N]，我们就知道 x 应当出现在数组中的 x−1 的位置
        for nums[i] >= 1 && nums[i] <= n && nums[nums[i]-1] != nums[i] {
            nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        }
    }
    for i := 0; i < n; i++ {
        if nums[i] != i + 1 {
            return i + 1
        }
    }
    return n + 1
}
```

分析

```go
让数值在1 ~~ len(nums)区间内的数放在nums[i] - 1位置上，最后从左向右找出第一个nums[i] != i+1的数

将给定的数组「恢复」成下面的形式：
如果数组中包含 x∈[1,N]，那么恢复后，数组的第 x−1 个元素为 x。

在恢复后，数组应当有 [1, 2, ..., N] 的形式，但其中有若干个位置上的数是错误的，每一个错误的位置就代表了一个缺失的正数。
以题目中的示例二 [3, 4, -1, 1] 为例，恢复后的数组应当为 [1, -1, 3, 4]，我们就可以知道缺失的数为 2。
```

# 链表

## [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

答案

```go
func removeElements(head *ListNode, val int) *ListNode {
    dummy := &ListNode{}
    dummy.Next = head
    cur := dummy
    for cur.Next!=nil {
        if cur.Next.Val == val {
            cur.Next = cur.Next.Next    // 注意：这里不需要将cur移动为下一个节点
        } else {
            cur = cur.Next
        }
    }
    return dummy.Next
}
```

分析

```go
在单链表中移除头结点 和 移除其他节点的操作方式是不一样的
设置一个虚拟头结点，这样原链表的所有节点就都可以按照统一的方式进行移除了
```

## 707.设计链表

答案

```go
type SingleNode struct {
    Val  int         // 节点的值
    Next *SingleNode // 下一个节点的指针
}

type MyLinkedList struct {
    dummyHead *SingleNode // 虚拟头节点
    Size      int         // 链表大小
}

func Constructor() MyLinkedList {
    newNode := &SingleNode{ // 创建新节点
        Next: nil,
    }
    return MyLinkedList{ // 返回链表
        dummyHead: newNode,
        Size:      0,
    }

}

func (this *MyLinkedList) Get(index int) int {
    if this == nil || index < 0 || index >= this.Size { // 如果索引无效则返回-1
        return -1
    }
    cur := this.dummyHead.Next   // 设置当前节点为真实头节点
    for i := 0; i < index; i++ { // 遍历到索引所在的节点
        cur = cur.Next
    }
    return cur.Val // 返回节点值
}

func (this *MyLinkedList) AddAtHead(val int) {
    newNode := &SingleNode{Val: val}   // 创建新节点
    newNode.Next = this.dummyHead.Next // 新节点指向当前头节点
    this.dummyHead.Next = newNode      // 新节点变为头节点
    this.Size++                        // 链表大小增加1
}

func (this *MyLinkedList) AddAtTail(val int) {
    newNode := &SingleNode{Val: val} // 创建新节点
    cur := this.dummyHead            // 设置当前节点为虚拟头节点
    for cur.Next != nil {            // 遍历到最后一个节点
        cur = cur.Next
    }
    cur.Next = newNode // 在尾部添加新节点
    this.Size++        // 链表大小增加1
}

func (this *MyLinkedList) AddAtIndex(index int, val int) {
    if index < 0 { // 如果索引小于0，设置为0
        index = 0
    } else if index > this.Size { // 如果索引大于链表长度，直接返回
        return
    }

    newNode := &SingleNode{Val: val} // 创建新节点
    cur := this.dummyHead            // 设置当前节点为虚拟头节点
    for i := 0; i < index; i++ {     // 遍历到指定索引的前一个节点
        cur = cur.Next
    }
    newNode.Next = cur.Next // 新节点指向原索引节点
    cur.Next = newNode      // 原索引的前一个节点指向新节点
    this.Size++             // 链表大小增加1
}

func (this *MyLinkedList) DeleteAtIndex(index int) {
    if index < 0 || index >= this.Size { // 如果索引无效则直接返回
        return
    }
    cur := this.dummyHead        // 设置当前节点为虚拟头节点
    for i := 0; i < index; i++ { // 遍历到要删除节点的前一个节点
        cur = cur.Next
    }
    if cur.Next != nil {
        cur.Next = cur.Next.Next // 当前节点直接指向下下个节点，即删除了下一个节点
    }
    this.Size-- // 注意删除节点后应将链表大小减一
}
```

分析

```go

```

## [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

答案

```go
func reverseList(head *ListNode) *ListNode {
    cur := head
    var pre *ListNode    // 不能用pre := &ListNode{}，因为输出结果会在最后多一个0
    for cur != nil {
        next := cur.Next
        cur.Next = pre
        pre = cur
        cur = next
    }
    return pre    // 是返回pre，不是cur，因为最后cur是nil。想不起来的话可以考虑只有一个节点的情况。
}
```

分析

```go
var pre *ListNode 和 pre := &ListNode{} 是不同的

p1 := &ListNode{}
fmt.Println("p1:", p1==nil)        // p1: false  p1的值为：{0, <nil>}，这就是上面为什么会多一个0的原因
var p2 *ListNode
fmt.Println("p2:", p2==nil)        // p2: true
```

## [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/)

给你单链表的头指针 `head` 和两个整数 `left` 和 `right` ，其中 `left <= right` 。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 **反转后的链表** 。

答案

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
    dummy := &ListNode{Next: head}
    pre := dummy
    for i:=0; i<left-1; i++ {
        pre = pre.Next
    }
    cur := pre.Next
    // 在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置
    // cur：指向待反转区域的第一个节点 left  cur的值不变，但是位置会往后移动
    // next：永远指向 cur 的下一个节点，循环过程中，cur 变化以后 next 的值和位置都会变化
    // pre：永远指向待反转区域的第一个节点 left 的前一个节点，在循环过程中值和位置都不变

    for i:=left; i<right; i++ {
        next := cur.Next
        cur.Next = next.Next
        next.Next = pre.Next    // 不能是 cur 因为必须是在反转部分的起始位置，即pre的下一个，cur会慢慢往后移动的
        pre.Next = next
    }
    return dummy.Next
}
------------------------------------------------------------------------
// 下面这种简单一点
func reverseBetween(head *ListNode, left, right int) *ListNode {
    dummy := &ListNode{}
    dummy.Next = head
    pre := dummy
    // 从虚拟头节点走 left - 1 步，来到 left 节点的前一个节点
    for i := 0; i < left-1; i++ {
        pre = pre.Next
    }
    // 从 pre 再走 right - left + 1 步，来到 right 节点
    rightNode := pre
    for i := left; i < right+1; i++ {
        rightNode = rightNode.Next
    }
    // 切断出一个子链表（截取链表）
    leftNode := pre.Next
    cur := rightNode.Next
    // 切断链接
    pre.Next = nil
    rightNode.Next = nil
    // 同第 206 题，反转链表的子区间
    reverseLinkedList(leftNode)
    // 接回到原来的链表中
    pre.Next = rightNode
    leftNode.Next = cur
    return dummy.Next
}

func reverseLinkedList(head *ListNode) {
    var pre *ListNode
    cur := head
    for cur != nil {
        next := cur.Next
        cur.Next = pre
        pre = cur
        cur = next
    }
}
```

分析

```go
画图！
pre  的值和位置都不变
cur  的值不变，但是位置会往后移动
next 的值和位置都会变化
```

## [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

答案

```go
func swapPairs(head *ListNode) *ListNode {
    dummy := &ListNode{}
    dummy.Next = head
    cur := dummy    // 初始时，cur指向虚拟头结点
    for head != nil && head.Next != nil {
        cur.Next = head.Next
        next := head.Next.Next
        head.Next.Next = head
        head.Next = next
        cur = head
        head = next
    }
    return dummy.Next
}
```

分析

初始时，cur指向虚拟头结点，然后进行如下三步：

<img src="https://code-thinking.cdn.bcebos.com/pics/24.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B91.png" alt="24.两两交换链表中的节点1" style="zoom:50%;" />

操作之后，链表如下：

<img src="https://code-thinking.cdn.bcebos.com/pics/24.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B92.png" alt="24.两两交换链表中的节点2" style="zoom:50%;" />

```go
还是要画一下图，记住三个步骤

head.Next.Next = head    对结点指向哪里进行修改
pre = head                用变量pre表示head结点（即pre和head同时表示一个结点）
```

## [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

答案

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{}
    dummy.Next = head
    fast, slow := dummy, dummy
    for i:=0; i<=n; i++ {
        fast = fast.Next
    }
    for fast != nil {
        slow = slow.Next
        fast = fast.Next
    }
    slow.Next = slow.Next.Next
    return dummy.Next
}
```

分析

```go
双指针的经典应用
定义fast指针和slow指针，初始值为虚拟头结点
fast首先走n + 1步 ，为什么是n+1呢，因为只有这样同时移动的时候slow才能指向删除节点的上一个节点（方便做删除操作）
fast和slow同时移动，直到fast指向末尾(NULL)
删除slow指向的下一个节点
```

## [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

答案

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    pa, pb := headA, headB
    for pa != pb {
        if pa == nil {
            pa = headB
        } else {
            pa = pa.Next
        }
        if pb == nil {
            pb = headA
        } else {
            pb = pb.Next
        }
    }
    return pa
}
```

分析

```go
双指针
如果有相交，那么相交时两个指针走的步数相等，重合在相交点
如果没有，两个指针会走完两轮，同时指向nil，此时相等，退出循环
```

## [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

答案

```go
func reverseKGroup(head *ListNode, k int) *ListNode {
    cur := head
    for i := 0; i < k; i++ {
        if cur == nil {
            return head
        }
        cur = cur.Next
    }
    newHead := reverse(head, cur)        // 从 head 开始进行反转长度为 k 的链表
    head.Next = reverseKGroup(cur, k)    // 递归处理下一个长度为 k 的链表
    return newHead
}

func reverse(head, tail *ListNode) *ListNode {
    var pre *ListNode
    cur := head
    for cur != tail {        // 从 head 到 tail 的前面一个结点进行遍历
        next := cur.Next
        cur.Next = pre
        pre = cur
        cur = next
    }
    return pre
}
```

分析

https://leetcode.cn/problems/reverse-nodes-in-k-group/solution/jian-dan-yi-dong-go-by-dfzhou6-4cha/

```go
采用递归，这样比较易于理解
```

## [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 `true` 。 否则，返回 `false` 。

答案

```go
// 定义快慢指针 slow、fast，初始指向 head。
// 快指针每次走两步，慢指针每次走一步，不断循环。
// 当相遇时，说明链表存在环。如果循环结束依然没有相遇，说明链表不存在环。
func hasCycle(head *ListNode) bool {
    slow, fast := head, head
    for fast!=nil && fast.Next!=nil {
        slow = slow.Next
        fast = fast.Next.Next
        if slow == fast {
            return true
        }
    }
    return false
}
```

分析

```go
for循环的条件为：for fast!=nil && fast.Next!=nil 
原因是如果fast是链表尾结点，即fast.Next=nil，此时fast.Next.Next是非法的，会报错
因此要判断fast和fast.Next是否为nil
如果fast.Next不是nil，那么fast.Next.Next是合法的（虽然可能等于nil）
```

## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

答案

```go
func detectCycle(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next   // 任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍
        if slow == fast {   // 找到重合的节点，说明在环中
        // 当 slow 与 fast 相遇时，head指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇
            for slow != head {
                slow = slow.Next
                head = head.Next
            }
            return head
        }
    }
    return nil
}
```

分析

<img src="https://assets.leetcode-cn.com/solution-static/142/142_fig1.png" alt="fig1" style="zoom: 25%;" />

```go
设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与 fast 相遇。
此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc。

根据题意，任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍。因此，我们有 a+(n+1)b+nc = 2(a+b)  ⟹  a = c+(n−1)(b+c)

有了 a=c+(n-1)(b+c) 的等量关系，我们会发现：从相遇点到入环点的距离加上 n-1 圈的环长，恰好等于从链表头部到入环点的距离。

因此，当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇。
```

## [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

答案

```go
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1
            list1 = list1.Next
        } else {
            cur.Next = list2
            list2 = list2.Next
        }
        cur = cur.Next
    }
    if list1 == nil {
        cur.Next = list2
    } else {
        cur.Next = list1
    }
    return dummy.Next
}
```

分析

```go

```

## [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

答案

```go
func mergeKLists(lists []*ListNode) *ListNode {
    n := len(lists)
    if n == 0 {
        return nil
    }
    if n == 1 { // 返回结果
        return lists[0]
    }
    if n % 2 == 1 { // K为奇数，那么先合并最后两个列表，将其变为偶数长度的列表
        lists[n-2] = mergeTwoLists(lists[n-2], lists[n-1])
        lists, n = lists[:n-1], n-1
    }
    mid := n / 2
    for i:=0; i<mid; i++{   // 后半部分合并到前半部分。
        lists[i] = mergeTwoLists(lists[i], lists[i+mid])
    }
    return mergeKLists(lists[:mid])
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummy := &ListNode{}
    cur := dummy
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1
            list1 = list1.Next
        } else {
            cur.Next = list2
            list2 = list2.Next
        }
        cur = cur.Next
    }
    if list1 == nil {
        cur.Next = list2
    } else {
        cur.Next = list1
    }
    return dummy.Next
}
```

分析

```go
归并法比暴力法要快非常多
暴力法：不断的把短链表合并到唯一的长链表中
归并法：将K个有序链表转换为多个合并两个有序链表的问题
```

## [143. 重排链表](https://leetcode.cn/problems/reorder-list/)

给定一个单链表 `L` 的头节点 `head` ，单链表 `L` 表示为：

L0 → L1 → … → Ln - 1 → Ln

请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

答案

```go
func reorderList(head *ListNode)  {
    if head == nil || head.Next == nil {
        return
    }
    slow, fast := head, head

    // 先通过快慢指针找到链表中点
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }

    // 将链表划分为左右两部分
    cur := slow.Next
    slow.Next = nil
    var pre *ListNode

    // 反转右半部分的链表
    for cur != nil {
        next := cur.Next
        cur.Next = pre
        pre = cur
    cur = next
    }

    // 将左右两个链接依次连接
    cur = head
    for pre != nil {
        next := pre.Next
        pre.Next = cur.Next
        cur.Next = pre
        cur = pre.Next
    pre = next
    }
}
```

分析

```go

```

## [82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)

给定一个已排序的链表的头 `head` ， *删除原始链表中所有重复数字的节点，只留下不同的数字* 。返回 *已排序的链表* 。

答案

```go
func deleteDuplicates(head *ListNode) *ListNode {
    if head == nil {
        return head
    }
    dummy := &ListNode{Next: head}
    pre := dummy
    cur := head
    for cur != nil && cur.Next != nil {
        next := cur.Next
        if cur.Val == next.Val {
            // 避免1 1 1 2 3 的情况， for循环可以去掉所有的1  
            for next.Next != nil && next.Val == next.Next.Val {
                next = next.Next
            }
            pre.Next = next.Next
            cur = next.Next
        } else {
            pre = cur
            cur = cur.Next
        }
    }
    return dummy.Next
}
```

分析

```go

```

## [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

答案

```go
func reverseList(head *ListNode) *ListNode {
    var prev, cur *ListNode = nil, head
    for cur != nil {
        nextTmp := cur.Next
        cur.Next = prev
        prev = cur
        cur = nextTmp
    }
    return prev
}

func endOfFirstHalf(head *ListNode) *ListNode {
    fast := head
    slow := head
    for fast.Next != nil && fast.Next.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
    }
    return slow
}

func isPalindrome(head *ListNode) bool {
    if head == nil {
        return true
    }

    // 找到前半部分链表的尾节点并反转后半部分链表
    firstHalfEnd := endOfFirstHalf(head)
    secondHalfStart := reverseList(firstHalfEnd.Next)

    // 判断是否回文
    p1 := head
    p2 := secondHalfStart
    result := true
    for result && p2 != nil {
        if p1.Val != p2.Val {
            result = false
        }
        p1 = p1.Next
        p2 = p2.Next
    }

    // 还原链表并返回结果
    firstHalfEnd.Next = reverseList(secondHalfStart)
    return result
}
```

分析

```go
找到前半部分链表的尾节点。
反转后半部分链表。
判断是否回文。
恢复链表。
返回结果。
```

## [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

答案

```go
func addTwoNumbers(l1, l2 *ListNode)  *ListNode {
    var tail *ListNode
    carry := 0
      var head *ListNode
    for l1 != nil || l2 != nil {
        n1, n2 := 0, 0
        if l1 != nil {
            n1 = l1.Val
            l1 = l1.Next
        }
        if l2 != nil {
            n2 = l2.Val
            l2 = l2.Next
        }
        sum := n1 + n2 + carry
        sum, carry = sum%10, sum/10
        if head == nil {
            head = &ListNode{Val: sum}
            tail = head
        } else {
            tail.Next = &ListNode{Val: sum}
            tail = tail.Next
        }
    }
    if carry > 0 {
        tail.Next = &ListNode{Val: carry}
    }
    return head
}
```

分析

```go
同时遍历两个链表，逐位计算它们的和，并与当前位置的进位值相加s
```

## [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)

给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 [深拷贝](https://baike.baidu.com/item/%E6%B7%B1%E6%8B%B7%E8%B4%9D/22785317?fr=aladdin)**。 深拷贝应该正好由 `n` 个 全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点** 。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为  `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

答案

```go
type Node struct {
    Val    int
    Next   *Node
    Random *Node
}

func copyRandomList(head *Node) *Node {
    cachedNode := map[*Node]*Node{}    // 哈希表记录每一个节点对应新节点的创建情况
    var deepCopy func(node *Node) *Node
    deepCopy = func(node *Node) *Node {
        if node == nil {
            return nil
        }
        if n, has := cachedNode[node]; has {    // 如果已经创建过，直接从哈希表中取出并返回
            return n
        }
        newNode := &Node{Val: node.Val}
        cachedNode[node] = newNode
        newNode.Next = deepCopy(node.Next)    // 递归地创建「当前节点的后继节点」
        newNode.Random = deepCopy(node.Random)    // 递归地创建「当前节点的随机指针指向的节点」
        return newNode
    }
    return deepCopy(head)
}
```

分析

```go

```

# 哈希表

## [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

给定两个字符串 `s` 和 `t` ，编写一个函数来判断 `t` 是否是 `s` 的字母异位词。

注意：若 `s` 和 `t` 中每个字符出现的次数都相同，则称 `s` 和 `t` 互为字母异位词。

答案

```go
func isAnagram(s string, t string) bool {
    record := make([]int, 26)
    for i:=0; i<len(s); i++ {
        record[s[i]-'a']++
    }
    for i:=0; i<len(t); i++ {
        record[t[i]-'a']--
    }
    for i:=0; i<26; i++ {
        if record[i] != 0 {
            return false
        }
    }
    return true
}
```

分析

```go
把字符映射到数组也就是哈希表的索引下标上
因为字符a到字符z的ASCII是26个连续的数值
所以字符a映射为下标0，相应的字符z映射为下标25
```

## [349. 两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

给定两个数组 `nums1` 和 `nums2` ，返回 它们的交集。输出结果中的每个元素一定是 **唯一** 的。我们可以 **不考虑输出结果的顺序** 。

答案

```go
func intersection(nums1 []int, nums2 []int) []int {
    m := make(map[int]int)
    for _, v := range nums1 {
        m[v] = 1    // 注意是赋值为1，不是++，因为交集中每个数字只出现一次
    }
    res := make([]int, 0)
    for _, v := range nums2 {
        if count, ok := m[v]; ok && count>0 {    // 必须判断count>0
            res = append(res, v)
            m[v]--    // 避免交集中出现重复的数字
        }
    }
    return res
}
```

分析

```go

```

## [202. 快乐数](https://leetcode.cn/problems/happy-number/)

编写一个算法来判断一个数 `n` 是不是快乐数。

**「快乐数」** 定义为：

- 对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
- 然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。
- 如果这个过程 **结果为** 1，那么这个数就是快乐数。

如果 `n` 是 *快乐数* 就返回 `true` ；不是，则返回 `false` 。

答案

```go
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
```

分析

```go

```

## [1. 两数之和](https://leetcode.cn/problems/two-sum/)

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`*  的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

答案

```go
func twoSum(nums []int, target int) []int {
    hashMap := make(map[int]int)
    for i, v := range nums {
    // 这样处理是为了防止出现重复的数
        if j, ok := hashMap[target-v]; ok {
            return []int{i,j}
        }
        hashMap[v] = i
    }
    return []int{}
}
```

分析

```go
什么时候使用哈希法：当我们需要查询一个元素是否出现过，或者一个元素是否在集合里的时候，就要第一时间想到哈希法

不仅要知道元素有没有遍历过，还要知道这个元素对应的下标
所以需要使用 key value 结构来存放
key来存元素，value来存下标
```

## [454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/)

给你四个整数数组 `nums1`、`nums2`、`nums3` 和 `nums4` ，数组长度都是 `n` ，请你计算有多少个元组 `(i, j, k, l)` 能满足：

- `0 <= i, j, k, l < n`
- `nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0`

答案

```go
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
```

分析

```go
创建的map中，key放a和b两数之和，value 放a和b两数之和出现的次数
```

## [383. 赎金信](https://leetcode.cn/problems/ransom-note/)

给你两个字符串：`ransomNote` 和 `magazine` ，判断 `ransomNote` 能不能由 `magazine` 里面的字符构成。

如果可以，返回 `true` ；否则返回 `false` 。

`magazine` 中的每个字符只能在 `ransomNote` 中使用一次。

答案

```go
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
```

分析

```go
因为题目所只有小写字母，那可以采用空间换取时间的哈希策略，用一个长度为26的数组还记录magazine里字母出现的次数
```

## [15. 三数之和](https://leetcode.cn/problems/3sum/)

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请

你返回所有和为 `0` 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

答案

```go
func threeSum(nums []int) [][]int  {
    sort.Ints(nums)        // 排序是为了去重
    res := [][]int{}
    length := len(nums)
    for i:=0; i<length-2; i++ {
        n1 := nums[i]    // 先选第一个数 n1
        if n1 > 0 {        // 第一个数就大于0，则不可能三个数和为0
            break    // 接下来都不用再试了
        }
        if i>0 && nums[i]==nums[i-1] {    // 避免重复的答案 比如 -1 -1 0 1
            continue    // 只是跳过当前的i
        }
        l, r := i+1, length-1
        for l < r {
            n2, n3 := nums[l], nums[r]
            if n1+n2+n3 == 0 {
                res = append(res, []int{n1, n2, n3})
                for l<r && n2==nums[l] {
                    l++
                }
                for l<r && n3==nums[r] {
                    r--
                }
            } else if n1+n2+n3 < 0 {
                l++
            } else {
                r--
            }
        }
    }
    return res
}
```

分析

```go
这道题目使用 双指针法 要比 哈希法 高效一些
还要注意去重
```

## [18. 四数之和](https://leetcode.cn/problems/4sum/)

给你一个由 `n` 个整数组成的数组 `nums` ，和一个目标值 `target` 。请你找出并返回满足下述全部条件且**不重复**的四元组 `[nums[a], nums[b], nums[c], nums[d]]` （若两个四元组元素一一对应，则认为两个四元组重复）：

- `0 <= a, b, c, d < n`
- `a`、`b`、`c` 和 `d` **互不相同**
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

你可以按 **任意顺序** 返回答案 。

答案

```go
func fourSum(nums []int, target int) [][]int {
    if len(nums) < 4 {
        return nil
    }
    sort.Ints(nums)
    var res [][]int
    for i := 0; i < len(nums)-3; i++ {
        n1 := nums[i]
        // if n1 > target { // 不能这样写,因为可能是负数
        //     break
        // }
        if i > 0 && n1 == nums[i-1] {
            continue
        }
        for j := i + 1; j < len(nums)-2; j++ {
            n2 := nums[j]
            if j > i+1 && n2 == nums[j-1] {
                continue
            }
            l := j + 1
            r := len(nums) - 1
            for l < r {
                n3 := nums[l]
                n4 := nums[r]
                sum := n1 + n2 + n3 + n4
                if sum < target {
                    l++
                } else if sum > target {
                    r--
                } else {
                    res = append(res, []int{n1, n2, n3, n4})
                    for l < r && n3 == nums[l] { // 去重    这个for循环会至少进入一次（n3和自己作比较）
                        l++
                    }
                    for l < r && n4 == nums[r] { // 去重    这个for循环会至少进入一次（n4和自己作比较）
                        r--
                    }
                }
            }
        }
    }
    return res
}
```

分析

```go
四数之和，和15.三数之和是一个思路，都是使用双指针法, 基本解法就是在15.三数之和的基础上再套一层for循环

四数之和这道题目 target是任意值

对于15.三数之和，双指针法就是将原本暴力O(n^3)的解法，降为O(n^2)的解法
对于18.四数之和，双指针法就是将原本暴力O(n^4)的解法，降为O(n^3)的解法
```

## [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

答案

```go
func groupAnagrams(strs []string) [][]string {
    m := map[[26]int][]string{}    // key是长度为26的数组，value是字符串切片
    for _, str := range strs {    // 统计每个字符串的字母出现次数
        count := [26]int{}
        for _, b := range str {
            count[b-'a']++
        }
        m[count] = append(m[count], str)    // 必须用append，否则会直接覆盖掉
    }
      ans := make([][]string, 0)    // 不能是len(m)，因为会使出现多余的""字符串
    for _, v := range m {
        ans = append(ans, v)
    }
    return ans
}
```

分析

```go
由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中的相同字母出现的次数一定是相同的
故可以将每个字母出现的次数作为哈希表的键
```

## [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

答案

```go
func longestConsecutive(nums []int) int {
    numSet := map[int]bool{}    // 存储数组中的数，顺便去重
    for _, num := range nums {
        numSet[num] = true
    }
    ans := 0
    for num := range numSet {    // 注意遍历map时顺序是随机的，所以要在下面加if判断
        if !numSet[num-1] {    // 要枚举的数 x 一定是在数组中不存在前驱数 x−1 的
            currentNum := num
            count := 1
            for numSet[currentNum+1] {
                currentNum++
                count++
            }
            if ans < count {
                ans = count
            }
        }
    }
    return ans
}
```

分析

```go
用一个哈希表存储数组中的数，这样查看一个数是否存在即能优化至 O(1) 的时间复杂度

通过查找前驱数 x−1 可以让时间复杂度优化为O(n)
```

## [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的子数组的个数* 。

子数组是数组中元素的连续非空序列

答案

```go
// 枚举
func subarraySum(nums []int, k int) int {
    count := 0
    for end := 0; end < len(nums); end++ {
        sum := 0
        for start := end; start >= 0; start-- {
            sum += nums[start]
            if sum == k {
                count++
            }
        }
    }
    return count
}

// 前缀和
func subarraySum(nums []int, k int) int {
    ans, pre := 0, 0    // pre为 [0..i] 里所有数的和
    m := make(map[int]int)    // 哈希表m，和为key，出现次数为value
    m[0] = 1
    for i := 0; i < len(nums); i++ {
        pre += nums[i]
        if _, ok := m[pre - k]; ok { // 找到和为 k 的子串
            ans += m[pre - k]
        }
        m[pre] += 1
    }
    return ans
}
```

分析

```go
通过遍历数组，计算每个位置的前缀和，并使用一个哈希表来存储每个前缀和出现的次数。
在遍历的过程中，检查是否存在pre-k的前缀和，如果存在，说明从某个位置到当前位置的连续子数组的和为k，我们将对应的次数累加到结果中。
```

# 字符串

## [344. 反转字符串](https://leetcode.cn/problems/reverse-string/)

编写一个函数，作用是将输入的字符串反转过来。输入字符串以字符数组 `s` 的形式给出。

不要给另外的数组分配额外的空间，你必须[原地](https://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)修改输入数组、使用 O(1) 的额外空间解决这一问题。

答案

```go
func reverseString(s []byte)  {
    left := 0
    right := len(s)-1
    for left<right {
        s[left], s[right] = s[right], s[left]
        left++
        right--
    }
}
```

分析

```go
对于字符串，我们定义两个指针（也可以说是索引下标），一个从字符串前面，一个从字符串后面，两个指针同时向中间移动，并交换元素
```

## [541. 反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/)

给定一个字符串 `s` 和一个整数 `k`，从字符串开头算起，每计数至 `2k` 个字符，就反转这 `2k` 字符中的前 `k` 个字符。

- 如果剩余字符少于 `k` 个，则将剩余字符全部反转。
- 如果剩余字符小于 `2k` 但大于或等于 `k` 个，则反转前 `k` 个字符，其余字符保持原样。

答案

```go
func reverseStr(s string, k int) string {
    ss := []byte(s)
    length := len(ss)
    for i:=0; i<length; i+=2*k {    // 每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符
        if i+k <= length {
            reverse(ss[i:i+k])
        } else {
            reverse(ss[i:length])
        }
    }
    return string(ss)
}

func reverse(s []byte)  {
    left := 0
    right := len(s)-1
    for left<right {
        s[left], s[right] = s[right], s[left]
        left++
        right--
    }
}
```

分析

```go

```

## 剑指 Offer 05. 替换空格

答案

```go
func replaceSpace(s string) string {
    b := []byte(s)
    result := make([]byte, 0)
    for i:=0; i<len(b); i++ {
        if b[i]==' ' {
            result = append(result, []byte("%20")...)
        } else {
            result = append(result, b[i])
        }
    }
    return string(result)
}
```

分析

```go
很多数组填充类的问题，都可以先预先给数组扩容带填充后的大小，然后在从后向前进行操作。

这么做有两个好处：
    不用申请新数组。
    从后向前填充元素，避免了从前向后填充元素时，每次添加元素都要将添加元素之后的所有元素向后移动的问题。
```

## [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/)

给你一个字符串 `s` ，请你反转字符串中 **单词** 的顺序。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

返回 **单词** 顺序颠倒且 **单词** 之间用单个空格连接的结果字符串。

注意：输入字符串 `s`中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

答案

```go
func reverseWords(s string) string {
    //1.使用双指针删除冗余的空格
    slowIndex, fastIndex := 0, 0
    b := []byte(s)
    //删除头部冗余空格
    for len(b) > 0 && fastIndex < len(b) && b[fastIndex] == ' ' {
        fastIndex++
    }
    //删除单词间冗余空格
    for ; fastIndex < len(b); fastIndex++ {
        if fastIndex-1 > 0 && b[fastIndex-1] == b[fastIndex] && b[fastIndex] == ' ' {
            continue
        }
        b[slowIndex] = b[fastIndex]
        slowIndex++
    }
    //删除尾部冗余空格
    if slowIndex-1 > 0 && b[slowIndex-1] == ' ' {
        b = b[:slowIndex-1]
    } else {
        b = b[:slowIndex]
    }
    //2.反转整个字符串
    reverse151(b, 0, len(b)-1)
    //3.反转单个单词  i单词开始位置，j单词结束位置
    i := 0
    for i < len(b) {
        j := i
        for ; j < len(b) && b[j] != ' '; j++ {
        }
        reverse151(b, i, j-1)
        i = j
        i++
    }
    return string(b)
}

func reverse151(b []byte, left, right int) {
    for left < right {
        b[left], b[right] = b[right], b[left]
        left++
        right--
    }
}
```

分析

```go
将整个字符串都反转过来，那么单词的顺序指定是倒序了，只不过单词本身也倒序了，那么再把单词反转一下，单词不就正过来了。

所以解题思路如下：
    移除多余空格
    将整个字符串反转
    将每个单词反转
```

## 剑指 Offer 58 - II. 左旋转字符串

答案

```go
func reverseLeftWords(s string, n int) string {
    b := []byte(s)
    var reverse func(start, end int)
    reverse = func(start, end int) {
        for start < end {
            b[start], b[end] = b[end], b[start]
            start++
            end--
        }
    }
    m := len(b)
    reverse(0, n-1)
    reverse(n, m-1)
    reverse(0, m-1)
    return string(b)
}
```

分析

```go
可以通过局部反转+整体反转 达到左旋转的目的。

具体步骤为：
    反转区间为前n的子串
    反转区间为n到末尾的子串
    反转整个字符串

最后就可以达到左旋n的目的，而不用定义新的字符串，完全在本串上操作
```

## [415. 字符串相加](https://leetcode.cn/problems/add-strings/)

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 `BigInteger`）， 也不能直接将输入的字符串转换为整数形式。

答案

```go
func addStrings(num1 string, num2 string) string {
    add := 0    // 维护当前是否有进位
    ans := ""
    // 从末尾到开头逐位相加
    for i, j := len(num1)-1, len(num2)-1; i>=0 || j>=0 || add!=0; i, j = i-1, j-1 {
        var x, y int    // 默认为0，即在指针当前下标处于负数的时候返回 0   等价于对位数较短的数字进行了补零操作
        if i >= 0 {
            x = int(num1[i] - '0')
        }
        if j >= 0 {
            y = int(num2[j] - '0')
        }
        tmp := x + y + add
        ans = strconv.Itoa(tmp%10) + ans
        add = tmp / 10
    }
    return ans
}
```

分析

```go

```

## [14. 最长公共前缀](https://leetcode.cn/problems/longest-common-prefix/)

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

答案

```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    for i := 0; i < len(strs[0]); i++ {
        for j := 1; j < len(strs); j++ {
            if i == len(strs[j]) || strs[j][i] != strs[0][i] {
                return strs[0][:i]
            }
        }
    }
    return strs[0]
}
```

分析

```go
纵向扫描
从前往后遍历所有字符串的每一列，比较相同列上的字符是否相同
如果相同则继续对下一列进行比较，
如果不相同则当前列不再属于公共前缀，当前列之前的部分为最长公共前缀
```



## KMP算法

答案

```go

```

分析

https://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html

```go
已知空格与D不匹配时，前面六个字符"ABCDAB"是匹配的。查表可知，最后一个匹配字符B对应的"部分匹配值"为2，因此按照下面的公式算出向后移动的位数：
移动位数 = 已匹配的字符数 - 对应的部分匹配值
```



## 手搓哈希表

答案

```go
package main

import (
        "fmt"
)

// 定义一个链表节点
type Node struct {
        key   string
        value int
        next  *Node
}

// 定义一个HashMap
type HashMap struct {
        buckets []*Node    // 切片的每个元素是一个链表的头节点
        size    int    // 哈希表的大小，决定了哈希表有多少个桶
}

// 初始化HashMap
func NewHashMap(size int) *HashMap {
        return &HashMap{
                buckets: make([]*Node, size),
                size:    size,
        }
}

// 哈希函数，将字符串键转换为一个数组索引
// 每个字符的 ASCII 值乘以其在字符串中的位置（i + 1）来增加分散性，避免相似字符串得到相同的哈希值
// 最后通过 hash % size 确保哈希值在表的范围内。
func (hm *HashMap) hash(key string) int {
    var hash int
    for i, char := range key {
        hash += int(char) * (i + 1) // 使用字符 ASCII 值乘以其位置的权重
    }
    return hash % size // 取模确保哈希值在表大小范围内
}

// Put操作：在HashMap中插入或更新键值对
func (hm *HashMap) Put(key string, value int) {
        index := hm.hash(key)
        node := hm.buckets[index]
        
        // 遍历链表，更新已有的键
        for node != nil {
                if node.key == key {
                        node.value = value
                        return
                }
                node = node.next
        }
        
        // 插入新节点到链表头
        newNode := &Node{key: key, value: value, next: hm.buckets[index]}
        hm.buckets[index] = newNode
}

// Get操作：通过键查找值
func (hm *HashMap) Get(key string) (int, bool) {
        index := hm.hash(key)
        node := hm.buckets[index]
        
        // 遍历链表查找键
        for node != nil {
                if node.key == key {
                        return node.value, true
                }
                node = node.next
        }
        return 0, false // 如果找不到，返回false
}

// Delete操作：通过键删除键值对
func (hm *HashMap) Delete(key string) {
        index := hm.hash(key)
        node := hm.buckets[index]
        
        // 特殊情况：链表头就是目标节点
        if node != nil && node.key == key {
                hm.buckets[index] = node.next
                return
        }

        // 遍历链表找到目标节点，并删除它
        var prev *Node
        for node != nil {
                if node.key == key {
                        prev.next = node.next
                        return
                }
                prev = node
                node = node.next
        }
}

// 打印HashMap内容（用于调试）
func (hm *HashMap) Print() {
        for i, bucket := range hm.buckets {
                fmt.Printf("Bucket %d: ", i)
                node := bucket
                for node != nil {
                        fmt.Printf("%s:%d -> ", node.key, node.value)
                        node = node.next
                }
                fmt.Println("nil")
        }
}

func main() {
        hm := NewHashMap(10)
        
        hm.Put("apple", 5)
        hm.Put("banana", 7)
        hm.Put("orange", 8)
        hm.Put("grape", 6)
        
        fmt.Println("Get apple:", hm.Get("apple")) // 输出: 5, true
        fmt.Println("Get banana:", hm.Get("banana")) // 输出: 7, true
        
        hm.Delete("banana")
        fmt.Println("Get banana after delete:", hm.Get("banana")) // 输出: 0, false
        
        hm.Print() // 打印整个HashMap内容
}
```

分析

```go

```





# 树

## 二叉树递归遍历

![img](https://code-thinking-1253855093.file.myqcloud.com/pics/20200806191109896.png)

### [144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

```go
func preorderTraversal(root *TreeNode) (res []int) {
  var traversal func(node *TreeNode)
  traversal = func(node *TreeNode) {
    if node == nil {
              return
    }
    res = append(res,node.Val)
    traversal(node.Left)
    traversal(node.Right)
    }
  traversal(root)
  return res
}
```

### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```go
func inorderTraversal(root *TreeNode) (res []int) {
    var traversal func(node *TreeNode)
    traversal = func(node *TreeNode) {
        if node == nil {
            return
        }
        traversal(node.Left)
        res = append(res,node.Val)
        traversal(node.Right)
    }
    traversal(root)
    return res
}
```

### [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)

```go
func postorderTraversal(root *TreeNode) (res []int) {
    var traversal func(node *TreeNode)
    traversal = func(node *TreeNode) {
        if node == nil {
            return
        }
        traversal(node.Left)
        traversal(node.Right)
        res = append(res,node.Val)
    }
    traversal(root)
    return res
}
```



## 二叉树非递归遍历

为什么可以用迭代法（非递归的方式）来实现二叉树的前后中序遍历呢？

递归的实现就是：每一次递归调用都会把函数的局部变量、参数值和返回地址等压入调用栈中，然后递归返回的时候，从栈顶弹出上一次递归的各项参数，所以这就是递归为什么可以返回上一层位置的原因。

所以区别在于递归的时候隐式地维护了一个栈，而在迭代的时候需要显式地将这个栈模拟出来，其他都相同。



### [144. 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

```go
func preorderTraversal(root *TreeNode) (vals []int) {
    // 前序遍历是中左右，每次先处理的是中间节点，
    // 所以先将跟节点放入栈中，然后将右孩子加入栈，再加入左孩子。
	// 这样出栈的时候才是中左右的顺序。
    stack := []*TreeNode{}
    node := root
    for node != nil || len(stack) > 0 {
        for node != nil {
            vals = append(vals, node.Val)
            stack = append(stack, node)
            node = node.Left
        }
        node = stack[len(stack)-1].Right
        stack = stack[:len(stack)-1]
    }
    return
}
```

### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```go
func inorderTraversal(root *TreeNode) (res []int) {
	stack := []*TreeNode{}
    // 如果如果所有节点都遍历完成，或者放到栈中的数据全部被弹出，才遍历完成
	for root != nil || len(stack) > 0 {
        // 将树的左节点依次入栈
		for root != nil {	
			stack = append(stack, root)
			root = root.Left
		}
        // 左节点为空时，弹出栈顶元素并处理
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right	// 将从栈中提取的上一个节点的右子树的节点赋值给进行循环的节点
	}
	return
}
```

### [145. 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)

```go
func postorderTraversal(root *TreeNode) (ans []int) {
    // 先序遍历是中左右，后续遍历是左右中，
    // 只需要调整一下先序遍历的代码顺序，就变成中右左的遍历顺序，
    // 然后在反转result数组，输出的结果顺序就是左右中了
	if root == nil {
		return
	}
	stack := []*TreeNode{root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		ans = append(ans, node.Val)
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}
	for i, j := 0, len(ans)-1; i < j; i, j = i+1, j-1 {
		ans[i], ans[j] = ans[j], ans[i]
	}
	return
}

```



## [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

答案

```go
func levelOrder(root *TreeNode) [][]int {
    ans := [][]int{}
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue)>0 {
        tmp := []int{}
        length := len(queue)    //保存当前层的长度，然后处理当前层
        for i:=0; i<length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            tmp = append(tmp, node.Val)    //将值加入本层切片中
        }
        ans = append(ans, tmp)    //放入结果集
    }
    return ans
}
```

分析

```go

```

## [107. 二叉树的层序遍历 II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/)

给你二叉树的根节点 `root` ，返回其节点值 **自底向上的层序遍历** 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

答案

```go
func levelOrderBottom(root *TreeNode) [][]int {
    ans := [][]int{}
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        tmp := []int{}
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            tmp = append(tmp, node.Val) //将值加入本层切片中
        }
        ans = append(ans, tmp) //放入结果集
    }
    left, right := 0, len(ans)-1
    for left < right {
        ans[left], ans[right] = ans[right], ans[left]
        left++
        right--
    }
    return ans
}
```

分析

```go

```

## [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

答案

```go
func rightSideView(root *TreeNode) []int {
    ans := []int{}
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        tmp := []int{}
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            tmp = append(tmp, node.Val) //将值加入本层切片中
        }
        ans = append(ans, tmp[len(tmp)-1]) //放入结果集
    }
    return ans
}
```

分析

```go
每次返回每层的最后一个字段即可
```

## [637. 二叉树的层平均值](https://leetcode.cn/problems/average-of-levels-in-binary-tree/)

给定一个非空二叉树的根节点 `root` , 以数组的形式返回每一层节点的平均值。与实际答案相差 `10-5` 以内的答案可以被接受。

答案

```go
func averageOfLevels(root *TreeNode) []float64 {
    ans := []float64{}
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue)>0 {
    	//tmp := []int{}
        sum := 0
        length := len(queue)    //保存当前层的长度，然后处理当前层
        for i:=0; i<length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            //tmp = append(tmp, node.Val)    //将值加入本层切片中
            sum += node.Val
        }
        ans = append(ans, float64(sum)/float64(length))    //放入结果集
    }
    return ans
}
```

分析

```go
求平均数，结果用float64表示
被除数和除数都得先强制转换为float64类型才可以，否则会报错
```

## [429. N 叉树的层序遍历](https://leetcode.cn/problems/n-ary-tree-level-order-traversal/)

给定一个 N 叉树，返回其节点值的*层序遍历*。（即从左到右，逐层遍历）。

树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。

答案

```go
type Node struct {
    Val      int
    Children []*Node
}

func levelOrder(root *Node) [][]int {
    ans := [][]int{}
    if root == nil {
        return ans
    }
    queue := []*Node{root}
    for len(queue) > 0 {
        tmp := []int{}
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            for j := 0; j < len(node.Children); j++ {
                queue = append(queue, node.Children[j])
            }
            tmp = append(tmp, node.Val) //将值加入本层切片中
        }
        ans = append(ans, tmp) //放入结果集
    }
    return ans
}
```

分析

```go

```

## [515. 在每个树行中找最大值](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)

给定一棵二叉树的根节点 `root` ，请找出该二叉树中每一层的最大值。

答案

```go
func largestValues(root *TreeNode) []int {
    ans := []int{}
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        // maxNumber := int(math.Inf(-1)) //负无穷   因为节点的值会有负数
        length := len(queue)           // 保存当前层的长度，然后处理当前层
        maxNumber := queue[0].Val        // 不用担心越界，因为上面的for循环已经判断过了
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            maxNumber = max(maxNumber, node.Val)
        }
        ans = append(ans, maxNumber) // 放入结果集
    }
    return ans
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go

```

## [116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

给定一个 **完美二叉树** ，其所有叶子节点都在同一层，每个父节点都有两个子节点。

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL`。

初始状态下，所有 next 指针都被设置为 `NULL`。

答案

```go
type PerfectNode struct {
    Val   int
    Left  *PerfectNode
    Right *PerfectNode
    Next  *PerfectNode
}

func connect(root *PerfectNode) *PerfectNode {
    if root == nil {
        return root
    }
    queue := []*PerfectNode{root}
    for len(queue) > 0 {
        length := len(queue) // 保存当前层的长度，然后处理当前层
        pre := queue[0]      // 不用担心越界，因为上面的for循环已经判断过了
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            if i > 0 {
                pre.Next = node
                pre = node
            }
        }
    }
    return root
}
```

分析

```go

```

## [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)

给定一个二叉树：

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 `NULL` 。

初始状态下，所有 next 指针都被设置为 `NULL` 。

答案

```go
type PerfectNode struct {
    Val   int
    Left  *PerfectNode
    Right *PerfectNode
    Next  *PerfectNode
}

func connect(root *PerfectNode) *PerfectNode {
    if root == nil {
        return root
    }
    queue := []*PerfectNode{root}
    for len(queue) > 0 {
        length := len(queue) // 保存当前层的长度，然后处理当前层
        pre := queue[0]      // 不用担心越界，因为上面的for循环已经判断过了
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            if i > 0 {
                pre.Next = node
                pre = node
            }
        }
    }
    return root
}
```

分析

```go
和上一题没差别，只是二叉树是否为完美二叉树
```

## [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

答案

```go
func maxDepth(root *TreeNode) int {
    ans := 0
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        length := len(queue) // 保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        ans++
    }
    return ans
}
```

分析

```go

```

## [111. 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

答案

```go
func minDepth(root *TreeNode) int {
    ans := 0
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left == nil && node.Right == nil { // 当前节点没有左右节点，则代表此层是最小层
                return ans + 1 // 返回当前层 ans代表是上一层
            }
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        ans++
    }
    return ans
}
```

分析

```go
相对于 104.二叉树的最大深度 ，本题也可以使用层序遍历的方式来解决，思路是一样的。

需要注意的是，只有当左右孩子都为空的时候，才说明遍历的最低点了。如果其中一个孩子为空则不是最低点
```

## [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

答案

```go
func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return root
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            node.Left, node.Right = node.Right, node.Left
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
    }
    return root
}
```

分析

```go
只要把每一个节点的左右孩子翻转一下，就可以达到整体翻转的效果

这道题目使用前序遍历、后序遍历、层序遍历都可以，唯独中序遍历不方便，因为中序遍历会把某些节点的左右孩子翻转了两次
```

## [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

答案

```go
// 101. 对称二叉树
func isSymmetric(root *TreeNode) bool {
    var queue []*TreeNode
    if root != nil {
        queue = append(queue, root.Left, root.Right)
    }
    for len(queue) > 0 {
        left := queue[0]  // 将左子树头结点加入队列
        right := queue[1] // 将右子树头结点加入队列
        queue = queue[2:]
        if left == nil && right == nil { // 左节点为空、右节点为空，此时说明是对称的
            continue
        }
        // 左右一个节点不为空，或者都不为空但数值不相同，返回false
        if left == nil || right == nil || left.Val != right.Val {
            return false
        }
        // 依次加入：左节点左孩子、右节点右孩子、左节点右孩子、右节点左孩子
        queue = append(queue, left.Left, right.Right, left.Right, right.Left)
    }
    return true
}
```

分析

```go
通过队列来判断根节点的左子树和右子树的内侧和外侧是否相等
```

## [222. 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)

给你一棵 **完全二叉树** 的根节点 `root` ，求出该树的节点个数。

[完全二叉树](https://baike.baidu.com/item/%E5%AE%8C%E5%85%A8%E4%BA%8C%E5%8F%89%E6%A0%91/7773232?fr=aladdin) 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 `h` 层，则该层包含 `1~ 2^h` 个节点。

答案

```go
func countNodes(root *TreeNode) int {
    ans := 0
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            ans++
        }
    }
    return ans
}
```

分析

```go

```

## [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)

给你二叉树的根节点 `root` ，返回其节点值的 **锯齿形层序遍历** 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

答案

```go
func zigzagLevelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    queue := []*TreeNode{root}
    ans := make([][]int, 0)
    for level := 0; len(queue) > 0; level++ {
        length := len(queue)
        tmp := make([]int, 0)
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            tmp = append(tmp, node.Val)
            if node.Left != nil { // append之前要先判断是不是nil
                queue = append(queue, node.Left)
            }
            if node.Right != nil { // append之前要先判断是不是nil
                queue = append(queue, node.Right)
            }
        }
        if level%2 == 1 {
            i, j := 0, len(tmp)-1
            for i < j {
                tmp[i], tmp[j] = tmp[j], tmp[i]
                i++
                j--
            }
        }
        ans = append(ans, tmp)
    }
    return ans
}
```

分析

```go
层序遍历
```

## [513. 找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/)

给定一个二叉树的 **根节点** `root`，请找出该二叉树的 **最底层 最左边** 节点的值。

假设二叉树中至少有一个节点。

答案

```go
func findBottomLeftValue(root *TreeNode) int {
    ans := 0
    if root == nil {
        return ans
    }
    queue := []*TreeNode{root}
    for len(queue) > 0 {
        length := len(queue) //保存当前层的长度，然后处理当前层
        for i := 0; i < length; i++ {
            node := queue[0]
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
            if i == 0 {    // 记录每一行第一个元素
                ans = node.Val
            }
        }
    }
    return ans
}
```

分析

```go
在树的最后一行找到最左边的值
```

## [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

给定一个二叉树，判断它是否是 平衡二叉树。

**平衡二叉树** 是指该树所有节点的左右子树的深度相差不超过 1。

答案

```go
func isBalanced(root *TreeNode) bool {
    if root == nil {
        return true
    }

  // 自顶向下：
  // return abs(height(root.Left) - height(root.Right)) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)

  // 自底向上：
    if !isBalanced(root.Left) || !isBalanced(root.Right) {
        return false
    }
    // 分别求出其左右子树的高度
    LeftH := maxHeight(root.Left) + 1 // 以当前节点为根节点的树的最大高度
    RightH := maxHeight(root.Right) + 1
    // 如果差值大于1，表示已经不是二叉平衡树
    if abs(LeftH-RightH) > 1 {
        return false
    }
    return true
}

func maxHeight(root *TreeNode) int {
    if root == nil {
        return 0
    }
    return max(maxHeight(root.Left), maxHeight(root.Right)) + 1
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
二叉树节点的深度：指从根节点到该节点的最长简单路径边的条数。
二叉树节点的高度：指从该节点到叶子节点的最长简单路径边的条数。

自顶向下的递归  复杂度n方
对于当前遍历到的节点，首先计算左右子树的高度，判断左右子树的高度差是否不超过 1
再分别递归地遍历左右子节点，并判断左子树和右子树是否平衡

自底向上递归  复杂度n    每个节点的计算高度和判断是否平衡都只需要处理一次
类似于后序遍历，对于当前遍历到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。
如果两棵子树是平衡的，则求其高度。    如果存在一棵子树不平衡，则整个二叉树一定不平衡。

怎么理解递归   先不思考细节，要思考递归函数的功能是什么
```

## [257. 二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)

给你一个二叉树的根节点 `root` ，按 **任意顺序** ，返回所有从根节点到叶子节点的路径。

答案

```go
func binaryTreePaths(root *TreeNode) []string {
    res := make([]string, 0)
    var travel func(node *TreeNode, s string)
    travel = func(node *TreeNode, s string) {
        if node.Left == nil && node.Right == nil { // 找到了叶子节点
            v := s + strconv.Itoa(node.Val) // 找到一条路径
            res = append(res, v)            // 添加到结果集
            return
        }
        s += strconv.Itoa(node.Val) + "->" // 非叶子节点，后面多加符号
        if node.Left != nil {    // 要额外判断是否为空
            travel(node.Left, s)
        }
        if node.Right != nil {    // 要额外判断是否为空
            travel(node.Right, s)
        }
    }
    travel(root, "")
    return res
}
```

分析

```go

```

## [404. 左叶子之和](https://leetcode.cn/problems/sum-of-left-leaves/)

给定二叉树的根节点 `root` ，返回所有左叶子之和。

答案

```go
func sumOfLeftLeaves(root *TreeNode) int {
    res := 0
    var findLeft func(node *TreeNode)
    findLeft = func(node *TreeNode) {
        if node.Left != nil && node.Left.Left == nil && node.Left.Right == nil {
            res += node.Left.Val
        }
        if node.Left != nil {
            findLeft(node.Left)
        }
        if node.Right != nil {
            findLeft(node.Right)
        }
    }
    findLeft(root)
    return res
}
```

分析

```go
左叶子的明确定义：
节点A的左孩子不为空，且左孩子的左右孩子都为空（说明是叶子节点），那么A节点的左孩子为左叶子节点
```

## [112. 路径总和](https://leetcode.cn/problems/path-sum/)

给你二叉树的根节点 `root` 和一个表示目标和的整数 `targetSum` 。判断该树中是否存在 **根节点到叶子节点** 的路径，这条路径上所有节点值相加等于目标和 `targetSum` 。如果存在，返回 `true` ；否则，返回 `false` 。

答案

```go
func hasPathSum(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    targetSum -= root.Val
    if root.Left == nil && root.Right == nil && targetSum == 0 { // 遇到叶子节点，并且计数为0
        return true    
    }
  // 找到一个即可，所以不用回溯
    return hasPathSum(root.Left, targetSum) || hasPathSum(root.Right, targetSum)
}
```

分析

```go
根节点到叶子节点的路径
```

## [113. 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

答案

```go
func pathSum(root *TreeNode, targetSum int) [][]int {
    res := make([][]int, 0)
    curPath := make([]int, 0)
    var traverse func(node *TreeNode, targetSum int)
    traverse = func(node *TreeNode, targetSum int) {
        if node == nil {
            return
        }
        targetSum -= node.Val                                        // 将targetSum在遍历每层的时候都减去本层节点的值
        curPath = append(curPath, node.Val)                          // 把当前节点放到路径记录里
        if node.Left == nil && node.Right == nil && targetSum == 0 { // 遇到叶子节点，并且计数为0
            // 不能直接将currPath放到result里面, 因为currPath是共享的, 每次遍历子树时都会被修改
            pathCopy := make([]int, len(curPath))
            copy(pathCopy, curPath)
            res = append(res, pathCopy) // 将副本放到结果集里
        }
        traverse(node.Left, targetSum)
        traverse(node.Right, targetSum)
        curPath = curPath[:len(curPath)-1] // 当前节点遍历完成, 从路径记录里删除掉
    }
    traverse(root, targetSum)
    return res
}
```

分析

```go
要遍历整个树，找到所有路径，所以需要回溯
```

## [654. 最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)

给定一个不重复的整数数组 `nums` 。 **最大二叉树** 可以用下面的算法从 `nums` 递归地构建:

1. 创建一个根节点，其值为 `nums` 中的最大值。
2. 递归地在最大值 **左边** 的 **子数组前缀上** 构建左子树。
3. 递归地在最大值 **右边** 的 **子数组后缀上** 构建右子树。

返回 *`nums` 构建的* ***最大二叉树*** 。

答案

```go
func constructMaximumBinaryTree(nums []int) *TreeNode {
    if len(nums) < 1 {
        return nil
    }
    // 找到最大值
    index := findMax(nums)
    // 构造二叉树
    root := &TreeNode{
        Val:   nums[index],
        Left:  constructMaximumBinaryTree(nums[:index]),
        Right: constructMaximumBinaryTree(nums[index+1:]),
    }
    return root
}

func findMax(nums []int) (index int) {
    for i := 0; i < len(nums); i++ {
        if nums[i] > nums[index] {
            index = i
        }
    }
    return index
}
```

分析

```go

```

## [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

答案

```go
func findRootIndex(inorder []int, target int) (index int) {
    for i := 0; i < len(inorder); i++ {
        if target == inorder[i] {
            return i
        }
    }
    return -1
}

// 105. 从前序与中序遍历序列构造二叉树
func buildTree2(preorder []int, inorder []int) *TreeNode {
    if len(preorder) < 1 || len(inorder) < 1 {
        return nil
    }
    // 先找到根节点（先序遍历的第一个就是根节点）
    // 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
    mid := findRootIndex(inorder, preorder[0])
    // 构造root
    root := &TreeNode{
        Val:   preorder[0],
        Left:  buildTree2(preorder[1:mid+1], inorder[:mid]), // 将先序遍历一分为二，左边为左子树，右边为右子树
        Right: buildTree2(preorder[mid+1:], inorder[mid+1:]), // 一棵树的中序遍历和前序遍历的长度相等
    }
    return root
}
```

分析

```go
前序和中序可以唯一确定一棵二叉树。

后序和中序可以唯一确定一棵二叉树。

前序和后序不能唯一确定一棵二叉树！因为没有中序遍历无法确定左右部分，也就是无法分割
```

## [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

给定两个整数数组 `inorder` 和 `postorder` ，其中 `inorder` 是二叉树的中序遍历， `postorder` 是同一棵树的后序遍历，请你构造并返回这颗 *二叉树* 。

答案

```go
// 106. 从中序与后序遍历序列构造二叉树
func buildTree(inorder []int, postorder []int) *TreeNode {
    if len(inorder) < 1 || len(postorder) < 1 {
        return nil
    }
    // 先找到根节点（后续遍历的最后一个就是根节点）
    nodeValue := postorder[len(postorder)-1]
    // 从中序遍历中找到一分为二的点，左边为左子树，右边为右子树
    mid := findRootIndex(inorder, nodeValue)
    // 构造root
    root := &TreeNode{
        Val:   nodeValue,
        Left:  buildTree(inorder[:mid], postorder[:mid]), // 一棵树的中序遍历和后序遍历的长度相等
        Right: buildTree(inorder[mid+1:], postorder[mid:len(postorder)-1]),
    }
    return root
}

func findRootIndex(inorder []int, target int) (index int) {
    for i := 0; i < len(inorder); i++ {
        if target == inorder[i] {
            return i
        }
    }
    return -1
}
```

分析

```go
如何根据两个顺序构造一个唯一的二叉树：
以后序数组的最后一个元素为切割点，先切中序数组，根据中序数组，反过来再切后序数组。
一层一层切下去，每次后序数组最后一个元素就是节点元素。
```

## [617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)

给你两棵二叉树： `root1` 和 `root2` 。

想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，**不为** null 的节点将直接作为新二叉树的节点。

返回合并后的二叉树。

**注意:** 合并过程必须从两个树的根节点开始。

答案

```go
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
    if root1 == nil {
        return root2
    }
    if root2 == nil {
        return root1
    }
    root1.Val += root2.Val
    root1.Left = mergeTrees(root1.Left, root2.Left)
    root1.Right = mergeTrees(root1.Right, root2.Right)
    return root1
}
```

分析

```go

```

## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

答案

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    parent := map[int]*TreeNode{} // 题目给出：所有 Node.val 互不相同，所以可以用int存储
    visited := map[int]bool{}
    var dfs func(node *TreeNode)
    // 从根节点开始遍历整棵二叉树，用哈希表记录每个节点的父节点指针
    dfs = func(node *TreeNode) {
        if node == nil {
            return
        }
        if node.Left != nil {
            parent[node.Left.Val] = node
            dfs(node.Left)
        }
        if node.Right != nil {
            parent[node.Right.Val] = node
            dfs(node.Right)
        }
    }
    dfs(root)
    // 从 p 节点开始不断往它的祖先移动，并记录已经访问过的祖先节点
    for p != nil {
        visited[p.Val] = true
        p = parent[p.Val]
    }
    // 再从 q 节点开始不断往它的祖先移动，如果有祖先已经被访问过，即意味着这是 p 和 q 的深度最深的公共祖先，即 LCA 节点
    for q != nil {
        if visited[q.Val] {
            return q
        }
        q = parent[q.Val]
    }
    return nil
}
```

分析

```go
遇到这个题目首先想的是要是能自底向上查找就好了，这样就可以找到公共祖先了
二叉树回溯的过程就是从低到上，后序遍历（左右中）就是天然的回溯过程，可以根据左右子树的返回值，来处理中节点的逻辑
如果找到一个节点，发现左子树出现结点p，右子树出现节点q，或者 左子树出现结点q，右子树出现节点p，那么该节点就是节点p和q的最近公共祖先
```

## [700. 二叉搜索树中的搜索](https://leetcode.cn/problems/search-in-a-binary-search-tree/)

给定二叉搜索树（BST）的根节点 `root` 和一个整数值 `val`。

你需要在 BST 中找到节点值等于 `val` 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 `null` 。

答案

```go
// 递归
func searchBST(root *TreeNode, val int) *TreeNode {
    if root==nil{
        return nil
    }
    if root.Val == val {
        return root
    }
    if root.Val > val {
        return searchBST(root.Left, val)
    }
    return searchBST(root.Right, val)
}

// 迭代
func searchBST(root *TreeNode, val int) *TreeNode {
    for root != nil {
        if root.Val > val {
            root = root.Left
        } else if root.Val < val {
            root = root.Right
        } else {
            return root
        }
    }
    return nil
}
```

分析

```go
因为二叉搜索树的特殊性，也就是节点的有序性，可以不使用辅助栈或者队列就可以写出迭代法
```

## [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。

- 节点的右子树只包含 **大于** 当前节点的数。

- 所有左子树和右子树自身必须也是二叉搜索树。

答案

```go
func isValidBST(root *TreeNode) bool {
    var pre *TreeNode // 用来记录前一个节点
    var check func(node *TreeNode) bool
    check = func(node *TreeNode) bool {
        if node == nil {
            return true
        }
        // 中序遍历，验证遍历的元素是不是从小到大
        left := check(node.Left)
        if pre != nil && pre.Val >= node.Val {
            return false
        }
        pre = node
        right := check(node.Right)
        return left && right // 分别对左子树和右子树递归判断，如果左子树和右子树都符合则返回true
    }
    return check(root)
}

//二叉搜索树的中序遍历结果一定是递增的，记录中序遍历结果
func isValidBST(root *TreeNode) bool {
    var queue []int
    queue = midDFS(queue, root)

    // 验证大小顺序
    for i := 0; i < len(queue)-1; i++ {
        if queue[i] >= queue[i+1] {
            return false
        }
    }
    return true
}

// 中序遍历
func midDFS(queue []int, root *TreeNode) []int {
    if root == nil {
        return queue
    }

    queue = midDFS(queue, root.Left)
    queue = append(queue, root.Val)
    queue = midDFS(queue, root.Right)
    return queue
}
```

分析

```go
在中序遍历下，输出的二叉搜索树节点的数值是有序序列
有了这个特性，验证二叉搜索树，就相当于变成了判断一个序列是不是递增的了
```

## [530. 二叉搜索树的最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)

给你一个二叉搜索树的根节点 `root` ，返回 **树中任意两不同节点值之间的最小差值** 。

差值是一个正数，其数值等于两值之差的绝对值。

答案

```go
// 中序递归-前驱节点
func getMinimumDifference(root *TreeNode) int {
    var pre *TreeNode // 用来记录前一个节点
    minDelta := math.MaxInt64
    var calc func(node *TreeNode)
    calc = func(node *TreeNode) {
        if node == nil {
            return
        }
        // 中序遍历
        calc(node.Left)
        if pre != nil {
            minDelta = min(minDelta, node.Val-pre.Val)
        }
        pre = node
        calc(node.Right)
    }
    calc(root)
    return minDelta
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

------------------------------------------------------------------------------------------------
// 中序遍历-数组遍历
func getMinimumDifference(root *TreeNode) int {
    // 中序遍历成数组，遍历数组得绝对值差值
    res := []int{}
    var travel func(node *TreeNode)
    travel = func(node *TreeNode) {
        if node == nil {
            return
        }
        travel(node.Left)
        res = append(res, node.Val)
        travel(node.Right)
    }
    // 遍历BST到数组
    travel(root)
    if len(res) < 2 {
        return 0
    }
    // 找到最小差值
    Min := res[1] - res[0]
    for i:=2;i<len(res);i++ {
        diff := res[i] - res[i-1]
        if Min > diff {
            Min = diff
        }
    }
    return Min
}
```

分析

```go
最直观的想法，就是把二叉搜索树转换成有序数组，然后遍历一遍数组，就统计出来最小差值了
```

## [501. 二叉搜索树中的众数](https://leetcode.cn/problems/find-mode-in-binary-search-tree/)

给你一个含重复值的二叉搜索树（BST）的根节点 `root` ，找出并返回 BST 中的所有 [众数](https://baike.baidu.com/item/%E4%BC%97%E6%95%B0/44796)（即，出现频率最高的元素）。

如果树中有不止一个众数，可以按 **任意顺序** 返回。

假定 BST 满足如下定义：

- 结点左子树中所含节点的值 **小于等于** 当前节点的值
- 结点右子树中所含节点的值 **大于等于** 当前节点的值
- 左子树和右子树都是二叉搜索树

答案

```go
func findMode(root *TreeNode) []int {
    res := make([]int, 0)
    count := 1
    max := 1
    var prev *TreeNode
    var travel func(node *TreeNode)
    travel = func(node *TreeNode) { // 中序遍历
        if node == nil {
            return
        }
        travel(node.Left)
        if prev != nil && prev.Val == node.Val { // 遇到相同的值，计数+1
            count++
        } else {
            count = 1 // 遇到新的值，重新开始计数
        }
        if count >= max {
            if count > max && len(res) > 0 { // 遇到出现次数更多的值，重置res
                res = []int{node.Val}
            } else {
                res = append(res, node.Val) // 遇到出现次数相同的值，res多加一个值
            }
            max = count
        }
        prev = node
        travel(node.Right)
    }
    travel(root)
    return res
}
```

分析

```go

```

## [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

答案

```go
func lowestCommonAncestor2(root, p, q *TreeNode) *TreeNode {
    // 如果找到了 节点p或者q，或者遇到空节点，就返回。
    if root == nil {
        return nil
    }
    if root.Val >= p.Val && root.Val <= q.Val { // 当前节点的值在给定值的中间（或者等于），即为最深的祖先
        return root
    }
    if root.Val > p.Val && root.Val > q.Val { // 当前节点的值大于给定的值，则说明满足条件的点在左子树
        return lowestCommonAncestor(root.Left, p, q)
    }
    if root.Val < p.Val && root.Val < q.Val { // 当前节点的值小于各点的值，则说明满足条件的点在右子树
        return lowestCommonAncestor(root.Right, p, q)
    }
    return root
}
```

分析

```go
因为是有序树，所有 如果 中间节点是 p 和 q 的公共祖先，那么 中节点的数组 一定是在[p, q]区间的

当我们从上向下去递归遍历，第一次遇到cur节点是数值在[p, q]区间中，那么cur就是p和q的最近公共祖先
```

## [701. 二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)

给定二叉搜索树（BST）的根节点 `root` 和要插入树中的值 `value` ，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 **保证** ，新值和原始二叉搜索树中的任意节点值都不同。

**注意**，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 **任意有效的结果** 。

答案

```go
func insertIntoBST(root *TreeNode, val int) *TreeNode {
    // 找到遍历的节点为null的时候，就是要插入节点的位置了，并把插入的节点返回
    if root == nil {
        root = &TreeNode{Val: val}
        return root
    }
    if root.Val > val {
        root.Left = insertIntoBST(root.Left, val)
    } else {
        root.Right = insertIntoBST(root.Right, val)
    }
    return root
}
```

分析

```go
只要遍历二叉搜索树，找到空节点 插入元素就可以

递归的思路：
1、假设第一次调用就找到了符合的条件，对其进行处理
2、假设第一次的调用不符合处理的条件，要继续递归调用
```

## [450. 删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/)

给定一个二叉搜索树的根节点 **root** 和一个值 **key**，删除二叉搜索树中的 **key** 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1. 首先找到需要删除的节点；
2. 如果找到了，删除它。

答案

```go
func deleteNode(root *TreeNode, key int) *TreeNode {
    switch {
    // 没找到删除的节点，遍历到空节点直接返回了
    case root == nil:
        return nil
    // 说明要删除的节点在左子树    左递归
    case root.Val > key:
        root.Left = deleteNode(root.Left, key)
    // 说明要删除的节点在右子树    右递归
    case root.Val < key:
        root.Right = deleteNode(root.Right, key)
    // 找到了节点，且左儿子或右儿子有空的
    case root.Left == nil || root.Right == nil:
        // 右儿子为空，删除节点后左儿子补位
        if root.Left != nil {
            return root.Left
        }
        // 左儿子为空，删除节点后右儿子补位
        return root.Right
    // 找到了节点，且左右儿子都不为空
    // 则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
    // 并返回删除节点右孩子为新的根节点
    default:
        successor := root.Right
        for successor.Left != nil {
            successor = successor.Left
        }
        // 删除节点的右子树的最左面节点变为新的根节点
        // 所以新的根节点的右子树应该是删除节点的右子树去掉该新根节点
        // 新的根节点的左子树应该是删除节点的左子树
        successor.Right = deleteNode(root.Right, successor.Val)
        successor.Left = root.Left
        return successor
    }
    return root
}
```

分析

```go
只要遍历二叉搜索树，找到空节点 插入元素就可以
```

## [669. 修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)

给你二叉搜索树的根节点 `root` ，同时给定最小边界`low` 和最大边界 `high`。通过修剪二叉搜索树，使得所有节点的值在`[low, high]`中。修剪树 **不应该** 改变保留在树中的元素的相对结构 (即，如果没有被移除，原有的父代子代关系都应当保留)。 可以证明，存在 **唯一的答案** 。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

答案

```go
func trimBST(root *TreeNode, low int, high int) *TreeNode {
    if root == nil {
        return nil
    }
    // 如果该节点值小于最小值，则该节点更换为该节点的经过递归处理的右节点值，继续遍历
    if root.Val < low {
        return trimBST(root.Right, low, high)
    }
    // 如果该节点的值大于最大值，则该节点更换为该节点的经过递归处理的左节点值，继续遍历
    if root.Val > high {
        return trimBST(root.Left, low, high)
    }
    // 该节点的值在low和high之间，则处理符合条件的左右子树
    root.Left = trimBST(root.Left, low, high)
    root.Right = trimBST(root.Right, low, high)
    return root
}
```

分析

```go

```

## [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵平衡二叉搜索树。

答案

```go
func sortedArrayToBST(nums []int) *TreeNode {
    // 终止条件，最后数组为空则可以返回
    if len(nums) == 0 {
        return nil
    }
    // 按照BSL的特点，从中间构造节点
    root := &TreeNode{nums[len(nums)/2], nil, nil}
    // 数组的左边为左子树
    root.Left = sortedArrayToBST(nums[:len(nums)/2])
    // 数字的右边为右子树
    root.Right = sortedArrayToBST(nums[len(nums)/2+1:])
    return root
}
```

分析

```go
如果根据数组构造一棵二叉树，本质就是寻找分割点，分割点作为当前节点，然后递归左区间和右区间

有序数组构造二叉搜索树，分割点就是数组中间位置的节点。

本题要构造二叉树，依然用递归函数的返回值来构造中节点的左右孩子
```

## [538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

给出二叉 **搜索** 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 `node` 的新值等于原树中大于或等于 `node.val` 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

- 节点的左子树仅包含键 **小于** 节点键的节点。
- 节点的右子树仅包含键 **大于** 节点键的节点。
- 左右子树也必须是二叉搜索树。

答案

```go
func convertBST(root *TreeNode) *TreeNode {
    sum := 0
    var rightMLeft func(root *TreeNode)
    rightMLeft = func(root *TreeNode) {
        if root == nil { // 终止条件，遇到空节点就返回
            return
        }
        rightMLeft(root.Right) // 先遍历右边
        tmp := sum             // 暂存总和值
        sum += root.Val        // 将总和值变更
        root.Val += tmp        // 更新节点值
        rightMLeft(root.Left)  // 遍历左节点
        return
    }
    rightMLeft(root)
    return root
}
```

分析

```go

```

## [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)

二叉树中的 **路径** 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 **至多出现一次** 。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

答案

```go
func maxPathSum(root *TreeNode) int {
    maxSum := root.Val
    var maxGain func(node *TreeNode) int    // 计算包含该节点的最大贡献值
    maxGain = func(node *TreeNode) int {
        if node == nil {
            return 0
        }
        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        leftGain := max(0, maxGain(node.Left))
        rightGain := max(0, maxGain(node.Right))

        // 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
        pricePath := node.Val + leftGain + rightGain
        // 更新答案
        maxSum = max(maxSum, pricePath)
        // 返回节点的最大贡献值   贡献值指的是半边的路径总和
        return node.Val + max(leftGain, rightGain)    // 贡献值和答案不同，所以是max(leftGain, rightGain)
    }
    maxGain(root)
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
 叶节点的最大贡献值为   本身
非叶节点的最大贡献值为   本身 + max(左儿子,右儿子)

注意贡献值与0要做比较
```

## [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

答案

```go
func diameterOfBinaryTree(root *TreeNode) int {
    ans := 0
    var traversal func(node *TreeNode) int  // 获取以node为根的树的深度
    traversal = func(node * TreeNode) int {
        if node == nil {
            return 0
        }
        left := traversal(node.Left)    // 左儿子为根的子树的深度
        right := traversal(node.Right)  // 右儿子为根的子树的深度
        ans = max(ans, left + right + 1)    // 计算直径
        return max(left, right) + 1 // 返回最大的深度+1
    }
    traversal(root)
    return ans-1
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
一条路径的长度为该路径经过的节点数减一，所以求直径（路径长度的最大值）等效于求路径经过节点数的最大值减一
任意一条路径均可以被看作由某个节点为起点，从其左儿子和右儿子向下遍历的路径拼接得到

所以算法流程为：
定义一个递归函数，函数返回该节点为根的子树的深度。
先递归调用左儿子和右儿子求得它们为根的子树的深度 L 和 R ，
则该节点为根的子树的深度即为max(L, R) + 1，
该节点的直径为 L + R + 1
```

## [230. 二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。

答案

```go
func kthSmallest(root *TreeNode, k int) int {
    stack := []*TreeNode{}
    for {
        for root != nil {
            stack = append(stack, root)
            root = root.Left
        }
        stack, root = stack[:len(stack)-1], stack[len(stack)-1]
        k--
        if k == 0 {
            return root.Val
        }
        root = root.Right
    }
}
```

分析

```go
使用迭代方法，这样可以在找到答案后停止，不需要遍历整棵树

迭代遍历的图解：
https://leetcode.cn/problems/binary-tree-inorder-traversal/solutions/412886/er-cha-shu-de-zhong-xu-bian-li-by-leetcode-solutio/
```

## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树 [**先序遍历**](https://baike.baidu.com/item/%E5%85%88%E5%BA%8F%E9%81%8D%E5%8E%86/6442839?fr=aladdin) 顺序相同。

答案

```go
// 前序遍历
func flatten(root *TreeNode)  {
    list := preorderTraversal(root)
    for i := 1; i < len(list); i++ {
        prev, curr := list[i-1], list[i]
        prev.Left, prev.Right = nil, curr
    }
}

func preorderTraversal(root *TreeNode) []*TreeNode {
    list := []*TreeNode{}
    if root != nil {
        list = append(list, root)
        list = append(list, preorderTraversal(root.Left)...)
        list = append(list, preorderTraversal(root.Right)...)
    }
    return list
}


// 变形的后序遍历，遍历顺序是右子树->左子树->根节点
func flatten(root *TreeNode)  {
  var pre *TreeNode
  var traversal func(node *TreeNode)
  traversal = func(node *TreeNode) {
    if node == nil {
      return
    }
    traversal(node.Right)
    traversal(node.Left)
    node.Right = pre
    node.Left = nil
    pre = node
  }
  traversal(root)
}
```

分析

```go
解法2参考：
https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/solutions/17274/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--26/?envType=study-plan-v2&envId=top-100-liked
```

## [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

答案

```go
// 以节点 p 为起点向下且满足路径总和为 val 的路径数目
func rootSum(root *TreeNode, targetSum int) int {
    res := 0
    if root == nil {
        return res
    }
    val := root.Val
    if val == targetSum {
        res++
    }
    res += rootSum(root.Left, targetSum-val)
    res += rootSum(root.Right, targetSum-val)
    return res
}

func pathSum(root *TreeNode, targetSum int) int {
    if root == nil {
        return 0
    }
    res := rootSum(root, targetSum)    // 以root为起点向下
    res += pathSum(root.Left, targetSum)    // 以root.Left及其子树为起点向下
    res += pathSum(root.Right, targetSum)    // 以root.Right及其子树为起点向下
    return res
}
```

分析

```go

```



## 前缀树

https://study.geekai.co/posts/go-trie-tree-algorithm 

构建 Trie 树的过程比较耗时，对于有 n 个字符的字符串集合而言，需要遍历所有字符，对应的时间复杂度是 O(n)，但是一旦构建之后，查询效率很高，如果匹配串的长度是 k，那只需要匹配 k 次即可，与原来的主串没有关系，所以对应的时间复杂度是 O(k)，基本上是个常量级的数字。 
Trie 树显然也是一种空间换时间的做法，构建 Trie 树的过程需要额外的存储空间存储 Trie 树，而且这个额外的空间是原来的数倍。



答案

```go
// Trie 树节点
type trieNode struct {
    char string                    // Unicode 字符
    isEnding bool                  // 是否是单词结尾
    children map[rune]*trieNode    // 该节点的子节点字典
}

// 初始化 Trie 树节点
func NewTrieNode(char string) *trieNode  {
    return &trieNode{
        char: char,
        isEnding: false,
        children: make(map[rune]*trieNode),
    }
}

// Trie 树结构
type Trie struct {
    root *trieNode     // 根节点指针
}

// 初始化 Trie 树
func NewTrie() *Trie {
    // 初始化根节点
    trieNode := NewTrieNode("/")
    return &Trie{trieNode}
}

// 往 Trie 树中插入一个单词
func (t *Trie) Insert(word string) {
    node := t.root   // 获取根节点
    for _, code := range word {  // 以 Unicode 字符遍历该单词
        value, ok := node.children[code]  // 获取 code 编码对应子节点
        if !ok {
            // 不存在则初始化该节点
            value = NewTrieNode(string(code))
            // 然后将其添加到子节点字典
            node.children[code] = value
        }
        // 当前节点指针指向当前子节点
        node = value
    }
    node.isEnding = true  // 一个单词遍历完所有字符后将结尾字符打上标记
}

// 在 Trie 树中查找一个单词
func (t *Trie) Find(word string) bool {
    node := t.root
    for _, code := range word {
        value, ok := node.children[code] // 获取对应子节点
        if !ok {
            // 不存在则直接返回
            return false
        }
        // 否则继续往后遍历
        node = value
    }
    if node.isEnding == false {
        return false // 不能完全匹配，只是前缀
    }
    return true // 找到对应单词
}

func main() {
    trie := NewTrie()
    words := []string{"Golang", "学院君", "Language", "Trie", "Go"}
    // 构建 Trie 树
    for _, word := range words {
        trie.Insert(word)
    }
    // 从 Trie 树中查找字符串
    term := "学院君"
    if trie.Find(term) {
        fmt.Printf("包含单词\"%s\"\n", term)
    } else {
        fmt.Printf("不包含单词\"%s\"\n", term)
    }
}
```

分析

```go

```

# 



# 回溯

## [77. 组合](https://leetcode.cn/problems/combinations/)

给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案

答案

```go
func combine(n int, k int) [][]int {
    res := [][]int{}
    var backtrace func(start int, trace []int)
    backtrace = func(start int, trace []int) {
        if len(trace) == k {
            tmp := make([]int, k)
            copy(tmp, trace)
            res = append(res, tmp)
        }
        if len(trace)+n-start+1 < k { // 剪枝优化
            return
        }
        for i := start; i <= n; i++ { // 选择本层集合中元素，控制树的横向遍历
            trace = append(trace, i)     // 处理节点
            backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
            trace = trace[:len(trace)-1] // 回溯，撤销处理结果
        }
    }
    backtrace(1, []int{})
    return res
}
```

分析

```go

```

## [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

找出所有相加之和为 `n` 的 `k` 个数的组合，且满足下列条件：

- 只使用数字1到9
- 每个数字 **最多使用一次** 

返回 *所有可能的有效组合的列表* 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。

答案

```go
func combinationSum3(k int, n int) [][]int {
    res := [][]int{}
    var backtrace func(start int, trace []int)
    backtrace = func(start int, trace []int) {
        if len(trace) == k {
            sum := 0
            tmp := make([]int, k)
            for i, v := range trace {
                sum += v
                tmp[i] = v
            }
            if sum == n {
                res = append(res, tmp)
            }
            return
        }
        if start > n { // 剪枝优化
            return
        }
        for i := start; i <= 9; i++ { // 选择本层集合中元素，控制树的横向遍历
            trace = append(trace, i)     // 处理节点
            backtrace(i+1, trace)        // 递归：控制树的纵向遍历，注意下一层搜索要从i+1开始
            trace = trace[:len(trace)-1] // 回溯，撤销处理结果
        }
    }
    backtrace(1, []int{})
    return res
}
```

分析

```go

```

## [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

答案

```go
func letterCombinations(digits string) []string {
    digitsMap := [10]string{
        "",     // 0
        "",     // 1
        "abc",  // 2
        "def",  // 3
        "ghi",  // 4
        "jkl",  // 5
        "mno",  // 6
        "pqrs", // 7
        "tuv",  // 8
        "wxyz", // 9
    }

    res := []string{}
    length := len(digits)
    if length <= 0 || length > 4 {
        return res
    }
    var backtrace func(s string, index int)
    backtrace = func(s string, index int) {
        if len(s) == length {
            res = append(res, s)
            return
        }
        num := digits[index] - '0' // 将index指向的数字转为int
        letter := digitsMap[num]   // 取数字对应的字符集
        for i := 0; i < len(letter); i++ {
            s += string(letter[i])
            backtrace(s, index+1)
            s = s[:len(s)-1]
        }
    }
    backtrace("", 0)
    return res
}
```

分析

```go

```

## [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

答案

```go
func combinationSum(candidates []int, target int) [][]int {
    res := [][]int{}
    trace := []int{}
    var backtrace func(start, sum int)
    backtrace = func(start, sum int) {
        if sum == target {
            tmp := make([]int, len(trace))    // 注意这里  测试案例输出：[[2,2,3],[7]]
            copy(tmp, trace)                // 必须创建一个用来拷贝的，使用copy函数
            res = append(res, tmp)
            return
        }
        if sum > target {    // 本题没有组合数量要求，仅仅是总和的限制，所以递归没有层数的限制，只要选取的元素总和超过target，就返回
            return
        }
        // start用来控制for循环的起始位置
        for i:=start; i<len(candidates); i++ {
            trace = append(trace, candidates[i])
            sum += candidates[i]
            backtrace(i, sum) // 本题元素为可重复选取的，所以关键点:不用i+1了，表示可以重复读取当前的数
            trace = trace[:len(trace)-1]
            sum -= candidates[i]
        }
    }
    backtrace(0, 0)
    return res
}
```

分析

```go
----------------------------------------------------------------------
func combinationSum(candidates []int, target int) [][]int {
    res := [][]int{}
    trace := []int{}
    var backtrace func(start, sum int)
    backtrace = func(start, sum int) {
        if sum == target {    
            res = append(res, trace)    // 测试案例输出：[[7,7,7],[7]]
            return
        }
----------------------------------------------------------------------

剪枝优化的思路：
    对candidates排序之后，如果下一层的sum（就是本层的 sum + candidates[i]）已经大于target，就可以结束本轮for循环的遍历。
```

## [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

给定一个候选人编号的集合 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的每个数字在每个组合中只能使用 **一次** 。

注意：解集不能包含重复的组合。

答案

```go
func combinationSum2(candidates []int, target int) [][]int {
    res := [][]int{}      // 存放组合集合
    trace := []int{}      // 符合条件的组合
    sort.Ints(candidates) // 首先把给candidates排序，让其相同的元素都挨在一起。
    var backtrace func(start, sum int)
    backtrace = func(start, sum int) {
        if sum == target {
            tmp := make([]int, len(trace))  // 注意长度设置
            copy(tmp, trace)
            res = append(res, tmp)
            return
        }
        if sum > target {
            return
        }

        for i := start; i < len(candidates); i++ {
            // 前一个树枝，使用了candidates[i - 1]，也就是说同一树层使用过candidates[i - 1]。
            // 这个判断是用来在集合内有重复元素且解集不包含重复组合的情况下进行去重的
            if i > start && candidates[i] == candidates[i-1] {
                continue
            }
            trace = append(trace, candidates[i])
            sum += candidates[i]
            backtrace(i+1, sum) // 和39.组合总和的区别，这里是i+1，每个数字在每个组合中只能使用一次
            trace = trace[:len(trace)-1]
            sum -= candidates[i]
        }
    }
    backtrace(0, 0)
    return res
}
```

分析

```
本题的难点在于：集合（数组candidates）有重复元素，但还不能有重复的组合
即要去重的是同一树层上的“使用过”。同一树枝上的都是一个组合里的元素，不用去重
```

## [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 回文串。返回 `s` 所有可能的分割方案。

答案

```go
func partition(s string) [][]string {
    res := [][]string{}
    trace := []string{}
    var backtrace func(start int)
    backtrace = func(start int) {
        if start == len(s) {    // 已经切到字符串的结尾位置了
            tmp := make([]string, len(trace))   // 注意长度
            copy(tmp, trace)
            res = append(res, tmp)
            return
        }

        for i := start; i < len(s); i++ { // 横向遍历：找切割线  切割到字符串的结尾位置
            if isPartition(s, start, i) { // 判断 s[start:i+1] 是否为回文串
                trace = append(trace, s[start:i+1])
            } else {
                continue
            }
            backtrace(i + 1) // i+1 表示下一轮递归遍历的起始位置
            trace = trace[:len(trace)-1]
        }
    }
    backtrace(0)
    return res
}

// 判断是否为回文
func isPartition(s string, startIndex, end int) bool {
    for startIndex < end {
        if s[startIndex] != s[end] {
            return false
        }
        //移动左右指针
        startIndex++
        end--
    }
    return true
}
```

分析

```

```

## [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

**有效 IP 地址** 由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。

- 例如：`"0.1.2.201"` 和 `"192.168.1.1"` 是 **有效** IP 地址，但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。

给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。

答案

```go
func restoreIpAddresses(s string) []string {
    var res, path []string
    var backtrace func(start int)
    backtrace = func(start int) {
        if start == len(s) && len(path) == 4 {
            tmpString := path[0] + "." + path[1] + "." + path[2] + "." + path[3]
            // fmt.Println("tmpString:", tmpString)
            res = append(res, tmpString)
        }
        for i := start; i < len(s); i++ {
            path = append(path, s[start:i+1])
            // fmt.Println("path:", path)
            if i-start+1 <= 3 && len(path) <= 4 && isIP(s, start, i) { 
                backtrace(i + 1)
            } else {    // 剪枝
                path = path[:len(path)-1] // 因为下面是return，所以要提前回溯
                return    // 直接返回 
            }
            path = path[:len(path)-1]
        }
    }
    backtrace(0)
    return res
}

// 判断字符串s在左闭右闭区间[start, end]所组成的数字是否合法
func isIP(s string, start int, end int) bool {
    check, err := strconv.Atoi(s[start : end+1])
    if err != nil { // 遇到非数字字符不合法
        return false
    }
    if end-start+1 > 1 && s[start] == '0' { // 0开头的两位、三位数字不合法
        return false
    }
    if check > 255 { // 大于255了不合法
        return false
    }
    return true
}
```

分析

```
注意：
0开头的数字不合法
有的题目不会写在上面
```

## [78. 子集](https://leetcode.cn/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

答案

```go
func subsets(nums []int) [][]int {
    res := [][]int{}
    trace := []int{}
    sort.Ints(nums)
    var backtrace func(start int)
    backtrace = func(start int) {
        tmp := make([]int, len(trace))
        copy(tmp, trace)
        res = append(res, tmp)
        for i := start; i < len(nums); i++ {
            trace = append(trace, nums[i])
            backtrace(i + 1) // 取过的元素不会重复取
            trace = trace[:len(trace)-1]
        }
    }
    backtrace(0)
    return res
}
```

分析

```
子集问题是找树的所有节点
求取子集问题，不需要任何剪枝！因为子集就是要遍历整棵树
```

## [90. 子集 II](https://leetcode.cn/problems/subsets-ii/)

给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的 子集（幂集）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。

答案

```go
func subsetsWithDup(nums []int) [][]int {
    res := [][]int{}
    trace := []int{}
    sort.Ints(nums)    // 去重需要排序
    var backtrace func(start int)
    backtrace = func(start int) {
        tmp := make([]int, len(trace))
        copy(tmp, trace)
        res = append(res, tmp)
        for i := start; i < len(nums); i++ {
            if i > start && nums[i] == nums[i-1] { // 对同一树层使用过的元素进行跳过
                continue
            }
            trace = append(trace, nums[i])
            backtrace(i + 1)
            trace = trace[:len(trace)-1]
        }
    }
    backtrace(0)
    return res
}
```

分析

```
关于回溯算法中的树层去重问题，在 40.组合总和II 中已经详细讲解过了，和本题是一个套路。
```

## [491. 递增子序列](https://leetcode.cn/problems/non-decreasing-subsequences/)

给你一个整数数组 `nums` ，找出并返回所有该数组中不同的递增子序列，递增子序列中 **至少有两个元素** 。你可以按 **任意顺序** 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

答案

```go
func findSubsequences(nums []int) [][]int {
    res := [][]int{}
    trace := []int{}    
    var backtrace func(start int)
    backtrace = func(start int) {
        if len(trace) > 1 {
            tmp := make([]int, len(trace))
            copy(tmp, trace)
            res = append(res, tmp)
            // 注意这里不要加return，因为要取树上的所有节点
        }
        // used数组放在backtrace中，所以只针对同一树层
        used := [201]int{} // 使用数组来进行去重操作，题目表明数值范围[-100, 100]
        for i := start; i < len(nums); i++ {
            if len(trace) > 0 && nums[i] < trace[len(trace)-1] || used[nums[i]+100] == 1 {
                continue // 非递增 或 同一树层使用过相同的数字，则跳过
            }
            used[nums[i]+100] = 1 // 记录这个元素在本层用过了，本层后面不能再用了
            trace = append(trace, nums[i])
            backtrace(i + 1)
            trace = trace[:len(trace)-1]
        }
    }
    backtrace(0)
    return res
}
```

分析

```
本题求自增子序列，是不能对原数组进行排序的，排完序的数组都是自增子序列了

同一父节点下的同层上使用过的元素就不能再使用了
```

## [46. 全排列](https://leetcode.cn/problems/permutations/)

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

答案

```go
func permute(nums []int) [][]int {
    res := [][]int{}
    trace := []int{}
    used := [21]int{}    // 这里的used放在backtrace外面，所以是针对整棵树的去重
    var backtrace func()
    backtrace = func() {
        if len(trace) == len(nums) {
            tmp := make([]int, len(trace))
            copy(tmp, trace)
            res = append(res, tmp) // 如果这里是res = append(res, trace)，则res里的每个值会随着trace的变化而变化
            return
        }
        for i := 0; i < len(nums); i++ {
            if used[nums[i]+10] == 0 { // 因为 nums[i] 为 -10 ~ 10
                trace = append(trace, nums[i])
                used[nums[i]+10] = 1
                backtrace()
                trace = trace[:len(trace)-1] // 回溯时要消除之前的影响
                used[nums[i]+10] = 0
            }
        }
    }
    backtrace()
    return res
}
```

分析

```
处理排列问题就不用使用startIndex
但排列问题需要一个used数组，标记已经选择的元素
当收集元素的数组path的大小达到和nums数组一样大的时候，说明找到了一个全排列，也表示到达了叶子节点
```

## [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

给定一个可包含重复数字的序列 `nums` ，***按任意顺序*** 返回所有不重复的全排列。

答案

```go
func permuteUnique(nums []int) [][]int {
    res := [][]int{}
    trace := []int{}
    used := [10]int{}
    sort.Ints(nums)
    var backtrace func()
    backtrace = func() {
        if len(trace) == len(nums) {
            tmp := make([]int, len(trace))
            copy(tmp, trace)
            res = append(res, tmp) // 如果这里是res = append(res, trace)，则res里的每个值会随着trace的变化而变化
            return
        }
        for i := 0; i < len(nums); i++ {
            if i > 0 && nums[i] == nums[i-1] && used[i-1] == 0 { // 对树层中前一位去重，used=0表示已经切换到同一树层的新的树枝了
                continue // 要对树层中前一位去重，used[i-1]=0；要对树枝前一位去重，used[i-1]=1
            }
            if used[i] == 0 {
                used[i] = 1
                trace = append(trace, nums[i])
                backtrace()
                trace = trace[:len(trace)-1] // 回溯时要消除之前的影响
                used[i] = 0
            }
        }
    }
    backtrace()
    return res
}
```

分析

![img](https://code-thinking-1253855093.file.myqcloud.com/pics/20201124201331223.png)

```
一般来说：组合问题和排列问题是在树形结构的叶子节点上收集结果，而子集问题就是取树上所有节点的结果。

使用到nums[i] == nums[i-1]时，要对数组进行排序：sort.Ints(nums)

注意，换到另一树枝上时，used数组其实会清空（父节点以上不清空）
所以 i>0 && nums[i]==nums[i-1] && used[i-1]==0 表明切换到另一条树枝了，前一条树枝已经用了i-1和i
```

## [51. N 皇后](https://leetcode.cn/problems/n-queens/)

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

答案

```go
func solveNQueens(n int) [][]string {
    res := [][]string{}
    chessboard := make([][]string, n)
    for i := 0; i < n; i++ {
        chessboard[i] = make([]string, n)
        for j := 0; j < n; j++ {
            chessboard[i][j] = "."
        }
    }

    var backtrace func(row int)
    backtrace = func(row int) { // n是棋盘的大小，用row来记录当前遍历到棋盘的第几层
        if row == n {
            tmp := make([]string, n)
            for i, rowStr := range chessboard {
                tmp[i] = strings.Join(rowStr, "") // 将rowStr中的子串连接成一个单独的字符串，子串之间用""分隔
            }
            res = append(res, tmp)
            return
        }
        for i := 0; i < n; i++ { // 在第row行，第i列放皇后  每次都是要从新的一行的起始位置开始搜，所以都是从0开始
            if isValidForQueens(n, row, i, chessboard) {
                chessboard[row][i] = "Q" // 放置皇后
                backtrace(row + 1)
                chessboard[row][i] = "." // 回溯，撤销皇后
            }
        }
    }
    backtrace(0)
    return res
}

func isValidForQueens(n, row, col int, chessboard [][]string) bool {
    // 3个判断都进行了剪枝  因为大于row的行还没有处理，所以不可能有皇后
    // 检查列（正上方）
    for i := 0; i < row; i++ {
        if chessboard[i][col] == "Q" { // 行不固定，列固定，遍历检查
            return false
        }
    }
    // 检查 45度角是否有皇后（左上角）
    for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
        if chessboard[i][j] == "Q" {
            return false
        }
    }
    // 检查 135度角是否有皇后（右上角）
    for i, j := row-1, col+1; i >= 0 && j < n; i, j = i-1, j+1 {
        if chessboard[i][j] == "Q" {
            return false
        }
    }
    return true
}
```

分析

https://phpmianshi.com/?id=1947

```go
tmp[i] = strings.Join(rowStr, "")    // 将rowStr中的子串连接成一个单独的字符串，子串之间用""分隔

二维矩阵中矩阵的高就是这棵树的高度，矩阵的宽就是树形结构中每一个节点的宽度
用皇后们的约束条件，来回溯搜索这棵树，只要搜索到了树的叶子节点，说明就找到了皇后们的合理位置了
```

## [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)

编写一个程序，通过填充空格来解决数独问题。

数独的解法需 **遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）

数独部分空格内已填入了数字，空白格用 `'.'` 表示。

答案

```go
func solveSudoku(board [][]byte) {
    var dfs func(board [][]byte) bool
    dfs = func(board [][]byte) bool {
        for i := 0; i < 9; i++ { // 遍历行
            for j := 0; j < 9; j++ { // 遍历列
                // 判断此位置是否适合填数字
                if board[i][j] != '.' {
                    continue
                }

                // 尝试填1-9
                for k := '1'; k <= '9'; k++ {
                    if isvalid(i, j, byte(k), board) == true { // (i, j) 这个位置放k是否合适
                        board[i][j] = byte(k)   // 放置k
                        if dfs(board) == true { // 如果找到合适一组立刻返回
                            return true
                        }
                        board[i][j] = '.' // 回溯，撤销k
                    }
                }
                return false // 9个数都试完了，都不行，那么就返回false
            }
        }
        return true // 遍历完没有返回false，说明找到了合适棋盘位置了
    }
    dfs(board)
}

// 判断填入数字是否满足要求
func isvalid(row, col int, k byte, board [][]byte) bool {
    for i := 0; i < 9; i++ { // 判断行里是否重复
        if board[row][i] == k {
            return false
        }
    }
    for i := 0; i < 9; i++ { // 判断列里是否重复
        if board[i][col] == k {
            return false
        }
    }
    startrow := (row / 3) * 3
    startcol := (col / 3) * 3
    for i := startrow; i < startrow+3; i++ { // 判断9方格里是否重复
        for j := startcol; j < startcol+3; j++ {
            if board[i][j] == k {
                return false
            }
        }
    }
    return true
}
```

分析

```
递归函数的返回值需要是bool类型，因为解数独找到一个符合的条件（就在树的叶子节点上）立刻就返回，相当于找从根节点到叶子节点一条唯一路径，所以需要使用bool返回值。

本题递归不用终止条件，解数独是要遍历整个树形结构寻找可能的叶子节点就立刻返回

一个for循环遍历棋盘的行，一个for循环遍历棋盘的列，一行一列确定下来之后，递归遍历这个位置放9个数字的可能性
```

## [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

答案

```go
func generateParenthesis(n int) []string {
    res := make([]string, 0)
    var dfs func(left, right int,  path string)
    dfs = func(left, right int,  path string) {
        if left == n && right == n {
            res = append(res, path)
            return
        }
        if left < n {
            dfs(left+1, right, path+"(")    // path+"(" 这样就不用手动回溯
        }
        if right < left {
            dfs(left, right+1, path+")")
        }
    }
    dfs(0, 0, "")
    return res
}
```

分析

```
https://leetcode.cn/problems/generate-parentheses/solutions/938191/shen-du-you-xian-bian-li-zui-jian-jie-yi-ypti/?envType=study-plan-v2&envId=top-100-liked

一个合法的括号序列需要满足两个条件：
1、左右括号数量相等
2、任意前缀中左括号数量 >= 右括号数量 （也就是说每一个右括号总能找到相匹配的左括号）

使用深度优先搜索，将搜索顺序定义为枚举序列的每一位填什么
```

## [79. 单词搜索](https://leetcode.cn/problems/word-search/)

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

答案

```go
type pair struct{ x, y int }

var directions = []pair{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} // 上下左右

func exist(board [][]byte, word string) bool {
    h, w := len(board), len(board[0])
    vis := make([][]bool, h)
    for i := range vis {
        vis[i] = make([]bool, w)
    }
    var dfs func(i, j, k int) bool
    dfs = func(i, j, k int) bool {
        if board[i][j] != word[k] { // 剪枝：当前字符不匹配
            return false
        }
        if k == len(word)-1 { // 单词存在于网格中
            return true
        }
        vis[i][j] = true
        for _, dir := range directions {
            if newI, newJ := i+dir.x, j+dir.y; 0 <= newI && newI < h && 0 <= newJ && newJ < w && !vis[newI][newJ] {
                if dfs(newI, newJ, k+1) {
                    vis[i][j] = false    // 回溯时还原已访问的单元格
                    return true
                }
            }
        }
        vis[i][j] = false    // 回溯时还原已访问的单元格
        return false
    }
    for i, row := range board {
        for j := range row {
            if dfs(i, j, 0) {
                return true
            }
        }
    }
    return false
}
```

分析

```

```

# 贪心

## [455. 分发饼干](https://leetcode.cn/problems/assign-cookies/)

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 `i`，都有一个胃口值 `g[i]`，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 `j`，都有一个尺寸 `s[j]` 。如果 `s[j] >= g[i]`，我们可以将这个饼干 `j` 分配给孩子 `i` ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

答案

```go
func findContentChildren(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    res := 0
    child := len(g) - 1
    for i := len(s) - 1; i >= 0 && child >= 0; child-- {
        if s[i] >= g[child] {
            res++
            i--
        }
    }
    return res
}
```

分析

```go
想清楚局部最优，想清楚全局最优，感觉局部最优是可以推出全局最优，并想不出反例，那么就试一试贪心

这里的局部最优就是大饼干喂给胃口大的，充分利用饼干尺寸喂饱一个，全局最优就是喂饱尽可能多的小孩。

先将饼干数组和小孩数组排序，然后从后向前遍历小孩数组，用大饼干优先满足胃口大的，并统计满足小孩数量。
```

## [376. 摆动序列](https://leetcode.cn/problems/wiggle-subsequence/)

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 **摆动序列 。**第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

- 例如， `[1, 7, 4, 9, 2, 5]` 是一个 **摆动序列** ，因为差值 `(6, -3, 5, -7, 3)` 是正负交替出现的。

- 相反，`[1, 4, 7, 2, 5]` 和 `[1, 7, 4, 5, 5]` 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

**子序列** 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 `nums` ，返回 `nums` 中作为 **摆动序列** 的 **最长子序列的长度** 。

答案

```go
func wiggleMaxLength(nums []int) int {
    count, preDiff, curDiff := 1, 0, 0 // 序列默认序列最右边有一个峰值
    if len(nums) < 2 {
        return 1
    }
    for i := 0; i < len(nums)-1; i++ {
        curDiff = nums[i+1] - nums[i]
        // 如果有正有负则更新下标值||或者只有前一个元素为0（针对两个不等元素的序列也视作摆动序列，且摆动长度为2）
        if (curDiff > 0 && preDiff <= 0) || (preDiff >= 0 && curDiff < 0) {
            preDiff = curDiff
            count++ // 统计数组的峰值数量    相当于是删除单一坡度上的节点，然后统计长度
        }
    }
    return count
}
```

分析

```go
在计算是否有峰值的时候，根据遍历下标 i ，
计算 prediff（nums[i] - nums[i-1]） 和 curdiff（nums[i+1] - nums[i]），
如果prediff < 0 && curdiff > 0 或者 prediff > 0 && curdiff < 0 此时就有波动就需要统计
```

## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**是数组中的一个连续部分。

答案

```go
func maxSubArray(nums []int) int {
    length := len(nums)
    if length == 1 {
        return nums[0]
    }
    res, sum := nums[0], nums[0]
    for i := 1; i < length; i++ {
        if sum < 0 { // 相当于重置最大子序起始位置，因为sum是负数时继续累加下去也一定是拉低新的总和
            sum = nums[i]
        } else {
            sum += nums[i] // 取区间累计的最大值（相当于不断确定最大子序终止位置）
        }
        res = max(res, sum)
    }
    return res
}
```

分析

```go
局部最优：当前“连续和”为负数的时候立刻放弃，从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会越来越小。

全局最优：选取最大“连续和”

局部最优的情况下，并记录最大的“连续和”，可以推出全局最优。
```

## [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

给你一个整数数组 `prices` ，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回 *你能获得的 **最大** 利润* 。

答案

```go
func maxProfit(prices []int) int {
    sum := 0
    for i := 1; i < len(prices); i++ { // 第一天没有利润，至少要第二天才会有利润
        if prices[i]-prices[i-1] > 0 {
            sum += prices[i] - prices[i-1]
        }
    }
    return sum
}
```

分析

```go
把利润分解为每天为单位的维度

局部最优：收集每天的正利润，全局最优：求得最大利润

假如第 0 天买入，第 3 天卖出，那么利润为：prices[3] - prices[0]。
相当于(prices[3] - prices[2]) + (prices[2] - prices[1]) + (prices[1] - prices[0])。
```

## [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

答案

```go
func canJump(nums []int) bool {
    mx := 0    // 可以跳跃的最大长度
    for i, num := range nums {
        if i > mx {
            return false
        }
        mx = max(mx, i+num)
    }
    return true
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
不用拘泥于每次究竟跳几步，而是看覆盖范围，覆盖范围内一定是可以跳过来的，不用管是怎么跳的。
```

## [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

答案

```go
func jump(nums []int) int {
    curMax := 0                   // 当前覆盖的最远距离下标
    ans := 0                           // 记录走的最大步数
    nextMax := 0                  // 下一步覆盖的最远距离下标
    for i := 0; i < len(nums)-1; i++ { // 注意这里是小于nums.size() - 1，这是关键所在
        if nums[i]+i > nextMax {
            nextMax = nums[i] + i // 更新下一步覆盖的最远距离下标
        }
        if i == curMax { // 遇到当前覆盖的最远距离下标，直接步数加一
            curMax = nextMax // 更新当前覆盖的最远距离下标
            ans++
        }
    }
    return ans
}
```

分析

```go
贪心的思路，局部最优：当前可移动距离尽可能多走，如果还没到终点，步数再加一。整体最优：一步尽可能多走，从而达到最小步数。

所以真正解题的时候，要从覆盖范围出发，不管怎么跳，覆盖范围内一定是可以跳到的，以最小的步数增加覆盖范围，覆盖范围一旦覆盖了终点，得到的就是最小步数

如果移动下标达到了当前这一步的最大覆盖最远距离了，还没有到终点的话，那么就必须再走一步来增加覆盖范围，直到覆盖范围覆盖了终点。    
```

## [1005. K 次取反后最大化的数组和](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)

给你一个整数数组 `nums` 和一个整数 `k` ，按以下方法修改该数组：

- 选择某个下标 `i` 并将 `nums[i]` 替换为 `-nums[i]` 。

重复这个过程恰好 `k` 次。可以多次选择同一个下标 `i` 。

以这种方式修改数组后，返回数组 **可能的最大和** 。

答案

```go
func largestSumAfterKNegations(nums []int, K int) int {
    // 将数组按照绝对值大小从大到小排序
    sort.Slice(nums, func(i, j int) bool {
        return math.Abs(float64(nums[i])) > math.Abs(float64(nums[j]))
    })
    // 从前向后遍历，遇到负数将其变为正数
    for i := 0; i < len(nums); i++ {
        if K > 0 && nums[i] < 0 {
            K--
            nums[i] = -nums[i]
        }
    }
    // 如果K还大于0，那么反复转变数值最小的元素，将K用完
    if K%2 == 1 {
        nums[len(nums)-1] *= -1
    }
    // 求和
    res := 0
    for i := 0; i < len(nums); i++ {
        res += nums[i]
    }
    return res
}
```

分析

```go

```

## [738. 单调递增的数字](https://leetcode.cn/problems/monotone-increasing-digits/)

当且仅当每个相邻位数上的数字 `x` 和 `y` 满足 `x <= y` 时，我们称这个整数是**单调递增**的。

给定一个整数 `n` ，返回 *小于或等于 `n` 的最大数字，且数字呈 **单调递增*** 。

答案

```go
func monotoneIncreasingDigits(N int) int {
    //将数字转为字符串，方便使用下标
    s := strconv.Itoa(N)
    //将字符串转为byte数组，方便更改
    ss := []byte(s)
    n := len(ss)
    if n < 2 {
        return N
    }
    for i := n - 1; i > 0; i-- {    // 从后向前遍历
        if ss[i-1] > ss[i] { //前一个大于后一位,前一位减1，后面的全部置为9
            ss[i-1] -= 1
            for j := i; j < n; j++ { //后面的全部置为9
                ss[j] = '9'
            }
        }
    }
    res, _ := strconv.Atoi(string(ss))
    return res
}
```

分析

```go

```

## [763. 划分字母区间](https://leetcode.cn/problems/partition-labels/)

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。

答案

```go
func partitionLabels(s string) []int {
    partition := make([]int, 0)
    lastPos := [26]int{}    // 储存每个字母最后一次出现的下标位置
    for i, c := range s {
        lastPos[c-'a'] = i    // 记录一遍
    }
    start, end := 0, 0    // 当前片段的开始下标和结束下标
    for i, c := range s {
        if lastPos[c-'a'] > end {    // 遍历到的字母的最后出现位置大于当前片段的结束下标
            end = lastPos[c-'a']    // 则需要向后移动当前片段的结束下标
        }
        if i == end {    // 遍历到当前片段的结束下标
            partition = append(partition, end-start+1)    // 一个片段结束了，添加到结果集中
            start = end + 1    // 重置片段
        }
    }
    return partition
}
```

分析

```go
在得到每个字母最后一次出现的下标位置之后，可以使用贪心的方法将字符串划分为尽可能多的片段，具体做法如下。

从左到右遍历字符串，遍历的同时维护当前片段的开始下标 start 和结束下标 end，初始时 start=end=0。
对于每个访问到的字母 c，得到当前字母的最后一次出现的下标位置 endc，则当前片段的结束下标一定不会小于 endc
当访问到下标 end 时，当前片段访问结束，当前片段的下标范围是 [start,end]，长度为 end−start+1，将当前片段的长度添加到返回值，然后令 start=end+1，继续寻找下一个片段。
重复上述过程，直到遍历完字符串。
```

# 动态规划

## [509. 斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

**斐波那契数** （通常用 `F(n)` 表示）形成的序列称为 **斐波那契数列** 。该数列由 `0` 和 `1` 开始，后面的每一项数字都是前面两项数字的和。也就是：

F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1

给定 `n` ，请计算 `F(n)` 。

答案

```go
func fib(n int) int {
    if n < 2 {
        return n
    }
    a, b, c := 0, 1, 1
    for i := 2; i <= n; i++ {
        c = a + b
        a, b = b, c
    }
    return c
}
```

分析

```go
状态转移方程 dp[i] = dp[i - 1] + dp[i - 2]

初始化：dp[0] = 0;    dp[1] = 1
```

## [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

给定一个非负整数 `numRows`，*生成「杨辉三角」的前 `numRows`* 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。

答案

```go
func generate(numRows int) [][]int {
    ans := make([][]int, numRows)
    for i := range ans {
        ans[i] = make([]int, i+1)
        ans[i][0] = 1
        ans[i][i] = 1
        for j := 1; j < i; j++ {
            ans[i][j] = ans[i-1][j-1] + ans[i-1][j]
        }
    }
    return ans
}
```

分析

```go

```

## [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

答案

```go
func climbStairs(n int) int {
    if n < 3 {
        return n
    }
    a, b, c := 1, 2, 3
    for i := 3; i <= n; i++ { // 如果从i=2开始遍历，那么判断条件要去掉=
        c = a + b
        a, b = b, c
    }
    return c
}
```

分析

```go
dp[i] = dp[i - 1] + dp[i - 2]

初始化：dp[1] = 1，dp[2] = 2
```

## [746. 使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

给你一个整数数组 `cost` ，其中 `cost[i]` 是从楼梯第 `i` 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 `0` 或下标为 `1` 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。

答案

```go
func minCostClimbingStairs(cost []int) int {
    a, b, c := 0, 0, 0
    for i := 2; i <= len(cost); i++ {
        c = min(a+cost[i-2], b+cost[i-1])
        a, b = b, c
    }
    return c
}
```

分析

```go
dp[i]的定义：到达第i台阶所花费的最少体力为dp[i]

dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

初始化 ：dp[0] = 0，dp[1] = 0
```

## [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

答案

```go
// 动态规划
func uniquePaths(m int, n int) int {
    dp := make([][]int, m)
    for i:=0; i<m; i++ {
        dp[i] = make([]int, n)
        dp[i][0] = 1
    }
    for j:=0; j<n; j++ {
        dp[0][j] = 1
    }
    for i:=1; i<m; i++ {
        for j:=1; j<n; j++ {
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
        }
    }
    return dp[m-1][n-1]
}


// 数论
func uniquePaths(m int, n int) int {
    a := 1     // 分子  从m+n-2开始累乘m-1个数
    b := m - 1 // 分母 从 1 累乘到 m-1
    t := m + n - 2
    for i := m - 1; i > 0; i-- {
        a *= t
        t--
        for b != 0 && a%b == 0 {
            a /= b
            b--
        }
    }
    return a
}
```

分析

```go
dp[i][j] ：表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径

dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

初始化：dp[i][0]一定都是1，因为从(0, 0)的位置到(i, 0)的路径只有一条，那么dp[0][j]也同理。
for (int i = 0; i < m; i++) dp[i][0] = 1;
for (int j = 0; j < n; j++) dp[0][j] = 1;
```

## [63. 不同路径 II](https://leetcode.cn/problems/unique-paths-ii/)

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

答案

```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
    m, n := len(obstacleGrid), len(obstacleGrid[0])
    dp := make([][]int, m)
    for i := 0; i < m; i++ {
        dp[i] = make([]int, n)
    }
    for i := 0; i < m && obstacleGrid[i][0] == 0; i++ { // 障碍之后的dp还是初始值0
        dp[i][0] = 1
    }
    for j := 0; j < n && obstacleGrid[0][j] == 0; j++ { // 障碍之后的dp还是初始值0
        dp[0][j] = 1
    }
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if obstacleGrid[i][j] != 1 {
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            }
        }
    }
    return dp[m-1][n-1]
}
```

分析

```go
dp[i][j] ：表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径

dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
(i, j)如果就是障碍的话应该就保持初始状态（初始状态为0）

初始化：
因为从(0, 0)的位置到(i, 0)的路径只有一条，所以dp[i][0]一定为1，dp[0][j]也同理。
但如果(i, 0) 这条边有了障碍之后，障碍之后（包括障碍）都是走不到的位置了，所以障碍之后的dp[i][0]应该还是初始值0
下标(0, j)的初始化情况同理
```

## [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

给定一个包含非负整数的 `m x n` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

答案

```go
func minPathSum(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    /*
    dp := make([][]int, m)
    for i := 0; i < m; i++ {
        dp[i] = make([]int, n)
    }
    */
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if i == 0 && j == 0 {    // 当左边和上边都是矩阵边界时
                continue
            } else if i == 0 {    // 当只有上边是矩阵边界
                grid[0][j] += grid[0][j-1]
            } else if j == 0 {    // 当只有左边是矩阵边界
                grid[i][0] += grid[i-1][0]
            } else {    // 当左边和上边都不是矩阵边界
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
            }

        }
    }
    return grid[m-1][n-1]
}
```

分析

```go
设 dp 为大小 m×n 矩阵，其中 dp[i][j] 的值代表直到走到 (i,j) 的最小路径和。

当前单元格 (i,j) 只能从左方单元格 (i−1,j) 或上方单元格 (i,j−1) 走到，因此只需要考虑矩阵左边界和上边界
走到当前单元格 (i,j) 的最小路径和 = “从左方单元格 (i−1,j) 与 从上方单元格 (i,j−1) 走来的两个最小路径和中较小的 ” + 当前单元格值 grid[i][j]

其实我们完全不需要建立 dp 矩阵浪费额外空间，直接遍历 grid[i][j] 修改即可。
这是因为：grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j] ；原 grid 矩阵元素中被覆盖为 dp 元素后（都处于当前遍历点的左上方），不会再被使用到。
```

## [343. 整数拆分](https://leetcode.cn/problems/integer-break/)

给定一个正整数 `n` ，将其拆分为 `k` 个 **正整数** 的和（ `k >= 2` ），并使这些整数的乘积最大化。

返回 *你可以获得的最大乘积* 。

答案

```go
func integerBreak(n int) int {
    dp := make([]int, n+1)
    dp[2] = 1
    for i:=3; i<=n; i++ {
        for j:=1; j<=i-2; j++ {
            dp[i] = max(dp[i], max(j*(i-j), dp[i-j]*j))
        }
    }
    return dp[n]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i]：分拆数字i，可以得到的最大乘积为dp[i]

递推公式：dp[i] = max({dp[i], (i - j) * j, dp[i - j] * j})

初始化：dp[2] = 1，从dp[i]的定义来说，拆分数字2，得到的最大乘积是1
-------------------------------
数学证明方法：尽量分成3，不然就分2
```

## [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

答案

```go
func numTrees(n int)int{
    dp := make([]int, n+1)
    // dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]
    dp[0] = 1
    for i:=1; i<=n; i++ {
        for j:=1; j<=i; j++ {
            // j-1 ： 以j为头结点时，左子树的数量（一共有i个节点，有j-1个小于j的节点）
            // i-j ： 以j为头结点时，右子树的数量（一共有i个节点，有i-j个大于j的节点）
            dp[i] += dp[j-1] * dp[i-j]
        }
    }
    return dp[n]
}
```

分析

```go
给定一个有序序列 1⋯n，为了构建出一棵二叉搜索树，我们可以遍历每个数字 i，将该数字作为树根，
将 1⋯(i−1) 序列作为左子树，将 (i+1)⋯n 序列作为右子树。接着我们可以按照同样的方式递归构建左子树和右子树。

dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]

递推公式：dp[i] += dp[j - 1] * dp[i - j]
j-1 为j为头结点左子树节点数量，i-j 为以j为头结点右子树节点数量

dp[0] = 1
```

## [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

答案

```go
func canPartition(nums []int) bool {
    n := len(nums)
    sum := 0
    for i := 0; i < n; i++ {
        sum += nums[i]
    }
    if sum%2 == 1 {
        return false
    } else {
        sum /= 2
    }

    dp := make([]int, sum+1)
    for i := 0; i < n; i++ {
        for j := sum; j >= nums[i]; j-- { // 每一个元素一定是不可重复放入，所以从大到小遍历
            dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
        }
    }
    return dp[sum] == sum // 集合中的元素正好可以凑成总和target
}
```

分析

```go
dp[j]表示 背包总容量（所能装的总重量）是j，放进物品后，背的最大重量为dp[j]。
本题中每一个元素的数值既是容量，也是价值，所以不会有价值超过容量的情况。所以dp[j] <= j

dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);

dp[0]一定是0。
如果题目给的价值都是正整数那么非0下标都初始化为0就可以了，如果题目给的价值有负数，那么非0下标就要初始化为负无穷
```

## [1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)

有一堆石头，用整数数组 `stones` 表示。其中 `stones[i]` 表示第 `i` 块石头的重量。

每一回合，从中选出**任意两块石头**，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y-x`。

最后，**最多只会剩下一块** 石头。返回此石头 **最小的可能重量** 。如果没有石头剩下，就返回 `0`。

答案

```go
func lastStoneWeightII(stones []int) int {
    n := len(stones)
    sum := 0
    target := 0
    for i := 0; i < n; i++ {
        sum += stones[i]
    }
    target = sum / 2 // target总是较小的，且dp[target]<=target

    dp := make([]int, sum+1)
    for i := 0; i < n; i++ {
        for j := target; j >= stones[i]; j-- { // 每一个元素一定是不可重复放入，所以从大到小遍历
            dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
        }
    }
    return sum - dp[target] - dp[target]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[j]表示容量（这里说容量更形象，其实就是重量）为j的背包，最多可以背最大重量为dp[j]
本题中，石头的重量是 stones[i]，石头的价值也是 stones[i] ，可以 “最多可以装的价值为 dp[j]” == “最多可以背的重量为dp[j]“

dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])

重量都不会是负数，所以dp[j]都初始化为0
```

## [494. 目标和](https://leetcode.cn/problems/target-sum/)

给你一个非负整数数组 `nums` 和一个整数 `target` 。

向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** ：

- 例如，`nums = [2, 1]` ，可以在 `2` 之前添加 `'+'` ，在 `1` 之前添加 `'-'` ，然后串联起来得到表达式 `"+2-1"` 。

返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

答案

```go
func findTargetSumWays(nums []int, target int) int {
    n := len(nums)
    sum := 0
    for i:=0; i<n; i++ {
        sum += nums[i]
    }
    x := (sum+target)/2
    if (sum+target)%2 == 1 || abs(target)>sum {
        return 0
    }
    dp := make([]int, x+1)
    dp[0] = 1
    for i:=0; i<n; i++ {
        for j:=x; j>=nums[i]; j-- {
            dp[j] += dp[j-nums[i]]
        }
    }
    return dp[x]
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// 求两个数中的较小者
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}

// 求绝对值
func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```

分析

```go
dp[j] 表示：填满j（包括j）这么大容积的包，有dp[j]种方法

只要搞到nums[i]，凑成dp[j]就有dp[j - nums[i]] 种方法

dp[0] 为 1
```

## [474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/)

给你一个二进制字符串数组 `strs` 和两个整数 `m` 和 `n` 。

请你找出并返回 `strs` 的最大子集的长度，该子集中 **最多** 有 `m` 个 `0` 和 `n` 个 `1` 。

如果 `x` 的所有元素也是 `y` 的元素，集合 `x` 是集合 `y` 的 **子集** 。

答案

```go
func findMaxForm(strs []string, m int, n int) int {
    one, zero := 0, 0
    dp := make([][]int, m+1)
    for i := 0; i <= m; i++ {
        dp[i] = make([]int, n+1)
    }
    for i := 0; i < len(strs); i++ { // 遍历物品
        zero, one = 0, 0 // 物品i中0和1的数量
        for _, v := range strs[i] {
            if v == '0' {
                zero++
            }
        }
        one = len(strs[i]) - zero
        for j := m; j >= zero; j-- { // 遍历背包容量且从后向前遍历！
            for k := n; k >= one; k-- {
                dp[j][k] = max(dp[j][k], dp[j-zero][k-one]+1)
            }
        }
    }
    return dp[m][n]
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp[i][j]

dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)
这就是一个典型的01背包！ 只不过物品的重量有了两个维度而已

物品价值不会是负数，初始为0，保证递推的时候dp[i][j]不会被初始值覆盖
```

## [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

给你一个整数数组 `coins` 表示不同面额的硬币，另给一个整数 `amount` 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 `0` 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。

答案

```go
func change(amount int, coins []int) int {
    n := len(coins)
    dp := make([]int, amount+1)
    dp[0] = 1
    for i := 0; i < n; i++ { // 遍历物品
        for j := coins[i]; j <= amount; j++ { // 遍历背包
            dp[j] += dp[j-coins[i]]
        }
    }
    return dp[amount]
}
```

分析

```go
dp[j]：凑成总金额j的货币组合数为dp[j]

dp[j] 就是所有的dp[j - coins[i]]（考虑coins[i]的情况）相加。
所以递推公式：dp[j] += dp[j - coins[i]]
求装满背包有几种方法，公式都是：dp[j] += dp[j - nums[i]]

dp[0] = 1是 递归公式的基础。如果dp[0] = 0 的话，后面所有推导出来的值都是0

如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。
```

## [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)

给你一个由 **不同** 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

答案

```go
func combinationSum4(nums []int, target int) int {
    n := len(nums)
    dp := make([]int, target+1)
    dp[0] = 1
    for i := 0; i <= target; i++ { // 遍历背包
        for j := 0; j < n; j++ { // 遍历物品
            if i >= nums[j] {
                dp[i] += dp[i-nums[j]]
            }
        }
    }
    return dp[target]
}
```

分析

```go
dp[i]: 凑成目标正整数为i的排列个数为dp[i]

dp[i]（考虑nums[j]）可以由 dp[i - nums[j]]（不考虑nums[j]） 推导出来。
因为只要得到nums[j]，排列个数dp[i - nums[j]]，就是dp[i]的一部分。
求装满背包有几种方法，递推公式一般都是dp[i] += dp[i - nums[j]];

因为递推公式dp[i] += dp[i - nums[j]]的缘故，dp[0]要初始化为1，这样递归其他dp[i]的时候才会有数值基础。
至于dp[0] = 1 有没有意义呢？其实没有意义，仅仅是为了推导递推公式。
至于非0下标的dp[i]应该初始为多少呢？初始化为0，这样才不会影响dp[i]累加所有的dp[i - nums[j]]。
```

[https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#%E6%80%9D%E8%B7%AF](https://programmercarl.com/0377.组合总和Ⅳ.html#思路)

<img src="https://img-blog.csdnimg.cn/20210131174250148.jpg" alt="377.组合总和Ⅳ" style="zoom:50%;" />

```go
主要是这张图很好
不知道怎么推导dp计算过程的时候
可以参考一下这张图
-------------------------------
内外循环怎么决定也可以看链接里的分析
排列是物品可以打乱的，所以物品只能放到内循环里
```

## [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

答案

```go
func coinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for j := 1; j <= amount; j++ { // 遍历背包
        dp[j] = math.MaxInt32
        for i := 0; i < len(coins); i++ { // 遍历物品
            if j >= coins[i] {
                dp[j] = min(dp[j], dp[j-coins[i]]+1)
            }
        }
    }
    if dp[amount] == math.MaxInt32 {
        return -1
    } else {
        return dp[amount]
    }
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// 求两个数中的较小者
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

分析

```go
dp[j]：凑足总额为j所需钱币的最少个数为dp[j]

凑足总额为j - coins[i]的最少个数为dp[j - coins[i]]，那么只需要加上一个钱币coins[i]即dp[j - coins[i]] + 1就是dp[j]（考虑coins[i]），所以dp[j] 要取所有 dp[j - coins[i]] + 1 中最小的。
递推公式：dp[j] = min(dp[j - coins[i]] + 1, dp[j]);

凑足总金额为0所需钱币的个数一定是0，那么dp[0] = 0
考虑到递推公式的特性，dp[j]必须初始化为一个最大的数，否则就会在min(dp[j - coins[i]] + 1, dp[j])比较的过程中被初始值覆盖。所以下标非0的元素都是应该是最大值

--------------------------------------------------------------------------------------
dp[j] 要取所有 dp[j - coins[i]] + 1 中最小的，所以用min()
这里dp[j]是会不断赋值更新的，要求其中最小的，所以不能是dp[j] = dp[j-coins[i]]+1

本题求钱币最小个数，那么钱币有顺序和没有顺序都可以，都不影响钱币的最小个数。
所以本题并不强调集合是组合还是排列。
如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。

本题钱币数量可以无限使用，那么是完全背包。所以遍历的内循环是正序
```

## [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/description/)

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

答案

```go
func numSquares(n int) int {
    dp := make([]int, n+1)
    for j := 1; j <= n; j++ { // 遍历背包
        dp[j] = math.MaxInt32
        for i := 1; i*i <= j; i++ {    // 遍历物品
            if j >= i*i {
                dp[j] = min(dp[j], dp[j-i*i]+1)
            }
        }
    }
    return dp[n]
}

// 求两个数中的较小者
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

分析

```go
dp[j]：和为j的完全平方数的最少数量为dp[j]

dp[j] 可以由dp[j - i * i]推出， dp[j - i * i] + 1 便可以凑成dp[j]。
此时我们要选择最小的dp[j]，所以递推公式：dp[j] = min(dp[j - i * i] + 1, dp[j])

dp[0]表示 和为0的完全平方数的最小数量，那么dp[0]一定是0
从递归公式dp[j] = min(dp[j - i * i] + 1, dp[j]);中可以看出每次dp[j]都要选最小的，所以非0下标的dp[j]一定要初始为最大值，这样dp[j]在递推的时候才不会被初始值覆盖
```

## [198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

答案

```go
func rob(nums []int) int {
    n := len(nums)
    if n == 1 {
        return nums[0]
    }
    if n == 2 {
        return max(nums[0], nums[1])
    }
    dp := make([]int, n)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i:=2; i<n; i++ {
        dp[i] = max(dp[i-2]+nums[i], dp[i-1])
    }
    return dp[n-1]
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i]：考虑下标i（包括i）以内的房屋，最多可以偷窃的金额为dp[i]

dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
如果偷第i房间，那么dp[i] = dp[i - 2] + nums[i];
如果不偷第i房间，那么dp[i] = dp[i - 1]，即考 虑i-1房，（注意这里是考虑，并不是一定要偷i-1房，这是很多同学容易混淆的点）

初始化：dp[0] 一定是 nums[0]，dp[1]就是nums[0]和nums[1]的最大值
```

## [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/description/)

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

答案

```go
func rob(nums []int) int {
    n := len(nums)
    if n == 1 {
        return nums[0]
    }
    if n == 2 {
        return max(nums[0], nums[1])
    }
    res1 := robRange(nums, 0, n-1) // 考虑包含首元素，不包含尾元素
    res2 := robRange(nums, 1, n)   // 考虑包含尾元素，不包含首元素
    return max(res1, res2)
}

func robRange(nums []int, start, end int) int {
    dp := make([]int, len(nums))
    dp[start] = nums[start]
    dp[start+1] = max(nums[start], nums[start+1])
    for i := start + 2; i < end; i++ {
        dp[i] = max(dp[i-2]+nums[i], dp[i-1])
    }
    return dp[end-1]
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
数组成环

虽然是考虑包含首/尾元素，但不一定要选首/尾元素
```

## [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/description/)

小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root` 。

除了 `root` 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 **两个直接相连的房子在同一天晚上被打劫** ，房屋将自动报警。

给定二叉树的 `root` 。返回 **在不触动警报的情况下** ，小偷能够盗取的最高金额 。

答案

```go
func rob(root *TreeNode) int {
    var search func(node *TreeNode) []int
    search = func(node *TreeNode) []int {
        if node == nil {
            return []int{0, 0}
        }
        // 后序遍历
        left := search(node.Left)
        right := search(node.Right)
        sum1 := node.Val + left[0] + right[0]                   // 偷当前结点
        sum0 := max(left[0], left[1]) + max(right[1], right[0]) // 不偷当前结点，所以可以考虑偷或不偷子节点
        return []int{sum0, sum1}                                // {不偷，偷}
    }
    res := search(root)
    return max(res[0], res[1])
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱。

如果是偷当前节点，那么左右孩子就不能偷，val1 = cur->val + left[0] + right[0]
如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的，所以：val2 = max(left[0], left[1]) + max(right[0], right[1])
left[0]表示左子树不偷左孩子的最大值，left[1]表示左子树偷左孩子的最大值
```

## [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/)

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

答案

```go
func maxProfit(prices []int) int {
    dp := [2]int{}
    dp[0] = -prices[0]
    for i:=1; i<len(prices); i++ {
        dp[0] = max(dp[0], -prices[i])    // 持有股票所得最多现金
        dp[1] = max(dp[1], dp[0]+prices[i])    // 不持有股票所得最多现金
    }
    return dp[1]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i][0] 表示第i天持有股票所得最多现金
dp[i][1] 表示第i天不持有股票所得最多现金
“持有”不代表就是当天“买入”，也有可能是昨天就买入了，今天保持持有的状态

dp[i][0] = max(dp[i - 1][0], -prices[i])
dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0])

dp[0][0]表示第0天持有股票，所以dp[0][0] = -prices[0];
dp[0][1]表示第0天不持有股票，不持有股票那么现金就是0，所以dp[0][1] = 0;
```

## [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/)

给你一个整数数组 `prices` ，其中 `prices[i]` 表示某支股票第 `i` 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以先购买，然后在 **同一天** 出售。

返回 *你能获得的 **最大** 利润* 。

答案

```go
func maxProfit(prices []int) int {
    dp := [2]int{}
    dp[0] = -prices[0]
    for i := 1; i < len(prices); i++ {
        dp[0] = max(dp[0], dp[1]-prices[i]) // 今天持有股票=max(昨天持有，昨天未持有且今天买入）
        dp[1] = max(dp[1], dp[0]+prices[i]) // 今天不持有股票=max(昨天未持有，昨天持有且今天卖出）
    }
    return dp[1]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
121，122，123其实都可以只用一维数组，不需要用二维的
```

## [123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

答案

```go
func maxProfit(prices []int) int {
    dp := [5]int{}
    dp[1] = -prices[0] // 第一次持有股票，即买入第一天
    dp[3] = -prices[0] // 第二次持有股票，即买入第一天
    for i := 1; i < len(prices); i++ {
        dp[1] = max(dp[1], dp[0]-prices[i]) // 第一次持有股票=max(昨天持有，昨天未持有且今天买入)
        dp[2] = max(dp[2], dp[1]+prices[i]) // 第一次不持有股票=max(昨天不持有，昨天持有且今天卖出)
        dp[3] = max(dp[3], dp[2]-prices[i]) // 第二次持有股票=max(昨天持有，昨天未持有且今天买入)
        dp[4] = max(dp[4], dp[3]+prices[i]) // 第二次不持有股票=max(昨天不持有，昨天持有且今天卖出)
    }
    return dp[4]
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp数组以及下标的含义：一天一共就有五个状态，
0.没有操作 （其实我们也可以不设置这个状态）
1.第一次持有股票
2.第一次不持有股票
3.第二次持有股票
4.第二次不持有股票
```

## 188. 买卖股票的最佳时机 IV

答案

```go
func maxProfit(k int, prices []int) int {
    if k == 0 || len(prices) == 0 {
        return 0
    }
    n := 2*k + 1    // 买k次，卖k次，加上什么操作都没有的0，所以n=2k+1
    dp := make([]int, n)
    for i := 1; i < n-1; i += 2 {
        dp[i] = -prices[0] // 第i(奇数次)持有股票，即买入第一天
    }
    for i := 1; i < len(prices); i++ {
        for j := 0; j < n-1; j += 2 { // 除了0以外，偶数就是卖出，奇数就是买入
            dp[j+1] = max(dp[j+1], dp[j]-prices[i])   // 持有股票=max(昨天持有，昨天未持有且今天买入)
            dp[j+2] = max(dp[j+2], dp[j+1]+prices[i]) // 不持有股票=max(昨天不持有，昨天持有且今天卖出)
        }
    }
    return dp[n-1]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
甚至Hard也可以只用一维数组

但是后面含冷冻期的不能用一维数组
原因可能是不能在同一天无限买卖，卖之后必须得到第二天

含手续费可以用一维数组，可能是因为虽然卖出有手续费，但是可以同一天无限买卖
```

## 309. 买卖股票的最佳时机含冷冻期

答案

```go
func maxProfit(prices []int) int {
    n := len(prices)
    if n < 2 {
        return 0
    }

    dp := make([][]int, n)
    for i := 0; i<n; i++ {
        dp[i] = make([]int,4)
    }
    dp[0][0] = -prices[0]

    for i := 1; i < n; i++ {
        // 达到买入股票状态：前一天就是持有股票状态、前一天是保持卖出股票的状态且今天买入、前一天是冷冻期且今天买入
        dp[i][0] = max(dp[i-1][0], max(dp[i-1][1]-prices[i], dp[i-1][3]-prices[i]))
        // 达到保持卖出股票状态：前一天是保持卖出股票的状态、前一天是冷冻期
        dp[i][1] = max(dp[i-1][1], dp[i-1][3])
        // 达到今天就卖出股票状态：昨天一定是持有股票状态且今天卖出
        dp[i][2] = dp[i-1][0] + prices[i]
        // 达到冷冻期状态：昨天卖出了股票
        dp[i][3] = dp[i-1][2]
    }

    return max(dp[n-1][1], max(dp[n-1][2], dp[n-1][3]))
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
具体可以区分出如下四个状态：
    0、状态一：持有股票状态（今天买入股票，或者是之前就买入了股票然后没有操作，一直持有）
    不持有股票状态，这里就有两种卖出股票状态
        1、状态二：保持卖出股票的状态（两天前就卖出了股票，度过一天冷冻期。或者是前一天就是卖出股票状态，一直没操作）
        2、状态三：今天卖出股票
    3、状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天
```

## 714. 买卖股票的最佳时机含手续费

答案

```go
func maxProfit(prices []int, fee int) int {
    dp := [2]int{}
    dp[0] = -prices[0]
    for i := 1; i < len(prices); i++ {
        dp[0] = max(dp[0], dp[1]-prices[i])        // 买入
        dp[1] = max(dp[1], dp[0]+prices[i]-fee)    // 卖出  区别就是这里多了一个减去手续费的操作
    }
    return dp[1]
}

// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go

```

## [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列

答案

```go
func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    ans := 1
    dp := make([]int, len(nums)) // dp[i] 为以第 i 个数字结尾的最长上升子序列的长度，nums[i] 必须被选取
    dp[0] = 1
    for i := 1; i < len(nums); i++ {
        dp[i] = 1
        for j := 0; j < i; j++ {
            if nums[i] > nums[j] { // if成立时，位置i的最长升序子序列 可以等于 位置j的最长升序子序列 + 1
                dp[i] = max(dp[i], dp[j]+1) // dp[i] 从 dp[j] 这个状态转移过来
            }
        }
        ans = max(ans, dp[i])
    }
    return ans
}


// 求两个数中的较大者
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i] 为以第 i 个数字结尾的最长上升子序列的长度，nums[i] 必须被选取

位置i的最长升序子序列等于j从0到i-1各个位置的最长升序子序列 + 1 的最大值
if (nums[i] > nums[j]) 
    dp[i] = max(dp[i], dp[j] + 1)

每一个i，对应的dp[i]（即最长递增子序列）起始大小至少都是1
```

## [674. 最长连续递增序列](https://leetcode.cn/problems/longest-continuous-increasing-subsequence/description/)

给定一个未经排序的整数数组，找到最长且 **连续递增的子序列**，并返回该序列的长度。

**连续递增的子序列** 可以由两个下标 `l` 和 `r`（`l < r`）确定，如果对于每个 `l <= i < r`，都有 `nums[i] < nums[i + 1]` ，那么子序列 `[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]]` 就是连续递增子序列。

答案

```go
func findLengthOfLCIS(nums []int) int {
    n := len(nums)
    if n < 2 {    // 不能漏这一步，否则有样例过不了
        return 1
    }
    dp := make([]int, n)
    res := 0
    for i := 0; i < n; i++ {
        dp[i] = 1        // dp[i]的初始化 : 每一个i，对应的dp[i]（即最长上升子序列）起始大小至少都是1
    }
    for i := 0; i < n-1; i++ {
        if nums[i+1] > nums[i] {
            dp[i+1] = dp[i] + 1
        }
        res = max(res, dp[i+1])
    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i]：以下标i为结尾的连续递增的子序列长度为dp[i]
一定是以下标i为结尾，并不是说一定以下标0为起始位置

如果 nums[i] > nums[i - 1]，那么以 i 为结尾的连续递增的子序列长度 一定等于 以i - 1为结尾的连续递增的子序列长度 + 1 。
即：dp[i] = dp[i - 1] + 1
本题要求连续递增子序列，所以就只要比较nums[i]与nums[i - 1]，而不用去比较nums[j]与nums[i] （j是在0到i之间遍历）。
既然不用j了，那么也不用两层for循环，本题一层for循环就行，比较nums[i] 和 nums[i - 1]

以下标i为结尾的连续递增的子序列长度最少也应该是1
```

## [718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/description/)

给两个整数数组 `nums1` 和 `nums2` ，返回 *两个数组中 **公共的** 、长度最长的子数组的长度* 

答案

```go
func findLength(A []int, B []int) int {
    m, n := len(A), len(B)
    dp := make([][]int, m+1) // dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]
    res := 0
    for i := 0; i <= m; i++ {
        dp[i] = make([]int, n+1)
    }
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if A[i-1] == B[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            }
            res = max(res, dp[i][j])
        }

    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i][j] ：以下标i - 1为结尾的A，和以下标j - 1为结尾的B，最长重复子数组长度为dp[i][j]

当A[i - 1] 和B[j - 1]相等的时候，dp[i][j] = dp[i - 1][j - 1] + 1

dp[i][0] 和dp[0][j]初始化为0
dp[1][1] = dp[0][0] + 1，只有dp[0][0]初始为0，正好符合递推公式逐步累加起来
```

## [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/)

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

答案

```go
func longestCommonSubsequence(text1 string, text2 string) int {
    // dp[i][j]：长度为[0, i - 1]的字符串text1与长度为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i:=0; i<=m; i++ {
        dp[i] = make([]int, n+1)
    }
    for i:=1; i<=m; i++ {
        for j:=1; j<=n; j++ {
            if text1[i-1] == text2[j-1] {
                // 状态转移方程 : 主要就是两大情况
                // 如果text1[i - 1] 与 text2[j - 1]相同，即找到了一个公共元素，所以dp[i][j] = dp[i - 1][j - 1] + 1;
                // 如果text1[i - 1] 与 text2[j - 1]不相同，那么：dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }

    }
    return dp[m][n]
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i][j]：下标为[0, i - 1]的字符串text1与下标为[0, j - 1]的字符串text2的最长公共子序列为dp[i][j]

如果text1[i - 1] 与 text2[j - 1]相同，即找到了一个公共元素，所以dp[i][j] = dp[i - 1][j - 1] + 1
如果text1[i - 1] 与 text2[j - 1]不相同，那就在text1[0, i - 2]与text2[0, j - 1]的最长公共子序列 和 text1[0, i - 1]与text2[0, j - 2]的最长公共子序列中取最大的，即：dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
```

## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

答案

```go
func maxSubArray(nums []int) int {
    // dp[i]：包括下标i之前的最大连续子序列和为dp[i]
    n := len(nums)
    dp := make([]int, n)
    // dp[i]的初始化    由于dp 状态转移方程依赖dp[0]
    dp[0] = nums[0]
    // 初始化最大的和
    res := nums[0]
    for i:=1; i<n; i++ {
        // dp[i]只有两个方向可以推出来：nums[i]加入当前连续子序列和    从头开始计算当前连续子序列和
        dp[i] = max(dp[i-1]+nums[i], nums[i])
        res = max(res, dp[i])
    }
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

分析

```go
dp[i]：包括下标i（以nums[i]为结尾）的最大连续子序列和为dp[i]。

dp[i]只有两个方向可以推出来：
dp[i - 1] + nums[i]，即：nums[i]加入当前连续子序列和
nums[i]，即：从头开始计算当前连续子序列和
一定是取最大的，所以dp[i] = max(dp[i - 1] + nums[i], nums[i])

dp[0]应为nums[0]
```

## [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/)

给你一个字符串 `s`，找到 `s` 中最长的 回文子串

答案

```go
func longestPalindrome(s string) string {
    n := len(s)
    if n == 1 {
        return s
    }
    start, maxLen := 0, 1
    dp := make([][]bool, n)    // dp[i][j] 表示 s[i..j] 是否是回文串
    for i := 0; i < n; i++ {    // 初始化：所有长度为 1 的子串都是回文串
        dp[i] = make([]bool, n)
        dp[i][i] = true
    }
    for Len := 2; Len <= n; Len++ { // 先枚举子串长度
        for i := 0; i < n; i++ { // 枚举左边界
            j := i + Len - 1 // 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
            if j >= n {      // 检查右边界是否越界
                break
            }
            if s[i] != s[j] {    // s[i..j]不可能是回文串
                dp[i][j] = false
            } else {    // s[i..j]可能是回文串，需要进一步判断
                if j-i == 1 {    // 不可能j-i=0，因为Len至少为2
                    dp[i][j] = true // 下标i与j相差为1（例如aa），是回文子串
                } else {
                    dp[i][j] = dp[i+1][j-1]    // 看dp[i + 1][j - 1]是否为true
                }
            }
            // 只要 dp[i][j] == true 成立，就表示子串 s[i..j] 是回文，此时记录回文长度和起始位置
            if dp[i][j] && j-i+1 > maxLen {
                maxLen = j - i + 1
                start = i
            }
        }
    }
    return s[start : start+maxLen]
}
```

分析

```go
布尔类型的dp[i][j]：表示区间范围[i,j] （注意是左闭右闭）的子串是否是回文子串，如果是回文字串dp[i][j]为true

当s[i]与s[j]不相等，dp[i][j]一定是false
当s[i]与s[j]相等时，有如下三种情况
        情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
        情况二：下标i 与 j相差为1，例如aa，也是回文子串
        情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，我们看i到j区间是不是回文子串就看aba是不是回文                    就可以了，那么aba的区间就是 i+1 与 j-1 区间，即看dp[i + 1][j - 1]是否为true
```

## [72. 编辑距离](https://leetcode.cn/problems/edit-distance/description/)

给你两个单词 `word1` 和 `word2`， *请返回将 `word1` 转换成 `word2` 所使用的最少操作数*  

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

答案

```go
func minDistance(word1 string, word2 string) int {
    // dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]
    m, n := len(word1), len(word2)
    dp := make([][]int, m+1)

    // 初始化
    for i := 0; i <= m; i++ {
        dp[i] = make([]int, n+1)
        dp[i][0] = i // dp[i][0] ：以下标i-1为结尾的字符串word1，和空字符串word2，最近编辑距离dp[i][0]为i
    }
    for j := 0; j <= n; j++ {
        dp[0][j] = j // dp[0][j] ：以下标j-1为结尾的字符串word2，和空字符串word1，最近编辑距离dp[0][j]为j
    }

    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
            }
        }
    }
    return dp[m][n]
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

分析

```go
dp[i][j] 表示以下标i-1为结尾的字符串word1，和以下标j-1为结尾的字符串word2，最近编辑距离为dp[i][j]。

if (word1[i - 1] == word2[j - 1])
    不操作
if (word1[i - 1] != word2[j - 1])
    增
    删
    换

递推公式：
word1[i-1] == word2[j-1]    
        说明不用任何编辑，dp[i][j] = dp[i - 1][j - 1]
word1[i-1] != word2[j-1]
        操作一：word1删除一个元素，那么就是以下标i - 2为结尾的word1 与 j-1为结尾的word2的最近编辑距离 再加上一个操作。
                即 dp[i][j] = dp[i-1][j] + 1
        操作二：word2删除一个元素，那么就是以下标i - 1为结尾的word1 与 j-2为结尾的word2的最近编辑距离 再加上一个操作。
                即 dp[i][j] = dp[i][j-1] + 1
                word2添加一个元素，相当于word1删除一个元素，所以没有添加操作
        操作三：替换元素，word1替换word1[i-1]，使其与word2[j-1]相同，此时不用增删加元素
                dp[i][j] = dp[i - 1][j - 1] + 1
        综上，当 word1[i-1] != word2[j-1] 时取最小的，即：dp[i][j] = min({dp[i-1][j-1], dp[i-1][j], dp[i][j-1]}) + 1

初始化：
        dp[i][0] ：以下标i-1为结尾的字符串word1，和空字符串word2，最近编辑距离为dp[i][0]
        那么dp[i][0]就应该是i，对word1里的元素全部做删除操作，即：dp[i][0] = i
        同理dp[0][j] = j    
```

## [139. 单词拆分](https://leetcode.cn/problems/word-break/)

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

答案

```go
func wordBreak(s string, wordDict []string) bool {
    wordDictSet := make(map[string]bool)
    for _, w := range wordDict {
        wordDictSet[w] = true
    }
    dp := make([]bool, len(s) + 1)
    dp[0] = true
    for i := 1; i <= len(s); i++ {    // 遍历 dp[i] 能否由 dp[j] 与 wordDictSet[s[j:i]] 组成
        for j := 0; j < i; j++ {
            if dp[j] && wordDictSet[s[j:i]] {
                dp[i] = true
                break
            }
        }
    }
    return dp[len(s)]
}
```

分析

```go
dp[i]表示字符串s前 i 个字符组成的字符串 s[0..i−1] 是否能被空格拆分成若干个字典中出现的单词
```

## [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

答案

```go
func maxProduct(nums []int) int {
    maxF, minF, ans := nums[0], nums[0], nums[0]
    for i := 1; i < len(nums); i++ {
        maxTmp, minTmp := maxF, minF
        // 三者取大，就是第 i 个元素结尾的乘积最大子数组的乘积
        maxF = max(maxTmp * nums[i], max(nums[i], minTmp * nums[i]))
        minF = min(minTmp * nums[i], min(nums[i], maxTmp * nums[i]))
        ans = max(maxF, ans)
    }
    return ans
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}
```

分析

```go
要根据正负性进行分类讨论
当前位置如果是一个负数，我们希望以它前一个位置结尾的某个段的积也是个负数，这样就可以负负得正，并且希望这个积尽可能「负得更多」，即尽可能小。
当前位置如果是一个正数，我们更希望以它前一个位置结尾的某个段的积也是个正数，并且希望它尽可能地大。
```

# 设计/模拟

## 146. LRU 缓存

手搓答案

```go
// 哈希表 + 双向链表
// 双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。
// 哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。
// 首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在 O(1) 的时间内完成 get 或者 put 操作

// 在双向链表的实现中，使用一个伪头部（dummy head）和伪尾部（dummy tail）标记界限
// 这样在添加节点和删除节点的时候就不需要检查相邻的节点是否存在


type LRUCache struct {
    size int
    capacity int
    cache map[int]*DLinkedNode
    head, tail *DLinkedNode
}

type DLinkedNode struct {
    key, value int
    prev, next *DLinkedNode
}

func initDLinkedNode(key, value int) *DLinkedNode {
    return &DLinkedNode{
        key: key,
        value: value,
    }
}

func Constructor(capacity int) LRUCache {
    l := LRUCache{
        cache: map[int]*DLinkedNode{},
        head: initDLinkedNode(0, 0),
        tail: initDLinkedNode(0, 0),
        capacity: capacity,
    }
    l.head.next = l.tail
    l.tail.prev = l.head
    return l
}

func (this *LRUCache) Get(key int) int {
    if _, ok := this.cache[key]; !ok {    // 首先判断 key 是否存在
        return -1
    }
    node := this.cache[key]
    this.moveToHead(node)    // key 对应的节点是最近被使用的节点， 将其移动到双向链表的头部
    return node.value
}


func (this *LRUCache) Put(key int, value int)  {
    if _, ok := this.cache[key]; !ok {    // 首先判断 key 是否存在
        node := initDLinkedNode(key, value)    // 使用 key 和 value 创建一个新的节点
        this.cache[key] = node    // 将 key 和该节点添加进哈希表中
        this.addToHead(node)    // 在双向链表的头部添加该节点
        this.size++
        if this.size > this.capacity {    // 判断双向链表的节点数是否超出容量
            removed := this.removeTail()    // 删除双向链表的尾部节点
            delete(this.cache, removed.key)    // 删除哈希表中对应的项
            this.size--
        }
    } else {
        node := this.cache[key]
        node.value = value    // 将对应的节点的值更新为 value
        this.moveToHead(node)    // 将该节点移到双向链表的头部
    }
}

func (this *LRUCache) addToHead(node *DLinkedNode) {
    node.prev = this.head
    node.next = this.head.next
    this.head.next.prev = node
    this.head.next = node
}

func (this *LRUCache) removeNode(node *DLinkedNode) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (this *LRUCache) moveToHead(node *DLinkedNode) {
    this.removeNode(node)
    this.addToHead(node)
}

func (this *LRUCache) removeTail() *DLinkedNode {
    node := this.tail.prev
    this.removeNode(node)
    return node
}

作者：力扣官方题解
链接：https://leetcode.cn/problems/lru-cache/solutions/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

List包答案

```go
type LRUNode struct {
    key int
  value int
}

type LRUCache struct {
    capacity int
    cache map[int]*list.Element        // Element 用于代表双链表的元素
    LRUList *list.List    // List 用于表示双链表
}

func Constructor(capacity int) LRUCache {
    return LRUCache{
        capacity: capacity,
        cache: map[int]*list.Element{},
        LRUList: list.New(),    // 通过 container/list 包的 New() 函数初始化 list
    }
}


func (this *LRUCache) Get(key int) int {
    element := this.cache[key]    // 获得 valve
    if element == nil {        // 关键字 key 不在缓存中
        return -1
    }
    this.LRUList.MoveToFront(element)    // 刷新缓存使用时间    将元素 e 移动到链表的开头
    return element.Value.(LRUNode).value
}


func (this *LRUCache) Put(key int, value int)  {
    element := this.cache[key]
    if element != nil {        // 关键字 key 已经存在，则变更其数据值 value
        element.Value = LRUNode{key: key, value: value}
        this.LRUList.MoveToFront(element)    // 刷新缓存使用时间    将元素 e 移动到链表的开头
        return
    }
    // 如果不存在，则向缓存中插入该组 key-value
    this.cache[key] = this.LRUList.PushFront(LRUNode{ key: key, value: value})    //将包含了值v的元素e插入到链表的开头并返回e
    if len(this.cache) > this.capacity {    // 如果插入操作导致关键字数量超过 capacity ，则应该逐出最久未使用的关键字
        delete(this.cache, this.LRUList.Remove(this.LRUList.Back()).(LRUNode).key)
    }
}
```

分析

Go语言list（列表）    http://c.biancheng.net/view/35.html

Go标准库中文文档    container/list    http://cngolib.com/container-list.html#container-list

代码参考  https://leetcode.cn/problems/lru-cache/solution/golang-jian-ji-xie-fa-by-endlesscheng-a3b2/

原理分析  https://leetcode.cn/problems/lru-cache/solution/jian-dan-shi-li-xiang-xi-jiang-jie-lru-s-exsd/

```go
当缓存容量已满，我们不仅仅要删除最后一个 Node 节点，还要把 map 中映射到该节点的 key 同时删除，而这个 key 只能由 Node 得到。如果 Node 结构中只存储 val，那么我们就无法得知 key 是什么，就无法删除 map 中的键，造成错误。

作者：labuladong
链接：https://leetcode.cn/problems/lru-cache/solution/lru-ce-lue-xiang-jie-he-shi-xian-by-labuladong/

所以这里
delete(this.cache, this.LRUList.Remove(this.LRUList.Back()).(LRUNode).key)
是先利用this.LRUList.Back()返回了尾结点，
然后再利用this.LRUList.Remove删除了尾结点，
最后利用delete删除了map里的key
```



## LRU

- get(key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。 
- put(key, value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
- 时间复杂度：所有操作均为 O(1)。

```go
type entry struct {
    key, value int
}

type LRUCache struct {
    capacity  int
    list      *list.List // 双向链表
    keyToNode map[int]*list.Element
}

func Constructor(capacity int) LRUCache {
    return LRUCache{capacity, list.New(), map[int]*list.Element{}}
}

func (c *LRUCache) Get(key int) int {
    node := c.keyToNode[key]
    if node == nil { // 没有要找的节点
        return -1
    }
    c.list.MoveToFront(node) // 把这个节点放在最上面
    return node.Value.(entry).value
}

func (c *LRUCache) Put(key, value int) {
    if node := c.keyToNode[key]; node != nil { // 有要找的节点
        node.Value = entry{key, value} // 更新
        c.list.MoveToFront(node) // 把节点放在最上面
        return
    }
    c.keyToNode[key] = c.list.PushFront(entry{key, value}) // 新节点，放在最上面
    if len(c.keyToNode) > c.capacity { // 节点太多了
        delete(c.keyToNode, c.list.Remove(c.list.Back()).(entry).key) // 去掉最后的节点
    }
}
```

分析

```go

```



## LFU

get( key) - 如果键 key 存在于缓存中，则获取键的值，否则返回 -1 。 

put(key, value) - 如果键 key 已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量 capacity 时，则应该在插入新项之前，移除最不经常使用的项。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最久未使用 的键。

```go
type entry struct {
    key, value, freq int // freq 表示这个节点被访问的次数
}

type LFUCache struct {
    capacity   int
    minFreq    int
    keyToNode  map[int]*list.Element
    freqToList map[int]*list.List    // 不同访问次数的节点list
}

func Constructor(capacity int) LFUCache {
    return LFUCache{
        capacity:   capacity,
        keyToNode:  map[int]*list.Element{},
        freqToList: map[int]*list.List{},
    }
}

func (c *LFUCache) pushFront(e *entry) {
    if _, ok := c.freqToList[e.freq]; !ok {    // 找不到这个节点访问次数所属的节点list
        c.freqToList[e.freq] = list.New() // 双向链表
    }
    c.keyToNode[e.key] = c.freqToList[e.freq].PushFront(e)    // 放到对应的节点list中
}

func (c *LFUCache) getEntry(key int) *entry {
    node := c.keyToNode[key]
    if node == nil { // 没有找到节点
        return nil
    }
    e := node.Value.(*entry)
    lst := c.freqToList[e.freq]
    lst.Remove(node) // 把这个节点抽出来
    if lst.Len() == 0 { // 抽出来后，如果节点原来所属的节点list是空的
        delete(c.freqToList, e.freq) // 移除空链表
        if c.minFreq == e.freq { // 如果节点list是最左边的
            c.minFreq++
        }
    }
    e.freq++ // 访问次数 +1
    c.pushFront(e) // 放在右边节点list的最上面
    return e
}

func (c *LFUCache) Get(key int) int {
    if e := c.getEntry(key); e != nil { // 有这个节点
        return e.value
    }
    return -1 // 没有这个节点
}

func (c *LFUCache) Put(key, value int) {
    if e := c.getEntry(key); e != nil { // 有这个节点
        e.value = value // 更新 value
        return
    }
    if len(c.keyToNode) == c.capacity { // 节点太多了
        lst := c.freqToList[c.minFreq] // 最左边的节点list
        delete(c.keyToNode, lst.Remove(lst.Back()).(*entry).key) // 移除这个list最下面的节点
        if lst.Len() == 0 { // 这个节点list是空的
            delete(c.freqToList, c.minFreq) // 移除空链表
        }
    }
    c.pushFront(&entry{key, value, 1}) // 新节点放在「看过 1 次」的最上面
    c.minFreq = 1
}
```

分析

```go

```



## 415. 字符串相加

```go
func addStrings(num1 string, num2 string) string {
    add := 0    // 维护当前是否有进位
    ans := ""
    // 从末尾到开头逐位相加
    for i, j := len(num1)-1, len(num2)-1; i>=0 || j>=0 || add!=0; i, j = i-1, j-1 {
        var x, y int    // 默认为0，即在指针当前下标处于负数的时候返回 0   等价于对位数较短的数字进行了补零操作
        if i >= 0 {
            x = int(num1[i] - '0')
        }
        if j >= 0 {
            y = int(num2[j] - '0')
        }
        result := x + y + add
        ans = strconv.Itoa(result%10) + ans
        add = result / 10
    }
    return ans
}
```

分析

```go

```



## [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/)

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

第一种解法：

```go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    m, n := len(matrix), len(matrix[0])
    ans := make([]int, 0)
    top, bottom, left, right := 0, m-1, 0, n-1
    for left <= right && top <= bottom {
        // 将矩阵看成若干层，首先输出最外层的元素，其次输出次外层的元素，直到输出最内层的元素
        for i:=left; i<=right; i++ {            // 左上方到右
            ans = append(ans, matrix[top][i])
        }
        for i:=top+1; i<=bottom; i++ {            // 右上方到下
            ans = append(ans, matrix[i][right])
        }
        // left == right : 剩余的矩阵是一竖的形状  所以不需要再往上
        // top == bottom : 剩余的矩阵是一横的形状  所以不需要再往左
        if left < right && top < bottom {    // 当 left == right 或者 top == bottom 时，不会发生右到左和下到上，否则会重复计数
            for i:=right-1; i>=left; i-- {        // 右下方到左
                ans = append(ans, matrix[bottom][i])
            }
            for i:=bottom-1; i>top; i-- {        // 左下方到上   这里的判断条件是特殊的，不加=号
                ans = append(ans, matrix[i][left])
            }
        }
        top++
        bottom--
        left++
        right--
    }
    return ans
}
```

第二种解法，根据59改的：

```go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    m, n := len(matrix), len(matrix[0])
    ans := make([]int, 0)
    top, bottom, left, right := 0, m-1, 0, n-1
    for left <= right && top <= bottom {
        for i:=left; i<=right; i++ {            // 左上方到右
            ans = append(ans, matrix[top][i])
        }
    top++
        for i:=top; i<=bottom; i++ {            // 右上方到下
            ans = append(ans, matrix[i][right])
        }
    right--
    // 这里的判断条件必须是&&，不能是||
    // 这里可以是=，因为已经对top和right进行了处理，所以left可能等于right，top可能等于bottom，即有两行或两列
        if left <= right && top <= bottom {// 当 left > right 或者 top > bottom 时，不会发生右到左和下到上，否则会重复计数
            for i:=right; i>=left; i-- {        // 右下方到左
                ans = append(ans, matrix[bottom][i])
            }
      bottom--
            for i:=bottom; i>=top; i-- {        // 左下方到上
                ans = append(ans, matrix[i][left])
            }
      left++
        }
    }
    return ans
}
```

分析

```go
我个人觉得第二个解法比较好，和59是类似的，不用特判，条件也便于理解
```

## [59. 螺旋矩阵II](https://leetcode.cn/problems/spiral-matrix-ii/description/)

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

```go
func generateMatrix(n int) [][]int {
    top, bottom := 0, n-1
    left, right := 0, n-1
    num := 1    // 给矩阵赋值
    tar := n * n
    matrix := make([][]int, n)
    for i := 0; i < n; i++ {
        matrix[i] = make([]int, n)
    }
    for num <= tar {    // 从1～n^2遍历
        for i := left; i <= right; i++ {    // 左上到右
            matrix[top][i] = num
            num++
        }
        top++    // 右上角往下一格
        for i := top; i <= bottom; i++ {    // 右上到下
            matrix[i][right] = num
            num++
        }
        right--    // 右下角往左一格
        for i := right; i >= left; i-- {    // 右下到左
            matrix[bottom][i] = num
            num++
        }
        bottom--    // 左下角往上一格
        for i := bottom; i >= top; i-- {    // 左下到上
            matrix[i][left] = num
            num++
        }
        left++
    }
    return matrix
}
```

分析

```go
按照相同的原则，每条边左闭右开或左闭右闭
```

## [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

给定一个 `m x n` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)** 算法。

```go
// 使用标记数组
func setZeroes(matrix [][]int) {
    row := make([]bool, len(matrix))
    col := make([]bool, len(matrix[0]))
    for i, r := range matrix {
        for j, v := range r {
            if v == 0 {
                row[i] = true
                col[j] = true
            }
        }
    }
    for i, r := range matrix {
        for j := range r {
            if row[i] || col[j] {
                r[j] = 0
            }
        }
    }
}
```

分析

```go

```

## [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 **[原地](https://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)** 旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

```go
func rotate(matrix [][]int) {
    n := len(matrix)
    // 水平翻转
    for i := 0; i < n/2; i++ {
        matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
    }
    // 主对角线翻转
    for i := 0; i < n; i++ {
        for j := 0; j < i; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
}
```

分析

```go

```

## [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

编写一个高效的算法来搜索 `*m* x *n*` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

```go
func searchMatrix(matrix [][]int, target int) bool {
    m, n := len(matrix), len(matrix[0])
    x, y := 0, n-1    // 从矩阵 matrix 的右上角 (0,n−1) 进行搜索
    for x < m && y >= 0 {
        if matrix[x][y] == target {    // 搜索完成
            return true
        }
        if matrix[x][y] > target {    
            y--    // 所有位于第 y 列的元素都是严格大于 target 的，因此我们可以将它们全部忽略
        } else {
            x++    // 所有位于第 x 行的元素都是严格小于 target 的，因此我们可以将它们全部忽略
        }
    }
    return false
}
```

分析

```go

```



## 并查集

并查集是一种树（森林）型的数据结构。将编号分别为1-N的N个对象划分为不相交集合。在每个集合（树）中，选择1个元素（树的根）代表所在集合。能够高效的实现对集合进行合并和查询集合中元素的关系。

```go
package main
import "fmt"

/*
并查集，判连通用
*/
type UnionFindSet struct {
   father  []int // 存储结点的父亲
   nodeNum int   // 总结点个数
}

func (us *UnionFindSet) InitUnionSet(n int) {
   us.nodeNum = n+1 // 不加也可以，有人喜欢以0开头
   us.father = make([]int, us.nodeNum)
   for i, _ := range us.father {
      us.father[i] = i
   }
}

// 查找并返回给定元素 x 所在集合的代表元素（即根节点）
func (us *UnionFindSet) Find(x int) int {
    // 判断 x 是否是自己的父节点。如果 x 不是，说明它不是根节点，而是某个集合的子节点
    // 需要沿着父节点继续查找，直到找到根节点
   if us.father[x] != x {
   // 递归查找 x 的父节点，并将查找到的根节点直接赋值给 x。
   // 这就是路径压缩的关键：在递归的过程中，把路径上的所有节点直接连接到根节点。
   // 这样做的好处是，在未来的 Find 操作中，查找这些节点会更快，因为它们直接指向根节点。
       us.father[x] = us.Find(us.father[x])
   }
   return us.father[x]
}

//合并结点
func (us *UnionFindSet) Union(x, y int) bool {
   x = us.Find(x)    // 查找 x 的根节点
   y = us.Find(y)    // 查找 y 的根节点
   if x == y {    // x 和 y 已经在同一个集合中，无需再次合并，直接返回 false 表示没有发生变化。
      return false
   }
   us.father[x] = y    // 将 x 的根节点的父节点设置为 y（x 的集合被并入了 y 的集合）
   return true    // 成功合并
}

func main() {
   us:= &UnionFindSet{}
   us.InitUnionSet(11)
   us.Union(1, 2)
   us.Union(10, 9)
   us.Union(7,8)
   us.Union(8,3)
   us.Union(8,9)

   for i:=0;i<=10; i++  {
      fmt.Println(i, us.Find(i))
   }
}
```

分析

```go

```





# 堆

## 数组构建堆

堆是一种特殊的完全二叉树结构，在完全二叉树中，除了最后一层外，每一层都是完全填满的，而最后一层的节点则都集中在左侧。堆可以用数组来表示，不需要使用指针来构造节点之间的关系。

利用数组建立堆主要涉及到两个过程：构建初始堆（最大堆或最小堆）和调整堆（通过堆化）。
这里以最大堆为例说明如何使用数组来建立堆：

1. **理解堆的数组表示**
   - 堆可以被视为一棵完全二叉树，其中每个节点都有一个值，这棵树完全可以用一个数组来表示。
   - 对于数组中任意一个位置`i`的元素，其左子节点位置为`2*i + 1`，右子节点位置为`2*i + 2`，父节点位置为`(i-1)/2`（这里的`i`是从0开始的索引）。
2. **构建初始堆**
   - 起始时，可以将整个数组看作一个几乎完整的堆，只是这个堆的堆化性质可能尚未满足。
   - 从最后一个非叶子节点开始（即数组长度的一半处向下取整），对每个节点执行堆化操作。堆化是确保节点的值大于其子节点的值（对于最大堆）。
   - 通过逐步向上进行，直至根节点，可以确保所有的父节点都大于其子节点，从而建立最大堆。
   - 建堆过程的时间复杂度是O(n)
3. **堆化过程（`maxHeapify`或`heapify`）**
   - 在堆化过程中，对于每个节点，比较其与两个子节点的值。
   - 如果父节点小于其任一子节点（对于最大堆），则将父节点与最大的子节点交换。
   - 交换后，继续对交换下去的子节点进行堆化，确保下面的子树也满足最大堆的性质。
   - 这个过程一直进行，直到该节点的子树都满足堆的条件。
   - 一次堆化的时间复杂度是O(logn)
4. **重复调整**
   - 初始堆建立后，在堆的使用过程中（如堆排序或优先队列操作），每次从堆中取出顶部元素（最大元素）后，需要将堆的最后一个元素移到顶部，然后再次进行堆化操作，以保持堆的性质。

![](https://img2020.cnblogs.com/blog/953680/202005/953680-20200531004135177-1000133948.png)

## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `**k**` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```go
func findKthLargest(nums []int, k int) int {
    heapSize := len(nums)    // 代表堆中的元素数量
    buildMaxHeap(nums, heapSize)    // 建立最大堆后
    for i := len(nums) - 1; i >= len(nums) - k + 1; i-- {
        nums[0], nums[i] = nums[i], nums[0]    // // 将数组的第一个元素（最大堆中的最大元素）与当前堆的最后一个元素交换
        heapSize--    // 然后将堆的大小减1
        maxHeapify(nums, 0, heapSize)    // 为减小的堆恢复最大堆的性质
    }
    return nums[0]    // 上述过程重复k-1次，有效地从堆中移除最大的元素，并放在数组的末尾。最后，第k大的元素将位于堆的顶部
}

// 构建最大堆
func buildMaxHeap(a []int, heapSize int) {    
    for i := heapSize/2; i >= 0; i-- {    // 从数组的中间位置开始（heapSize/2），向上到根部遍历非叶子节点
        maxHeapify(a, i, heapSize)    // 确保堆的性质，即每个父节点都大于其子节点
    }
}

// 最大堆化    确保以索引i为根的子树满足最大堆的性质
func maxHeapify(a []int, i, heapSize int) {    
    l, r, largest := i * 2 + 1, i * 2 + 2, i    
    if l < heapSize && a[l] > a[largest] {    // 比较索引i处的元素与其左（l）子节点，并找出最大的元素
        largest = l
    }
    if r < heapSize && a[r] > a[largest] {    // 比较索引i处的元素与其右（r）子节点，并找出最大的元素
        largest = r
    }
    if largest != i {    // 如果最大的元素不是父节点（即i）
        a[i], a[largest] = a[largest], a[i]    // 进行交换
        maxHeapify(a, largest, heapSize)    // 并对交换后的最大元素所在的子树递归调用maxHeapify
    }
}
```

分析

```go
使用堆排序建立一个大根堆，做 k−1 次删除操作后堆顶元素就是我们要找的答案

时间复杂度：O(nlog⁡n)，建堆的时间代价是 O(n)，删除的总代价是 O(klog⁡n)，因为 k<n，故渐进时间复杂为 O(n+klog⁡n)=O(nlog⁡n)
空间复杂度：O(log⁡n)，即递归使用栈空间的空间代价。
```

## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

```go
func topKFrequent(nums []int, k int) []int {
    cnts := make(map[int]int, k)
    for i := range nums {
        cnts[nums[i]]++
    }
    //构建元素出现次数的数组,方便进行堆操作
    heapCnt := make([][2]int, 0, len(cnts))
    for num, cnt := range cnts {
        heapCnt = append(heapCnt, [2]int{num, cnt})
    }
    // fmt.Printf("cnts=%+v,heapCnt=%+v\n", cnts, heapCnt)

    /*
            1. 自顶向下调整堆小顶堆
            2. 小顶堆的堆顶为出现次数第k多的元素,堆里面的元素出现次数都>=堆顶元素出现的次数
                    0
                1       2
             3     4  5   6
    */
    heapify := func(heapCnt [][2]int, start, heapSize int) {    // start 为根节点索引
        for start < heapSize {
            pos, left, right := start, 2*start+1, 2*start+2    
            if left < heapSize && heapCnt[left][1] < heapCnt[pos][1] {    // 比较count与其左子节点，并找出最小的元素
                pos = left
            }
            if right < heapSize && heapCnt[right][1] < heapCnt[pos][1] {// 比较count与其右子节点，并找出最小的元素
                pos = right
            }
            if pos == start {    // 如果最小的元素是根节点，则不用继续堆化了
                break
            }
            heapCnt[pos], heapCnt[start] = heapCnt[start], heapCnt[pos]    // 进行交换
            start = pos    // 并对交换后的最小元素所在的子树递归调堆化（这里用for来处理）
        }
    }

    // 堆的大小为k,从最后一个非叶子节点开始堆化数组
    for p := k / 2; p >= 0; p-- {
        heapify(heapCnt, p, k)
    }

    //从第k+1个元素开始每个元素也入堆一次
    for i := k; i < len(heapCnt); i++ {
        /*
            和堆顶元素比较,如果出现次数大于堆顶元素,就和堆顶元素交换,
            相当于删除堆顶元素,然后堆顶元素下沉
        */
        if heapCnt[i][1] > heapCnt[0][1] {
            heapCnt[i], heapCnt[0] = heapCnt[0], heapCnt[i]
            heapify(heapCnt, 0, k)
        }
    }

    ret := make([]int, 0, k)
    //堆中元素就是要求的结果
    for i := 0; i < k; i++ {
        ret = append(ret, heapCnt[i][0])
    }

    return ret
}
```

分析

```go
在这里，我们可以利用堆的思想：建立一个小顶堆，然后遍历「出现次数数组」：

如果堆的元素个数小于 k，就可以直接插入堆中。
如果堆的元素个数等于 k，则检查堆顶与当前出现次数的大小。如果堆顶更大，说明至少有 k 个数字的出现次数比当前值大，故舍弃当前值；
如果堆的元素个数大于 k，就弹出堆顶，并将当前值插入堆中。
遍历完成后，堆中的元素就代表了「出现次数数组」中前 k 大的值。

www.leetcode.cn/problems/top-k-frequent-elements/solutions/1524980/by-zhaobulw-tdb6/?envType=study-plan-v2&envId=top-100-liked

大顶堆(效率不高)
用大顶堆堆化全部元素,堆顶元素保存的是出现次数最多的元素
依次删除堆顶元素k次,就是出现次数前k多的元素

小顶堆
设置一个容量为k的小顶堆,初始时只堆化前k个元素,堆顶元素是前k个元素中出现次数最少的
从第k+1个元素开始,如果该元素出现次数比堆顶元素多,就和堆顶元素交换,并且从堆顶开始下沉,
注意此时堆中元素个数依旧为k
这样到最后堆顶就是出现次数第k多的元素,堆中存储的就是出现次数前k多的所有元素
```

## [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)

**中位数**是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

```go
type MinHeap []int

func (hp MinHeap) Len() int           { return len(hp) }              // 返回堆中元素的数量
func (hp MinHeap) Less(i, j int) bool { return hp[i] < hp[j] }        // 比较两个元素的大小
func (hp MinHeap) Swap(i, j int)      { hp[i], hp[j] = hp[j], hp[i] } // 交换两个元素的位置

func (hp *MinHeap) Push(x interface{}) { // 添加元素到堆中
    *hp = append(*hp, x.(int))    // 使用了类型断言
}

func (hp *MinHeap) Pop() interface{} { // 从堆中移除并返回最顶部的元素
    n := len(*hp)
    x := (*hp)[n-1]
    *hp = (*hp)[:n-1]
    return x
}

type MaxHeap []int

func (hp MaxHeap) Len() int           { return len(hp) }
func (hp MaxHeap) Less(i, j int) bool { return hp[i] > hp[j] }
func (hp MaxHeap) Swap(i, j int)      { hp[i], hp[j] = hp[j], hp[i] }

func (hp *MaxHeap) Push(x interface{}) {
    *hp = append(*hp, x.(int))    // 使用了类型断言
}

func (hp *MaxHeap) Pop() interface{} {
    n := len(*hp)
    x := (*hp)[n-1]
    *hp = (*hp)[:n-1]
    return x
}

type MedianFinder struct {
    // 最小堆用于存储数据流的较大一半的元素
    minhp *MinHeap
    // 最大堆用于存储数据流的较小一半的元素
    maxhp *MaxHeap
}

func Constructor() MedianFinder {
    return MedianFinder{&MinHeap{}, &MaxHeap{}}
}

func (this *MedianFinder) AddNum(num int) {
    // 首先判断最小堆是否为空。如果为空，直接将元素添加到最小堆
    if this.minhp.Len() == 0 {
        heap.Push(this.minhp, num)
        return
    }
    // 如果最小堆不为空，根据两个堆的元素数量关系，决定元素应添加到哪个堆。目的是保持两个堆中的元素数量差不超过1
    // 如果最小堆的元素更多，新元素会被加到最小堆，然后将最小堆的最小元素移动到最大堆中
    // 如果两个堆的元素数量相同，新元素会被加到最大堆，然后将最大堆的最大元素移动到最小堆中
    if this.minhp.Len() > this.maxhp.Len() {
        heap.Push(this.minhp, num)    // 放入右侧的最小堆 
        heap.Push(this.maxhp, heap.Pop(this.minhp).(int))    // 最小堆给最大堆一个数
    } else {
        //一样多
        heap.Push(this.maxhp, num)    // 放入左侧的最大堆 
        heap.Push(this.minhp, heap.Pop(this.maxhp).(int))    // 最大堆给最小堆一个数
    }
}

func (this *MedianFinder) FindMedian() float64 {
    if this.minhp.Len() > this.maxhp.Len() {    // 右边的最小堆比左边的最大堆数量多，中位数是元素数量较多的最小堆的堆顶
        return float64((*this.minhp)[0])
    } else {    // 如果两个堆的元素数量相同，中位数是两个堆顶元素的平均值
        return float64((*this.minhp)[0]+(*this.maxhp)[0]) / 2
    }
}
```

分析

```go
一句话题解：左边大顶堆，右边小顶堆，小的加左边，大的加右边，平衡俩堆数，新加就弹出，堆顶给对家，奇数取多的，偶数取除2.

利用了heap包  "container/heap"


func (hp *MaxHeap) Push(x interface{}) {
    *hp = append(*hp, x.(int))    // 使用了类型断言
}
类型断言用于检查运行时的类型是否符合预期，并提取该类型的值。在这个例子中，x.(int)是一个类型断言，它检查x是否为int类型，并在是的情况下提取int值。
```

# 排序

## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/)

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```go
func findKthLargest(nums []int, k int) int {
    start, end := 0, len(nums)-1
    for {
        if start >= end {
            return nums[end]
        }
        p := quickSortPartition(nums, start, end)
        if p+1 == k { // 第K大   0 1 2 对应：第一大 第二大 第三大
            return nums[p]
        } else if p+1 < k { // 对p的右边数组进行分治, 即对 [p+1,right]进行分治
            start = p + 1
        } else { // 对p的左边数组进行分治, 即对 [left,p-1]进行分治
            end = p - 1
        }
    }
}

func quickSortPartition(nums []int, start, end int) int {
    // 从大到小排序
    pivot := nums[end]
    for i := start; i < end; i++ {
        if nums[i] > pivot { // 大的放左边
            nums[start], nums[i] = nums[i], nums[start]
            start++
        }
    }
    // for循环完毕, nums[start]左边的值, 均大于nums[start]右边的值
    nums[start], nums[end] = nums[end], nums[start] // 此时nums[end]是nums[start]右边最大的值，需要交换一下
    return start                                    // 确定了nums[start]的位置
}
```

分析

代码参考+原理分析  https://leetcode.cn/problems/kth-largest-element-in-an-array/solution/partition-zhu-shi-by-da-ma-yi-sheng/

```go
便捷：
func findKthLargest(nums []int, k int) int {
    n := len(nums)
    sort.Ints(nums)
    return nums[n-k]
}
```

## [912. 排序数组](https://leetcode.cn/problems/sort-an-array/description/)

给你一个整数数组 `nums`，请你将该数组升序排列。

```go
// 这种在极端情况下会超时，但是一般不会，易于理解
func sortArray(nums []int) []int {
    var quick func(left, right int)
    quick = func(left, right int) {
        // 递归终止条件
        if left >= right {
            return
        }
        pivot := nums[right] // 左右指针及主元
        start, end := left, right    
        for i := start; i < end; i++ {    // start前面的都是小于pivot的
            if nums[i] < pivot {
                nums[start], nums[i] = nums[i], nums[start]
                start++
            }
        }
        nums[start], nums[end] = nums[end], nums[start]    // 确定了start的位置
        quick(left, start-1)
        quick(start+1, right)
    }
    quick(0, len(nums)-1)
    return nums
}


// 这种也会超时......  解决方法：随机选取pivot
func sortArray(nums []int) []int {
    var quick func(left, right int)
    quick = func(left, right int) {
        // 递归终止条件
        if left > right {
            return
        }
        i, j, pivot := left, right, nums[left]    // 左右指针及主元
        for i < j {
            // 寻找小于主元的右边元素
            for i<j && nums[j]>=pivot {
                j--
            }
            // 寻找大于主元的左边元素
            for i<j && nums[i]<=pivot {
                i++
            }
            // 交换i, j下标元素
            nums[i], nums[j] = nums[j], nums[i]    // 交换之后，nums[i] < nums[pivot] < nums[j]
        }
        // 此时nums[left]还在第一位，需要和小于它的nums[i]，或是i,j重叠处交换一下
        nums[i], nums[left] = nums[left], nums[i]
        quick(left, i-1)
        quick(i+1,right)
    }
    quick(0, len(nums)-1)
    return nums
}
```

分析

```go
// 此时nums[left]还在第一位，需要和小于它的nums[i]，或是i,j重叠处交换一下
nums[i], nums[left] = nums[left], nums[i]

每一轮排序最后要交换一下
```

## [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

```go
func search(nums []int, target int) int {
    n := len(nums)
    if n == 1 {
        if nums[0] == target {
            return 0
        } else {
            return -1
        }
    }
    left, right, mid := 0, n-1, 0
    /*
    将数组一分为二，其中一定有一个是有序的，另一个可能是有序，也能是部分有序。
    此时有序部分用二分法查找。无序部分再一分为二，其中一个一定有序，另一个可能有序，可能无序。就这样循环.
    */
    for left <= right {
        mid = (left+right) / 2
        if nums[mid] == target {    // 判断是否找到target
            return mid
        }
        if nums[0] <= nums[mid] {    // 0~mid是有序的    这里必须加=号
            if nums[0]<=target && target<nums[mid] {    // target在0~mid范围内，进行查找
                right = mid - 1
            } else {    // target不在0~mid范围内，在无序的mid+1~n-1范围内重新查找
                left = mid + 1
            }
        } else {    // 0~mid不是有序的，mid~n是有序的
            if nums[mid]<target && target<=nums[n-1] {    // target在mid~n-1范围内，进行查找
                left = mid + 1
            } else {    // target不在mid~n-1范围内，在无序的0~mid-1范围内重新查找
                right = mid - 1
            }
        }
    }
    return -1
}
```

分析

```go
有序数组+logN ========== 二分法
```

## [704. 二分查找](https://leetcode.cn/problems/binary-search/)

给定一个 `n` 个元素有序的（升序）整型数组 `nums` 和一个目标值 `target`  ，写一个函数搜索 `nums` 中的 `target`，如果目标值存在返回下标，否则返回 `-1`。

```go
func search(nums []int, target int) int {
    high := len(nums)
    low := 0
    for low < high {
        mid := low + (high-low)/2
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            low = mid + 1
        } else {
            high = mid    // 因为是左闭右开区间，所以这里不能是high = mid - 1，mid-1有可能是答案
        }
    }
    return -1
}
```

分析

```go
有序数组+logN ========== 二分法
注意边界条件
```

## [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/)

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

```go
func merge(intervals [][]int) [][]int {
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    res := [][]int{}
    prev := intervals[0]
    // 合并区间
    for i := 1; i < len(intervals); i++ {
        cur := intervals[i]
        // 前一个区间的右边界和当前区间的左边界进行比较，判断有无重合
        if prev[1] < cur[0] { // 没有重合，说明之后遍历的区间的左边界都会大于这个prev的右边界，不用再考虑这个prev了
            res = append(res, prev) // 前一个区间合并完毕，加入结果集
            prev = cur
        } else { // 有重合
            prev[1] = max(prev[1], cur[1]) // 合并后的区间右边界为较大的那个
        }
    }
    // 当考察完最后一个区间，后面没区间了，遇不到不重合区间，最后的 prev 没推入 res。 要单独补上
    res = append(res, prev)
    return res
}

func max(a, b int) int {
    if a > b { return a }
    return b
}
```

分析

```go
思路是先排序，再合并，遇到不重合再推入 prev
```

## [451. 根据字符出现频率排序](https://leetcode.cn/problems/sort-characters-by-frequency/)

给定一个字符串 `s` ，根据字符出现的 **频率** 对其进行 **降序排序** 。一个字符出现的 **频率** 是它出现在字符串中的次数。

返回 *已排序的字符串* 。如果有多个答案，返回其中任何一个。

```go
func frequencySort(s string) string {
    countMap := map[byte]int{}
    for i := range s {
        countMap[s[i]]++
    }

    type pair struct {
        ch  byte
        count int
    }
    pairs := make([]pair, 0, len(countMap))
    for k, v := range countMap {
        pairs = append(pairs, pair{k, v})
    }
    sort.Slice(pairs, func(i, j int) bool { return pairs[i].count > pairs[j].count })

    ans := make([]byte, 0, len(s))
    for _, p := range pairs {
        ans = append(ans, bytes.Repeat([]byte{p.ch}, p.count)...)
    }
    return string(ans)
}
```

分析

```go
如果要求不用map，那就直接直接用数组[]int去记录出现的次数。
虽然有大小写字母，但一个字母的小写比大写的ASCII数值大了32，还是可以存得下的
```

## [35. 搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

```go
func searchInsert(nums []int, target int) int {
    n := len(nums)
    left, right := 0, n
    ans := n
    for left < right {
        mid := (right - left)/2 + left
        if target <= nums[mid] {
            ans = mid
            right = mid
        } else {
            left = mid + 1
        }
    }
    return ans
}
```

分析

```go

```

## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

```go
// 这道题也可以用35题的函数来做
func searchRange(nums []int, target int) []int {
    leftmost := sort.SearchInts(nums, target)
    if leftmost == len(nums) || nums[leftmost] != target {
        return []int{-1, -1}
    }
    rightmost := sort.SearchInts(nums, target + 1) - 1
    return []int{leftmost, rightmost}
}
```

分析

```go
其实我们要找的就是数组中「第一个等于 target 的位置」和「第一个大于 target 的位置减一」

SearchInts 在排序的整数切片中搜索 x 并返回 Search 指定的索引。 
如果 x 不存在，则返回值是插入 x 的索引（它可能是 len(a)）。 切片必须按升序排序。
```

## [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/)

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target` ，如果 `target` 在矩阵中，返回 `true` ；否则，返回 `false` 。

```go
func searchMatrix(matrix [][]int, target int) bool {
    m, n := len(matrix), len(matrix[0])
    i := sort.Search(m*n, func(i int) bool {
        return matrix[i/n][i%n] >= target
    })
    return i < m*n && matrix[i/n][i%n] == target
}
```

分析

```go
方法名
sort.Search()

使用模板
index := sort.Search(n int,f func(i int) bool) int

主要功能
该函数使用二分查找的方法，会从[0, n)中取出一个值index，index为[0, n)中最小的使函数f(index)为True的值，并且f(index+1)也为True。
如果无法找到该index值，则该方法为返回n。

常用场景
该方法一般用于从一个已经排序的数组中找到某个值所对应的索引。
或者从字符串数组中，找到满足某个条件的最小索引值，比如etcd中的键值范围查询就用到了该方法。

具体例子
func main() {
    a := []int{1,2,3,4,5}
    d := sort.Search(len(a), func(i int) bool { return a[i]>=3})
    fmt.Println(d)
}
执行结果：2
```

## [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 **互不相同** 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

```go
func findMin(nums []int) int {
    low, high := 0, len(nums) - 1
    for low < high {
        mid := low + (high - low) / 2
        if nums[mid] < nums[high] {
            high = mid
        } else {
            low = mid + 1
        }
    }
    return nums[low]
}
```

分析

<img src="https://assets.leetcode-cn.com/solution-static/153/1.png" style="zoom: 50%;" />

```go

```

# 栈/队列

## [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（`push`、`pop`、`peek`、`empty`）

```go
type MyQueue struct {
    stackIn  []int //输入栈
    stackOut []int //输出栈
}

func Constructor() MyQueue {
    return MyQueue{
        stackIn:  make([]int, 0),
        stackOut: make([]int, 0),
    }
}

// 往输入栈做push
func (this *MyQueue) Push(x int) {
    this.stackIn = append(this.stackIn, x)
}

// 在输出栈做pop，pop时如果输出栈数据为空，需要将输入栈全部数据导入，如果非空，则可直接使用
func (this *MyQueue) Pop() int {
    inLen, outLen := len(this.stackIn), len(this.stackOut)
    if outLen == 0 {
        if inLen == 0 {
            return -1
        }
        for i := inLen - 1; i >= 0; i-- {
            this.stackOut = append(this.stackOut, this.stackIn[i])
        }
        this.stackIn = []int{}      //导出后清空
        outLen = len(this.stackOut) //更新长度值
    }
    val := this.stackOut[outLen-1]
    this.stackOut = this.stackOut[:outLen-1]
    return val
}

func (this *MyQueue) Peek() int {
    val := this.Pop()
    if val == -1 {
        return -1
    }
    this.stackOut = append(this.stackOut, val)
    return val
}

func (this *MyQueue) Empty() bool {
    return len(this.stackIn) == 0 && len(this.stackOut) == 0
}
```

分析

```go

```

## [225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（`push`、`top`、`pop` 和 `empty`）。

```go
type MyStack struct {
    queue []int//创建一个队列
}

/** Initialize your data structure here. */
func Constructor() MyStack {
    return MyStack{   //初始化
        queue:make([]int,0),
    }
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int)  {
    //添加元素
    this.queue=append(this.queue,x)
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
    n:=len(this.queue)-1//判断长度
    val:=this.queue[n]
    this.queue=this.queue[:n]
    return val

}

/** Get the top element. */
func (this *MyStack) Top() int {
    //利用Pop函数，弹出来的元素重新添加
    val:=this.Pop()
    this.queue=append(this.queue,val)
    return val
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
    return len(this.queue)==0
}
```

分析

![](https://assets.leetcode-cn.com/solution-static/225/225_fig1.gif)

![](https://assets.leetcode-cn.com/solution-static/225/225_fig2.gif)

```go

```

## [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/description/)

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

```go
func isValid(s string) bool {
    hash := map[byte]byte{')': '(', ']': '[', '}': '{'}
    stack := []byte{}    // 注意是string中的每个字符是byte类型
    if s == "" {
        return true
    }
    for i:=0; i<len(s); i++ {
        if s[i]=='(' || s[i]=='{' ||s[i]=='[' {
            stack = append(stack, s[i])        // 注意是string中的每个字符是byte类型
        } else if len(stack)>0 && stack[len(stack)-1]==hash[s[i]] {
            stack = stack[:len(stack)-1]
        } else {
            return false
        }
    }
    return len(stack) == 0
}
```

分析

```go
由于栈结构的特殊性，非常适合做对称匹配类的题目
```

## [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

答案

```go
func longestValidParentheses(s string) int {
    maxAns := 0
    stack := []int{}
    // 如果一开始栈为空，第一个字符为左括号时会将其放入栈中，这样就不满足提及的「最后一个没有被匹配的右括号的下标」
    // 为了保持统一，一开始往栈中放入一个 −1
    stack = append(stack, -1)
    for i := 0; i < len(s); i++ {
        if s[i] == '(' {    // 对于遇到的每个 ‘(’，将其下标放入栈中
            stack = append(stack, i)
        } else {    // 对于遇到的每个 ‘)’ ，先弹出栈顶元素表示匹配了当前右括号
            stack = stack[:len(stack)-1]
            if len(stack) == 0 {    // 栈为空，说明当前的右括号为没有被匹配的右括号
                stack = append(stack, i)    // 将其下标放入栈中来更新「最后一个没有被匹配的右括号的下标」
            } else {    // 栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
                maxAns = max(maxAns, i - stack[len(stack)-1])
            }
        }
    }
    return maxAns
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

分析

动画过程：https://leetcode.cn/problems/longest-valid-parentheses/solutions/314683/zui-chang-you-xiao-gua-hao-by-leetcode-solution/?envType=study-plan-v2&envId=top-100-liked

```go
始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」
这样的做法主要是考虑了边界条件的处理，栈里其他元素维护左括号的下标

对于遇到的每个 ‘(’ ，我们将它的下标放入栈中
对于遇到的每个 ‘)’ ，我们先弹出栈顶元素表示匹配了当前右括号：
    如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
    如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
```

## [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/description/)

给出由小写字母组成的字符串 `S`，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

```go
func removeDuplicates(s string) string {
    stack := make([]byte, 0)
    for i:=0; i<len(s); i++ {
        // 栈不空 且 与栈顶元素相等
        if len(stack)>0 && stack[len(stack)-1]==s[i] {
            // 弹出栈顶元素 并 忽略当前元素(s[i])
            stack = stack[:len(stack)-1]
        } else {
            stack = append(stack, s[i])
        }
    }
    return string(stack)
}
```

分析

```go
栈用来存放遍历过的元素
当遍历当前元素时，去栈里看一下是不是遍历过相同数值的相邻元素，然后再去做对应的消除操作
```

## [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/description/)

给你一个字符串数组 `tokens` ，表示一个根据 [逆波兰表示法](https://baike.baidu.com/item/%E9%80%86%E6%B3%A2%E5%85%B0%E5%BC%8F/128437) 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

```go
func evalRPN(tokens []string) int {
    stack := make([]int, 0)
    for _, token := range tokens {
        val, err := strconv.Atoi(token)
        if err == nil {
            stack = append(stack, val)
        } else {
            num1, num2 := stack[len(stack)-2], stack[len(stack)-1]
            stack = stack[:len(stack)-2]
            switch token {
            case "+":
                stack = append(stack, num1+num2)
            case "-":
                stack = append(stack, num1-num2)
            case "*":
                stack = append(stack, num1*num2)
            case "/":
                stack = append(stack, num1/num2)
            }
        }
    }
    return stack[0]
}
```

分析

```go

```

## [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

```go
// push:当前元素e入队时，相对于前面的元素来说，e最后进入窗口，e一定是最后离开窗口，
// 那么前面比e小的元素，不可能成为最大值，因此比e小的元素可以“压缩”掉

// pop:在元素入队时，是按照下标i入队的，因此队列中剩余的元素，其下标一定是升序的。
// 窗口大小不变，最先被排除出窗口的，一直是下标最小的元素，设为r。元素r在队列中要么是头元素，要么不存在。

// 封装单调队列的方式解题
type MyQueue struct {
    queue []int
}

func NewMyQueue() *MyQueue {
    return &MyQueue{
        queue: make([]int, 0),
    }
}

func (m *MyQueue) Front() int {
    return m.queue[0]
}

func (m *MyQueue) Back() int {
    return m.queue[len(m.queue)-1]
}

func (m *MyQueue) Empty() bool {
    return len(m.queue) == 0
}

// 如果push的元素value大于队列尾的数值，那么就将队列尾的元素弹出，直到value小于等于队列尾元素的数值为止（或队列为空）
func (m *MyQueue) Push(val int) {
    for !m.Empty() && val > m.Back() {
        m.queue = m.queue[:len(m.queue)-1]
    }
    m.queue = append(m.queue, val)
}

// 如果窗口移除的元素value等于单调队列头元素，那么队列弹出元素，否则不用任何操作
func (m *MyQueue) Pop(val int) {
    if !m.Empty() && val == m.Front() {
        m.queue = m.queue[1:]
    }
}

func maxSlidingWindow(nums []int, k int) []int {
    queue := NewMyQueue()
    length := len(nums)
    res := make([]int, 0)
    // 先将前k个元素放入队列
    for i := 0; i < k; i++ {
        queue.Push(nums[i])
    }
    // 记录前k个元素的最大值
    res = append(res, queue.Front())

    for i := k; i < length; i++ {
        // 滑动窗口移除最前面的元素
        queue.Pop(nums[i-k])
        // 滑动窗口添加最后面的元素
        queue.Push(nums[i])
        // 记录最大值
        res = append(res, queue.Front())
    }
    return res
}
```

分析

```go
维护元素单调递减的队列就叫做单调队列，即单调递减或单调递增的队列
```

## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

```go
func topKFrequent(nums []int, k int) []int {
    // 初始化一个map，用来存数字和数字出现的次数
    hashMap := make(map[int]int)
    res := make([]int, 0)
    for _,v := range nums {
        // 先查询map看一下v有没有存入map中，如果存在ok为true
        if _, ok := hashMap[v]; ok {
            // 不是第一次出现map的value数值+1
            hashMap[v]++
        } else {
            hashMap[v] = 1
            res = append(res, v)
        }
    }
    // 将res 按照map的value进行排序
    sort.Slice(res, func(i, j int) bool {     //利用O(nlogn)排序
        return hashMap[res[i]] > hashMap[res[j]]
    })
    return res[:k]
}
```

分析

```go

```

## [71. 简化路径](https://leetcode.cn/problems/simplify-path/)

给你一个字符串 `path` ，表示指向某一文件或目录的 Unix 风格 **绝对路径** （以 `'/'` 开头），请你将其转化为更加简洁的规范路径。

在 Unix 风格的文件系统中，一个点（`.`）表示当前目录本身；此外，两个点 （`..`） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，`'//'`）都被视为单个斜杠 `'/'` 。 对于此问题，任何其他格式的点（例如，`'...'`）均被视为文件/目录名称。

请注意，返回的 **规范路径** 必须遵循下述格式：

- 始终以斜杠 `'/'` 开头。
- 两个目录名之间必须只有一个斜杠 `'/'` 。
- 最后一个目录名（如果存在）**不能** 以 `'/'` 结尾。
- 此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 `'.'` 或 `'..'`）。

返回简化后得到的 **规范路径** 。

```go
func simplifyPath(path string) string {
    stack := []string{}
    for _, name := range strings.Split(path, "/") {
        // 对于「空字符串」以及「一个点」，我们实际上无需对它们进行处理
        // 遇到「两个点」时，需要将目录切换到上一级，因此只要栈不为空，就弹出栈顶的目录
        if name == ".." {
            if len(stack) > 0 {
                stack = stack[:len(stack)-1]
            }
        } else if name != "" && name != "." { 
            stack = append(stack, name) // 遇到「目录名」时，就把它放入栈
        }
    }
    // 将从栈底到栈顶的字符串用 / 进行连接，再在最前面加上 / 表示根目录
    return "/" + strings.Join(stack, "/")
}
```

分析

```go
s := strings.Split("/1/2", "/")
fmt.Println("length is :", len(s))
length is : 3， 元素分别为:"", "1", "2"


将给定的字符串 path 根据 / 分割成一个由若干字符串组成的列表，记为 names。
根据题目中规定的「规范路径的下述格式」，names 中包含的字符串只能为以下几种：
    空字符串。例如当出现多个连续的 /，就会分割出空字符串；
    一个点 .；
    两个点 ..；
    只包含英文字母、数字或 _ 的目录名
```

## [155. 最小栈](https://leetcode.cn/problems/min-stack/)

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类:

- `MinStack()` 初始化堆栈对象。
- `void push(int val)` 将元素val推入堆栈。
- `void pop()` 删除堆栈顶部的元素。
- `int top()` 获取堆栈顶部的元素。
- `int getMin()` 获取堆栈中的最小元素。

```go
type MinStack struct {
    stack []int
    minStack []int
}

func Constructor() MinStack {
    return MinStack{
        stack: []int{},
        minStack: []int{math.MaxInt64},
    }
}

func (this *MinStack) Push(x int)  {
    this.stack = append(this.stack, x)    // 一个新的元素入栈
    top := this.minStack[len(this.minStack)-1]    // 获取当前辅助栈的栈顶存储的最小值
    this.minStack = append(this.minStack, min(x, top))    // 与当前元素比较得出最小值，将这个最小值插入辅助栈中
}

func (this *MinStack) Pop()  {
    this.stack = this.stack[:len(this.stack)-1]    // 一个元素出栈
    this.minStack = this.minStack[:len(this.minStack)-1]    // 把辅助栈的栈顶元素也一并弹出
}

func (this *MinStack) Top() int {
    return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
    return this.minStack[len(this.minStack)-1]
}

func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}
```

分析

```go

```

## [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 `k` ，例如不会出现像 `3a` 或 `2[4]` 的输入。

```go
func decodeString(s string) string {
    stk := []string{}
    ptr := 0
    for ptr < len(s) {
        cur := s[ptr]
        if cur >= '0' && cur <= '9' {    // 如果当前的字符为数位，解析出一个数字（连续的多个数位）并进栈
            digits := getDigits(s, &ptr)
            stk = append(stk, digits)
        } else if (cur >= 'a' && cur <= 'z' || cur >= 'A' && cur <= 'Z') || cur == '[' {    // 如果当前的字符为字母或者左括号，直接进栈
            stk = append(stk, string(cur))
            ptr++
        } else {    // 如果当前的字符为右括号，开始出栈，一直到左括号出栈
            ptr++
            sub := []string{}
            for stk[len(stk)-1] != "[" {
                sub = append(sub, stk[len(stk)-1])
                stk = stk[:len(stk)-1]
            }
            for i := 0; i < len(sub)/2; i++ {    // 出栈序列反转后拼接成一个字符串
                sub[i], sub[len(sub)-i-1] = sub[len(sub)-i-1], sub[i]
            }
            stk = stk[:len(stk)-1]    // 左括号 '[' 出栈 
            repTime, _ := strconv.Atoi(stk[len(stk)-1])    // 取出栈顶的数字，即这个字符串应该出现的次数
            stk = stk[:len(stk)-1]    
            t := strings.Repeat(getString(sub), repTime)    // 根据这个次数和字符串构造出新的字符串并进栈
            stk = append(stk, t)
        }
    }
    return getString(stk)
}

func getDigits(s string, ptr *int) string {
    ret := ""
    for ; s[*ptr] >= '0' && s[*ptr] <= '9'; *ptr++ {
        ret += string(s[*ptr])
    }
    return ret
}

func getString(v []string) string {
    ret := ""
    for _, s := range v {
        ret += s
    }
    return ret
}
```

分析

```go

```

## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

给定一个整数数组 `temperatures` ，表示每天的温度，返回一个数组 `answer` ，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

```go
func dailyTemperatures(temperatures []int) []int {
    length := len(temperatures)
    ans := make([]int, length)
    stack := []int{}
    for i := 0; i < length; i++ {
        temperature := temperatures[i]
        // 如果栈为空，则直接将 i 进栈
        // 如果栈不为空，则比较栈顶元素对应的温度和当前温度，当前温度更大则移除栈顶
        for len(stack) > 0 && temperature > temperatures[stack[len(stack)-1]] {
            prevIndex := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            ans[prevIndex] = i - prevIndex
        }
        // 重复上述操作直到栈为空或者栈顶元素对应的温度小于等于当前温度，然后将 i 进栈
        stack = append(stack, i)    
    }
    return ans
}
```

分析

```go
判别是否需要使用单调栈，如果需要找到左边或者右边第一个比当前位置的数大或者小，则可以考虑使用单调栈；单调栈的题目如矩形米面积等等
“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景。

维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。
```

## [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

```go
func largestRectangleArea(heights []int) int {
    n := len(heights)
    left, right := make([]int, n), make([]int, n)
    for i := 0; i < n; i++ {
        right[i] = n
    }
    mono_stack := []int{}
    for i := 0; i < n; i++ {
        for len(mono_stack) > 0 && heights[mono_stack[len(mono_stack)-1]] >= heights[i] {
            right[mono_stack[len(mono_stack)-1]] = i
            mono_stack = mono_stack[:len(mono_stack)-1]
        }
        if len(mono_stack) == 0 {
            left[i] = -1
        } else {
            left[i] = mono_stack[len(mono_stack)-1]
        }
        mono_stack = append(mono_stack, i)
    }
    ans := 0
    for i := 0; i < n; i++ {
        ans = max(ans, (right[i] - left[i] - 1) * heights[i])
    }
    return ans
}

func max(x, y int) int {
    if x > y {
        return x
    }
    return y
}
```

分析

```go
首先单调栈的经典应用场景是，在一维数组中，对每一个数字，找到前/后面第一个比自己大/小的元素。

该题的思路是：
对数组中的每个元素，若假定以它为高，能够展开的宽度越宽，那么以它为高的矩形面积就越大。
因此，思路就是找到每个元素左边第一个比它矮的矩形和右边第一个比它矮的矩形，在这中间的就是最大宽度
最后对每个元素遍历一遍找到最大值即可。
```

# 图

## 迪杰斯特拉算法

```go
func dijkstra(graph [][]int, start int) []int {
    n := len(graph)         // 图中顶点个数
    visit := make([]int, n) // 标记已经作为中间结点完成访问的顶点
    dist := make([]int, n)  // 存储从起点到其他顶点的最短路径

    for i := 0; i < n; i++ {
        dist[i] = graph[start][i] // 初始化遍历起点
    }
    visit[start] = 1 // 标记初始顶点

    // 更新其他顶点最短路径，循环n次
    for i := 0; i < n; i++ {
    minDist := math.MaxInt // 存储从起点到其他未被访问的结点中的最短路径
    midNode := 0   // 中间结点

        // 遍历n个顶点，寻找未被访问且起始位置到该点距离最小的顶点
        for j := 0; j < n; j++ {
            if visit[j] == 0 && minDist > dist[j] {
                minDist = dist[j] // 更新未被访问结点的最短路径
                midNode = j       // 更新中间结点
            }
        }

        // 以midNode为中间结点，再循环遍历其他节点更新最短路径
        for j := 0; j < n; j++ {
            // 若该节点未被访问且找到更短路径即更新最短路径
            if visit[j] == 0 && dist[j] > dist[midNode]+graph[midNode][j] {
                dist[j] = dist[midNode] + graph[midNode][j]
            }
        }
        visit[midNode] = 1 // 标记已访问

    }
    return dist
}

const INF = 100000

func main() {
    // 带权值邻接矩阵
    var gp = [][]int{
        // a b c d e f g s
        {0, 1, INF, INF, 5, INF, INF, 7},   // a
        {1, 0, INF, 1, 3, INF, INF, INF},   // b
        {2, INF, 0, 1, 1, 9, 6, INF},       // c
        {INF, 1, INF, 0, 1, INF, 9, INF},   // d
        {5, 3, 1, INF, 0, INF, INF, INF},   // e
        {INF, INF, 9, INF, INF, 0, 3, 3},   // f
        {INF, INF, 6, 9, INF, 3, 0, INF},   // g
        {7, INF, INF, INF, INF, 3, INF, 0}, // s
    }
    dist := dijkstra(gp, 1)
    fmt.Println(dist)
}
```

分析

```go

```

## 200. 岛屿数量

```go
func numIslands(grid [][]byte) int {
    ans := 0
    m, n := len(grid), len(grid[0])
    var dfs func(i, j int)
    dfs = func(i, j int) {
        grid[i][j] = '0' // '1'（陆地）  '0'（水）
        // 每遍历到一块陆地，就把这块陆地和与之相连的陆地全部变成水
        if i-1 >= 0 && grid[i-1][j] == '1' {
            dfs(i-1, j)
        }
        if i+1 < m && grid[i+1][j] == '1' {
            dfs(i+1, j)
        }
        if j-1 >= 0 && grid[i][j-1] == '1' {
            dfs(i, j-1)
        }
        if j+1 < n && grid[i][j+1] == '1' {
            dfs(i, j+1)
        }
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                ans++
                dfs(i, j)
            }
        }
    }
    return ans
}
```

分析

```go

```

## [695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)

给你一个大小为 `m x n` 的二进制矩阵 `grid` 。

**岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在 **水平或者竖直的四个方向上** 相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

岛屿的面积是岛上值为 `1` 的单元格的数目。

计算并返回 `grid` 中最大的岛屿面积。如果没有岛屿，则返回面积为 `0` 。

```go
func maxAreaOfIsland(grid [][]int) int {
    ans := 0
    m, n := len(grid), len(grid[0])
    var dfs func(i, j int) int
    dfs = func(i, j int) int {
        count := 1
        grid[i][j] = 0 //   1（陆地）  0（水）
        // 每遍历到一块陆地，就把这块陆地和与之相连的陆地全部变成水
        if i-1 >= 0 && grid[i-1][j] == 1 {
            count += dfs(i-1, j)
        }
        if i+1 < m && grid[i+1][j] == 1 {
            count += dfs(i+1, j)
        }
        if j-1 >= 0 && grid[i][j-1] == 1 {
            count += dfs(i, j-1)
        }
        if j+1 < n && grid[i][j+1] == 1 {
            count += dfs(i, j+1)
        }
        return count
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == 1 {
                count := dfs(i, j)
                ans = max(ans, count)
            }
        }
    }
    return ans
}
```

分析

```go
注意和上一题的区别
二维数组的类型变了，一个是byte，一个是int
```

## [1971. 寻找图中是否存在路径](https://leetcode.cn/problems/find-if-path-exists-in-graph/)

有一个具有 `n` 个顶点的 **双向** 图，其中每个顶点标记从 `0` 到 `n - 1`（包含 `0` 和 `n - 1`）。图中的边用一个二维整数数组 `edges` 表示，其中 `edges[i] = [ui, vi]` 表示顶点 `ui` 和顶点 `vi` 之间的双向边。 每个顶点对由 **最多一条** 边连接，并且没有顶点存在与自身相连的边。

请你确定是否存在从顶点 `source` 开始，到顶点 `destination` 结束的 **有效路径** 。

给你数组 `edges` 和整数 `n`、`source` 和 `destination`，如果从 `source` 到 `destination` 存在 **有效路径** ，则返回 `true`，否则返回 `false` 。

```go
// 并查集
func validPath(n int, edges [][]int, source int, destination int) bool {
    p := make([]int, n)
    for i := range p {
        p[i] = i
    }
    var find func(x int) int
    find = func(x int) int {
        if p[x] != x {
            p[x] = find(p[x])
        }
        return p[x]
    }
    for _, e := range edges {
        p[find(e[0])] = find(e[1])
    }
    return find(source) == find(destination)
}


链接：https://leetcode.cn/problems/find-if-path-exists-in-graph/solutions/2025571/by-lcbin-96dp/

// DFS
func validPath(n int, edges [][]int, source int, destination int) bool {
    vis := make([]bool, n)    // 记录已经访问过的顶点
    g := make([][]int, n)
  // 先将 edges 转换成图 g
    for _, e := range edges {
        a, b := e[0], e[1]
        g[a] = append(g[a], b)
        g[b] = append(g[b], a)
    }
    var dfs func(int) bool
    dfs = func(i int) bool {
        if i == destination {
            return true
        }
        vis[i] = true
        for _, j := range g[i] {
            if !vis[j] && dfs(j) {
                return true
            }
        }
        return false
    }
    return dfs(source)
}

// BFS        
func validPath(n int, edges [][]int, source int, destination int) bool {
    passMap := map[int]int{}    // 存储已经遍历过的点
    queue := []int{source}        // 存储接下来需要遍历的点
    for len(queue) > 0{
        pop := queue[0]
        queue = queue[1:]
        if pop == destination{
            return true
        }
        nexts := findNext(edges,pop)    // 找到与当前点相连的其他点
        for _,v := range nexts{
            if _,ok:=passMap[v];ok{    // 如果点已经遍历过(在遍历过的集合中) 那就不入队
                continue
            }
            passMap[v] = 1    // 如果点没有被遍历过，当前点入队
            queue = append(queue,v)
        }
    }
      // 一直到队列为空（表示从出发点出发 所有与之能够连通的点都遍历过了，没有找到最终的节点）
    return false
}
func findNext(edges [][]int,point int)[]int{
    res := []int{}
    for _,v:=range edges{
      // edges[i] = [ui, vi] 表示顶点 ui 和顶点 vi 之间的双向边
        if v[0] == point{
            res = append(res,v[1])
        }
        if v[1] == point{
            res = append(res,v[0])
        }
    }
    return res
}
```

分析

```go
给一个图、两个点，判断两个点之间是否是连通的，这题一看就知道是经典的广度优先搜索。
使用广度优先搜索主要是有两点需要用到的：
    用一个集合来存储已经遍历过的点（不然就会重复遍历）
    用一个队列来存接下来需要遍历的点 （因为一个点的第二步可以往很多个点走）
在这道题当中，我们的目的是判断图中的一个点是否能够到达另一个点，那我们就从一个点出发，深度/广度优先的往下走，往下走的过程中
    要么遍历完从这个点能到达的所有点之后都没有遇到我们的目标点，此时 返回false
    要么遍历过程中遇到了我们的目标点，此时返回true
在遍历的过程中我们主要采取这样的做法：
    出发点进入队列
    循环判断队列不为空
        出队列一个点
        将这个点加入边遍历过的集合中
        判断这个点是否为终点，如果为终点就直接return true
        从图中找到与当前点相连的其他点
            如果点已经遍历过(在遍历过的集合中) 那就不入队。（因为已经被遍历过了，没有到达终点，这个点之后的点也遍历过了）
            如果点没有被遍历过，当前点入队
    一直到队列为空（表示从出发点出发 所有与之能够连通的点都遍历过了，没有找到最终的节点） return false
```

## [207. 课程表](https://leetcode.cn/problems/course-schedule/)

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程  `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
    g := make([][]int, numCourses)    // 图
    indeg := make([]int, numCourses)    // 每个节点的入度
    for _, p := range prerequisites {
        a, b := p[0], p[1]    // b -> a
        g[b] = append(g[b], a)    // b可以到达的节点
        indeg[a]++    // a的入度+1
    }
    q := []int{}    // 存储入度为0的节点
    for i, x := range indeg {    // 遍历每个节点的入度
        if x == 0 {
            q = append(q, i)
        }
    }
    cnt := 0    // 入度为0的节点的数量
  // 对于每个入度为 0 的节点，我们将其出度的节点的入度减 1，直到所有节点都被遍历到
    for len(q) > 0 {
        i := q[0]
        q = q[1:]
        cnt++
        for _, j := range g[i] {    // 节点i能到达哪些节点
            indeg[j]--    // 能到达的节点的入度-1
            if indeg[j] == 0 {
                q = append(q, j)
            }
        }
    }
    return cnt == numCourses
}
```

分析

```go
拓扑排序
  给定一个包含 n 个节点的有向图 G，我们给出它的节点编号的一种排列，如果满足：
  对于图 G 中的任意一条有向边 (u,v)，u 在排列中都出现在 v 的前面。
  那么称该排列是图 G 的「拓扑排序」

考虑拓扑排序中最前面的节点，该节点一定不会有任何入边，也就是它没有任何的先修课程要求。当我们将一个节点加入答案中后，我们就可以移除它的所有出边，代表着它的相邻节点少了一门先修课程的要求。如果某个相邻节点变成了「没有任何入边的节点」，那么就代表着这门课可以开始学习了。按照这样的流程，我们不断地将没有入边的节点加入答案，直到答案中包含所有的节点（得到了一种拓扑排序）或者不存在没有入边的节点（图中包含环）

将课程看作图中的节点，先修课程看作图中的边，那么我们可以将本题转化为判断有向图中是否存在环。
具体地，我们可以使用拓扑排序的思想，对于每个入度为 0 的节点，我们将其出度的节点的入度减 1，直到所有节点都被遍历到。
如果所有节点都被遍历到，说明图中不存在环，那么我们就可以完成所有课程的学习；否则，我们就无法完成所有课程的学习。
```

## [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回 *直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`* 。

```go
// 广度优先搜索，一层一层扩展
func orangesRotting(grid [][]int) int {
    if grid == nil || len(grid) == 0 {
        return 0
    }
    //按照上右下左方向进行扩展
    dx := []int{-1, 0, 1, 0}
    dy := []int{0, 1, 0, -1}
    //行列值
    row := len(grid)
    col := len(grid[0])
    res := 0 //腐烂完成的时间
    queue := make([]int, 0)
    //首先找到一开始就是腐烂的橘子，将其作为一层
    for i := 0; i < row; i++ {
        for j := 0; j < col; j++ {
            if grid[i][j] == 2 {
                //存入映射关系（优秀的方式）
                queue = append(queue, i*col+j)
            }
        }
    }
    //bfs搜索
    for len(queue) != 0 {
        res++                 //每搜完一层，则时间加一分钟
        cursize := len(queue) //保存当前层的长度
        for i := 0; i < cursize; i++ {
            node := queue[0]
            queue = queue[1:]
            r, c := node/col, node%col
            for k := 0; k < 4; k++ {
                nr := r + dx[k]
                nc := c + dy[k]
                if nr >= 0 && nr < row && nc >= 0 && nc < col && grid[nr][nc] == 1 {
                    grid[nr][nc] = 2                 //将新鲜橘子腐烂
                    queue = append(queue, nr*col+nc) //将腐烂橘子入队
                }
            }
        }
    }
    //判断还有没有新鲜橘子，有就返回-1
    for i := 0; i < row; i++ {
        for j := 0; j < col; j++ {
            if grid[i][j] == 1 {
                return -1
            }
        }
    }
    //因为res在计算层的时候，把最开始的腐烂橘子也记为一层，
    //所以结果为res-1
    //存在一个特殊情况，及[[0]]，此时，res就为0，所以不需要-1
    if res == 0 {
        return res
    } else {
        return res - 1
    }
}
```

分析

```go

```

# 技巧

## [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

```go
func singleNumber(nums []int) int {
    single := 0
    for _, num := range nums {
        single ^= num
    }
    return single
}
```

分析

```go
对于这道题，可使用异或运算 ⊕。异或运算有以下三个性质。

任何数和 0 做异或运算，结果仍然是原来的数，即 a⊕0=a
任何数和其自身做异或运算，结果是 0，即 a⊕a=0
异或运算满足交换律和结合律，即 a⊕b⊕a = b⊕a⊕a = b⊕(a⊕a) = b⊕0 = b
```

## [169. 多数元素](https://leetcode.cn/problems/majority-element/)

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```go
func majorityElement(nums []int) int {
    res, count := 0, 0
    for _, num := range nums {
        if count == 0 {
            res = num    // 找到一个新的众数，维护它
            count = 1
        } else {
            if res == num {        // 新的数等于众数
                count++
            } else {            // 新的数不等于众数
                count--
            }
        }
    }
    return res
}
```

分析

```go
摩尔投票法
    如果我们把多数元素记为 +1，把其他数记为 −1，将它们全部加起来，显然和大于 0
```

## [75. 颜色分类](https://leetcode.cn/problems/sort-colors/)

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，**[原地](https://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

```go
func sortColors(nums []int) {
    p0, p1 := 0, 0    // p0交换0，p1交换1
    for i, c := range nums {
        if c == 0 {    // 找到了 0
            nums[i], nums[p0] = nums[p0], nums[i]
            if p0 < p1 {    // 此时p0的后面一定是1，为了防止这个1被交换到后面，需要再交换一次
                nums[i], nums[p1] = nums[p1], nums[i]    // nums[i] 的值为 1，将这个 1 放到「头部」的末端
            }
            p0++
            p1++
        } else if c == 1 {    // 找到了 1
            nums[i], nums[p1] = nums[p1], nums[i]
            p1++
        }
    }
}
```

分析

```go
思路一
统计出数组中 0,1,2 的个数，再根据它们的数量，重写整个数组
缺点是两次遍历

思路二
单指针。在第一次遍历中，我们将数组中所有的 0 交换到数组的头部。在第二次遍历中，我们将数组中所有的 1 交换到头部的 0 之后。此时，所有的 2 都出现在数组的尾部，这样我们就完成了排序。
缺点是两次遍历

思路三
双指针。使用两个指针分别用来交换 0 和 1。
```

## [31. 下一个排列](https://leetcode.cn/problems/next-permutation/)

整数数组的一个 **排列**  就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须 **[原地](https://baike.baidu.com/item/%E5%8E%9F%E5%9C%B0%E7%AE%97%E6%B3%95)** 修改，只允许使用额外常数空间。

```go
func nextPermutation(nums []int) {
    if len(nums) <= 1 {
        return
    }

    i, j, k := len(nums)-2, len(nums)-1, len(nums)-1

    // 从后向前 查找第一个相邻升序的元素对 (i,j)，满足 A[i]<A[j],此时 [j,end) 必然是降序 eg: 123654  i=2, j=3
    for i >= 0 && nums[i] >= nums[j] {
        i--
        j--
    }

    if i >= 0 { // 不是最后一个排列
        // 在 [j,end) 从后向前 查找第一个满足 A[i] < A[k] eg: 123654 i=2, j=3, k=5
        for nums[i] >= nums[k] {
            k--
        }
        // 将一个 尽可能小的「大数」 与前面的「小数」进行交换  eg: 123654 => 124653 
        nums[i], nums[k] = nums[k], nums[i]  
    }

    // 将「大数」换到前面后，需要将「大数」后面的所有数 重置为升序     eg: 124653 => 124356
    for i, j := j, len(nums)-1; i < j; i, j = i+1, j-1 {
        nums[i], nums[j] = nums[j], nums[i]
    }
}
```

分析

```go
我们希望下一个数 比当前数大，这样才满足 “下一个排列” 的定义。
因此只需要 将后面的「大数」与前面的「小数」交换，就能得到一个更大的数。比如 123456，将 5 和 6 交换就能得到一个更大的数 123465。

我们还希望下一个数 增加的幅度尽可能的小，这样才满足“下一个排列与当前排列紧邻“的要求。
为了满足这个要求，我们需要：在 尽可能靠右的低位 进行交换，需要 从后向前 查找
将一个 尽可能小的「大数」 与前面的「小数」交换。比如 123465，下一个排列应该把 5 和 4 交换而不是把 6 和 4 交换
将「大数」换到前面后，需要将「大数」后面的所有数 重置为升序，升序排列就是最小的排列。以 123465 为例：首先按照上一步，交换 5 和 4，得到 123564；然后需要将 5 之后的数重置为升序，得到 123546。显然 123546 比 123564 更小，123546 就是 123465 的下一个排列
```

## [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

```go
// 二分法
func findDuplicate(nums []int) int {
    lo, hi := 1, len(nums)-1
    for lo < hi {
        mid := (hi - lo) / 2 + lo    // 重复数要么落在[1, mid]，要么落在[mid + 1, n]
        count := 0    // 遍历原数组，统计 <= mid 的元素个数，记为 count
        for i := 0; i < len(nums); i++ {
            if nums[i] <= mid {
                count++
            }
        }
        if count > mid {    // 说明有超过 mid 个数落在[1, mid]，但该区间只有 mid 个“坑”，说明重复的数落在[1, mid]
            hi = mid
        } else {    // 说明重复数落在[mid + 1, n]
            lo = mid + 1
        }
    }
    return lo
}

// 快慢指针
func findDuplicate(nums []int) int {
    slow, fast := 0, 0
    for {
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast {    // 快慢指针重叠，fast重置为头指针并开始和slow指针一起前进
            fast = 0
            for {
                if slow == fast {
                    return slow
                }
                slow = nums[slow]
                fast = nums[fast]
            }
        }
    }
}
```

分析

```go
二分法
对值域二分。重复数落在 [1, n] ，可以对 [1, n] 这个值域二分查找。
对重复数所在的区间继续二分，直到区间闭合，重复数就找到了

快慢指针法
题目说数组必存在重复数，所以 nums 数组肯定可以抽象为有环链表。
有环链表，重复数就是入环口
```

![](https://pic.leetcode-cn.com/a393fd88e07b576de4d603fcccd47539e6648273a7f6626760b95ec28d2343b7-%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200526201809.png)

## 约瑟夫环

代码

```go
int cir(int n,int m)
{
    int p=0;
    for(int i=2;i<=n;i++)
    {
        p=(p+m)%i;
    }
    return p+1;
}
```

分析

https://blog.csdn.net/u011500062/article/details/72855826

```
f(N,M) = (f(N−1,M)+M) % N
```

# 并发编程

## 按顺序打印

### 方法1

代码

```go
package main

import (
    "fmt"
    "sync"
)

const (
    MAX     = 20 // 打印多少值
    GoCount = 4  // 几个协程
)

func main() {
    fmt.Println(solution(MAX, GoCount))
}

func solution(max, goCount int) *[]int {
    lock := sync.Mutex{}    // 保证对共享变量count和result的访问是线程安全的
    wg := sync.WaitGroup{}  // 用来等待所有协程执行完毕
    result := make([]int, 0, max)

    count := 1  // 记录当前要打印的数字
    wg.Add(goCount) // 表示有goCount个协程需要等待。
    for i := 0; i < goCount; i++ {  // i是协程的编号， 协程i只会打印满足now % GoCount == i的数字
        go func(i int) {
            // 每个协程进入一个无限循环，直到所有数字都打印完毕
            for {
                lock.Lock()
                now := count
                if now > max {  // 判断当前计数now是否大于max
                    lock.Unlock()   // 如果大于，则释放锁并退出当前协程
                    wg.Done()
                    return
                }
                if now%goCount == i {   // 如果当前计数now模goCount的余数等于当前协程的编号i
                // 则表示当前协程负责打印这个数字。将数字添加到result中，并将count自增
                    //fmt.Println(now)
                    result = append(result, now)
                    count++
                }
                lock.Unlock()
            }
        }(i)
    }
    wg.Wait()   // 通过wg.Wait()阻塞主协程，直到所有goCount个协程都执行完毕
    return &result
}
```

分析

```
打印的顺序实际上是每个协程按顺序获取锁，然后按照逻辑条件决定是否将数字加入结果中
这种方法有锁的争抢

lock：使用sync.Mutex互斥锁，以保护共享资源count和result，避免并发访问时出现数据竞态。
wg：使用sync.WaitGroup来同步所有goroutine，确保它们都完成任务后再继续执行。

sync.WaitGroup 的核心功能是提供一种机制，确保一组 goroutine 都完成工作后再继续执行后续的代码。它通常用于主 goroutine 等待其他 goroutine 完成执行。

Add(delta int):用于设置需要等待的 goroutine 的数量，delta 可以是正数或负数。通常在启动 goroutine 之前调用这个方法，参数一般是正数，表示增加需要等待的 goroutine 数量。

Done():当一个 goroutine 完成任务时，它应该调用 Done() 方法，表示这个 goroutine 已经完成。Done() 实际上是 Add(-1) 的简化形式。

Wait():会阻塞调用它的 goroutine，直到 WaitGroup 计数器减为 0。也就是说，Wait() 会一直等待，直到所有的 goroutine 都调用 Done() 完成了工作。

性能测试基于go test bench
3         390073667 ns/op
```

### 方法2

代码

```go
package main

import (
	"fmt"
	"sync"
)

const (
	MAX     = 20 // 打印多少值
	GoCount = 4  // 几个协程
)

func main() {
	fmt.Println(solution2(MAX, GoCount))
}

func solution2(max, goCount int) *[]int {
	result := make([]int, 0, max)
	wgLine := make([]*sync.WaitGroup, goCount) // 控制不同 goroutine 的执行顺序
	wg := &sync.WaitGroup{}                    // 等待所有 goroutine 的完成

	// 循环创建 goCount 个 goroutine
	// 每个 goroutine 都有一个自己的 WaitGroup（selfWg）和一个指向下一个 goroutine 的 WaitGroup（nextWg）
	for i := 0; i < goCount; i++ {
		wgLine[i] = &sync.WaitGroup{}
		wgLine[i].Add(1)
	}

	count := 1
	wg.Add(goCount)
	for i := 0; i < goCount; i++ { // 对于每个 goroutine
		go func(max int, selfWg, nextWg *sync.WaitGroup) {
			for {
				selfWg.Wait() // 在开始时等待自己的 WaitGroup（selfWg）
				if count > max {
					wg.Done() // 一个goroutine完成任务
					// 下面这句不能删除，因为 selfWg 的计数将在别的goroutine变为负数，而 WaitGroup 的计数不能小于 0，这会导致程序崩溃并抛出 panic 错误
					selfWg.Add(1) // 重新加一个等待计数到 selfWg
					// 这句也不能删除，因为去掉它后下一个 goroutine 的 WaitGroup 永远不会被解锁，所以所有 goroutine 卡住，导致死锁
					nextWg.Done() // 触发下一个 goroutine 的 WaitGroup，然后退出
					return
				}
				fmt.Println("it is goroutine ", i, " printed ", count)
				result = append(result, count)
				count++
				selfWg.Add(1) // 当前 goroutine 重新为自己的 WaitGroup 加一
				nextWg.Done() // 触发下一个 goroutine 的 WaitGroup
			}
		}(max, wgLine[i], wgLine[(i+1)%goCount])

		if i == 0 { // 手动触发第一个 goroutine
			wgLine[0].Done() // 这里可以控制第几个goroutine打印第一个数字
		}
	}
	wg.Wait()
	return &result
}

```

分析

```
这种方法没有锁的争抢

每个 goroutine 使用两个 WaitGroup 对象来同步：
    selfWg 用于等待当前 goroutine 的执行，
    nextWg 用于通知下一个 goroutine 可以开始执行。

性能测试基于go test bench
36          31884292 ns/op
```


### 方法3

代码

```go
package main

import (
	"fmt"
	"sync"
)

const (
	MAX     = 100 // 打印多少值
	GoCount = 4   // 几个协程
)

func main() {
	AlternatePrint2()
}

func AlternatePrint2() {
	chs := make([]chan int, GoCount)
	wg := sync.WaitGroup{}
	wg.Add(GoCount)
	for i := 0; i < GoCount; i++ {
        // 容量为1的话可以允许设置GoCount为1
        // 容量为0的无缓冲通道GoCount必须大于1
        // 区别在于 chs[(i+1)%GoCount] <- count 会不会卡住
		chs[i] = make(chan int, 1)  
	}
	count := 1 // 递增的数字，用于打印
	for i := 0; i < GoCount; i++ {
		// 每个 Goroutine 通过无缓冲通道 chs[i] 来等待执行权限
		go func(i int) {
			for {
				// 获得执行权
				count = <-chs[i]
				if count > MAX {
					fmt.Printf("go程%d done\n", i)
					wg.Done()
					chs[(i+1)%GoCount] <- count
					return
				}

				// 执行代码
				fmt.Printf("go程%d：%d\n", i, count)
				count++

				// 当前协程执行完毕后，将执行权交给下一个协程
				chs[(i+1)%GoCount] <- count
			}
		}(i)
	}

	// 给第1个协程执行权，第1个协程的序号是0
	chs[0] <- count

	// 等待协程执行完成
	wg.Wait()
}
```

分析

```
channel
```

## 协程池

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

// 定义一个任务类型
type Task func()

// Goroutine Pool 结构体
type Pool struct {
	taskQueue   chan Task  // 任务队列
	workerCount int        // 协程数量
	wg          sync.WaitGroup // 用于等待所有任务完成
}

// 创建一个新的 Goroutine Pool
func NewPool(workerCount int, taskQueueSize int) *Pool {
	return &Pool{
		taskQueue:   make(chan Task, taskQueueSize),
		workerCount: workerCount,
	}
}

// 工作协程，从任务队列中获取任务并执行
func (p *Pool) worker(workerID int) {
	for task := range p.taskQueue {
		fmt.Printf("Worker %d: Started task\n", workerID)
		task() // 执行任务
		fmt.Printf("Worker %d: Finished task\n", workerID)
		p.wg.Done() // 标记任务完成
	}
}

// 提交一个任务到任务队列
func (p *Pool) Submit(task Task) {
	p.wg.Add(1)
	p.taskQueue <- task
}

// 启动所有协程
func (p *Pool) Run() {
	for i := 0; i < p.workerCount; i++ {
		go p.worker(i)
	}
}

// 关闭任务队列并等待所有任务完成
func (p *Pool) Shutdown() {
	close(p.taskQueue) // 关闭任务队列
	p.wg.Wait()        // 等待所有任务完成
}

func main() {
	// 创建一个协程池，有 3 个 worker 和 5 个任务队列大小
	pool := NewPool(3, 5)

	// 启动协程池
	pool.Run()

	// 提交任务
	for i := 1; i <= 10; i++ {
		taskID := i
		pool.Submit(func() {
			fmt.Printf("Task %d is being processed\n", taskID)
			time.Sleep(1 * time.Second) // 模拟任务耗时
		})
	}

	// 关闭协程池并等待所有任务完成
	pool.Shutdown()

	fmt.Println("All tasks are finished.")
}

```



# 设计模式

## 单例模式

### 懒汉式

```go
// 懒汉式
// 需要去解决多线程的问题
package main

import (
    "fmt""sync"
)

var lock = &sync.Mutex{}

type single struct {
}

var singleInstance *single

func getInstance() *single {
    if singleInstance == nil {
        lock.Lock()
        defer lock.Unlock()
        if singleInstance == nil {
            fmt.Println("Creating single instance now.")
            singleInstance = &single{}
        } else {
            fmt.Println("Single instance already created.")
        }
    } else {
        fmt.Println("Single instance already created.")
    }

    return singleInstance
}
```



### 饿汉式

```go
// 饿汉式
// init时instance被初始化，单例的唯一实例被创建
package main

import (
    "fmt""sync"
)

var once sync.Once

type single struct {
}

var singleInstance *single

func getInstance() *single {
    if singleInstance == nil {
        once.Do(
            func() {
                fmt.Println("Creating single instance now.")
                singleInstance = &single{}
            })
    } else {
        fmt.Println("Single instance already created.")
    }

    return singleInstance
}
```



# ACM模式

OJ（牛客网）输入输出练习 Go实现    https://blog.csdn.net/aron_conli/article/details/113462234

OJ在线编程常见输入输出练习场     https://ac.nowcoder.com/acm/contest/5657#question

GoLang之ACM控制台输入输出    https://blog.csdn.net/weixin_52690231/article/details/125436414

https://blog.csdn.net/weixin_44211968/article/details/124632136

https://zhuanlan.zhihu.com/p/551393704

## fmt

### Scan

```
func Scan(a ...interface{}) (n int, err error)
```

> Scan从标准输入扫描文本，将成功读取的空白分隔的值保存进成功传递给本函数的参数。换行视为空白。
> 
> 返回成功扫描的条目个数和遇到的任何错误。如果读取的条目比提供的参数少，会返回一个错误报告原因。

```go
var a1 int
var a2 string
n, err := fmt.Scan(&a1, &a2)
```

参数间以空格或回车键进行分割。
如果输入的参数不够接收的，按回车后仍然会等待其他参数的输入。
如果输入的参数大于接收的参数，只有给定数量的参数被接收，其他参数自动忽略。

### Scanf

较少使用

```go
func Scanf(format string, a ...interface{}) (n int, err error)
```

`Scanf`也可以接收多个参数，但是接收字符串的话，只能在最后接收；
否则按照`"%s%d"`的格式进行接收，无论输入多少字符（包含数字），都会被认定是第一个字符串的内容，不会被第二个参数所接收。

```go
package main

import "fmt"

func main() {
    var a string

    n, err := fmt.Scanf("%s", &a) // afdsfdsfdsfds

    fmt.Println("n = ", n)     // n =  1
    fmt.Println("err = ", err) // err =  <nil>
    fmt.Println("a = ", a)     // a =  afdsfdsfdsfds
}
```

### Scanln

```
func Scanln(a ...interface{}) (n int, err error)
```

> Scanln与Scan类似，但在换行时停止扫描，并且在最后一项之后必须有换行或EOF。

```go
package main

import "fmt"

func main() {
    var a string
    var b string

    n, err := fmt.Scanln(&a, &b)

    fmt.Println("n = ", n)
    fmt.Println("err = ", err)
    fmt.Println("a = ", a)
    fmt.Println("b = ", b)
}
```

与`Scan`的区别：如果设置接收2个参数，`Scan`在输入一个参数后进行回车，会继续等待第二个参数的键入；
而`Scanln`直接认定输入了一个参数就截止了，只会接收一个参数并产生`error（unexpected newline）`，且`n = 1`。

说通俗写，就是`Scanln`认定回车标志着==阻塞接收参数==，而`Scan`认定回车只是一个==分隔符（或空白）==而已。

### scan + for

可以替代bufio，来实现读取若干行（不知道具体几行），但每行的数量要求是固定的

```go
package main

import "fmt"

func main() {
    var a, b int
    for {
        _, err := fmt.Scanln(&a, &b)
        if err != nil {
            break
        }
        fmt.Printf("%d\n\n", a+b)
    }
}
```

## bufio

`bufio`包是对IO的封装，可以操作文件等内容，同样可以用来接收键盘的输入，此时对象不是文件等，而是`os.Stdin`，也就是标准输入设备。

[bufio包文档](https://studygolang.com/pkgdoc)

`bufio`包含了Reader、Writer、Scanner等对象，封装了很多对IO内容的处理方法，但应对键盘输入来说，通过创建Reader对象，并调用其Read*

注意：在使用了bufio之后，不能再使用fmt.Scan()之类的输入，否则会出现读取行数错误的问题。

### NewScanner

必须整行读

容量有限制：https://www.nowcoder.com/feed/main/detail/723b81b2f3d847c3aa6475053da688f5

#### 任意数量求和

```go
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
    // 如果提示index out of range，可能是token大小不够
    // 如果少部分数据无法通过，可能是超int了，换int64
    inputs.Buffer(buf, 1024*1024*1000)        // 设置缓冲区和最大 token 大小
    for inputs.Scan() {                  //每次读入一行
        data := strings.Split(inputs.Text(), " ") //通过空格将他们分割，并存入一个字符串切片
        var sum int
        for _, v := range data {
            val, _ := strconv.Atoi(v) //将字符串转换为int
            sum += val
        }
        fmt.Println("sum = ", sum)
        fmt.Println("data = ", data)       // data 是 []string
        fmt.Println("data[0] = ", data[0])
    }
}
```

#### 任意数量[]int{}

```go
func main() {
    input := bufio.NewScanner(os.Stdin)
    input.Scan()
    data := strings.Split(input.Text(), " ")    // data 是 []string
    fmt.Println("data = ", data)
    nums := []int{}
    for i:=0; i<len(data); i++ {
        v, _ := strconv.Atoi(data[i])    // string转int
        nums = append(nums, v)
    }
    fmt.Println("nums = ", nums)        // nums 是 []int
}
```

#### 指定长宽矩阵

```go
package main

import (
    "fmt"
)

func main() {
    var m, n int
    fmt.Scanln(&m, &n)

    res := make([][]int, m)
    for i := range res {
        res[i] = make([]int, n)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            fmt.Scan(&res[i][j])
        }
    }
    fmt.Println(res)    // 每个切片中以“ ”分隔每个元素
}

/*
输入：
2 3
1 2 3 4 5 6

输出：
[[1 2 3] [4 5 6]]
*/
```

#### 任意矩阵

```go
package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
)

func main() {
    input := bufio.NewScanner(os.Stdin)
    input.Scan() //读取一行内容
    // 因为Atoi，所以要有两个接收符：m, _
    // 不加[0]就是[]int，加了就是int
    m, _ := strconv.Atoi(strings.Split(input.Text(), " ")[0])    // 第一行的第一个字符，转换为int
    n, _ := strconv.Atoi(strings.Split(input.Text(), " ")[1])    // 第一行的第二个字符，转换为int
    res := make([][]int, m)
    for i := range res {
        res[i] = make([]int, n)
    }
    for i := 0; i < m; i++ {
        input.Scan() //读取一行内容
        for j := 0; j < n; j++ {
            res[i][j], _ = strconv.Atoi(strings.Split(input.Text(), " ")[j])
        }
    }
}
```

### Reader

创建Reader对象

```go
reader := bufio.NewReader(os.Stdin)
```

#### ReadByte

```
func (b *Reader) ReadByte() (c byte, err error)
```

用来接收一个`byte`类型，会阻塞等待键盘输入。

```go
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {

    reader := bufio.NewReader(os.Stdin)

    n, err := reader.ReadByte() // a

    fmt.Println("n = ", n)             // n =  97
    fmt.Println("string =", string(n)) // string = a
    fmt.Println("err = ", err)         // err =  <nil>

}
```

#### ReadBytes

```
func (b *Reader) ReadBytes(delim byte) (line []byte, err error)
```

该方法，输入参数为一个byte字符，当输入遇到该字符，会停止接收并返回。
接收的内容包括**该停止字符**以及前面的内容。

```go
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    reader := bufio.NewReader(os.Stdin)
    n, err := reader.ReadBytes('\n')
    fmt.Println("n = ", n)
    fmt.Println("string =", string(n))
    fmt.Println("err = ", err)

}
```

输入的内容为【a】【空格】【b】【回车】
n同时接收了4个byte。包括空格和回车。

![命令行显示](https://img-blog.csdnimg.cn/79866a0b3b7c41e48e3d9d58d20fb0fc.png)

**Tips：**
Reader可以接收包含空格内容的字符串，而不进行分割，这是`bufio.Reader`与`fmt`系的一大不同。

#### ReadString

```
func (b *Reader) ReadString(delim byte) (line string, err error)
```

> ReadString读取直到第一次遇到delim字节，返回一个包含已读取的数据和**delim字节**的字符串。
> 
> 如果ReadString方法在读取到delim之前遇到了错误，它会返回在错误之前读取的数据以及该错误（一般是io.EOF）。当且仅当ReadString方法返回的切片不以delim结尾时，会返回一个非nil的错误。

```go
package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {

    reader := bufio.NewReader(os.Stdin)

    line, err := reader.ReadString('\n')

    fmt.Println("line = ", line)
    fmt.Println("err = ", err)

}
```

该函数可以接收包含空格的字符串。
如果设定回车键为delim byte，则遇到回车后结束接收，同时也会接收回车键。
当以’a’等byte为终止符时，如果没有遇到该符，即使回车也会继续接收。
如果在按下终止符后没有回车，继续键入内容，则只会接收第一次终止符及其之前的内容，之后的内容自动忽略。
