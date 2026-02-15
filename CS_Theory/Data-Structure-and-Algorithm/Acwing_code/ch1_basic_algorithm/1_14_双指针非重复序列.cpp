// 给定一个长度为 n 的整数序列，请找出最长的不包含重复的数的连续区间，输出它的长度。

// 输入格式
// 第一行包含整数 n。
// 第二行包含 n 个整数（均在 0∼105 范围内），表示整数序列。

// 输出格式
// 共一行，包含一个整数，表示最长的不包含重复的数的连续区间的长度。

// 输入样例：
// 5
// 1 2 2 3 5

// 输出样例：
// 3 （ 2 3 5 ）

#include <iostream>
using namespace std;
const int N = 100010;

int n;
int a[N], s[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);
    // 读入到数组a，s[n]存放a中数字n出现的个数

    int res = 0;
    // i在右，j在左
    for (int i = 0, j = 0; i < n; i ++ )
    {
        s[a[i]] ++ ;
        // while (j < i && s[a[i]] > 1) s[a[j ++ ]] -- ;
        while (s[a[i]] > 1) 
        {
            s[a[j]] -- ;
            j ++ ; //左边界j走到没有重复数字为止
        }
        res = max(res, i - j + 1);
    }
    cout << res << endl;
    return 0;
}
