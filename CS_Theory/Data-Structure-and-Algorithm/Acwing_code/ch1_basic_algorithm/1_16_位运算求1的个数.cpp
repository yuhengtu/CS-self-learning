// 给定一个长度为 n 的数列，请你求出数列中每个数的二进制表示中 1 的个数。

// 输入样例：
// 5
// 1 2 3 4 5
// 输出样例：
// 1 1 2 1 2

#include <iostream>
using namespace std;

int lowbit (int x)
{
    return x & -x;
}

int main()
{
    int n;
    scanf("%d", &n);
    
    while (n -- )
    {
        int x = 0;
        scanf("%d", &x);

        int res = 0;
        while(x) x -= lowbit(x), res ++ ;//每次减去x的最后一位1

        printf("%d ", res);
    }

    return 0;
}