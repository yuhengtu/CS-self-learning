// 维护一个集合，支持如下几种操作：
// I x，插入一个整数 x；
// Q x，询问整数 x 是否在集合中出现过；
// 现在要进行 N 次操作，对于每个询问操作输出对应的结果。
// 数据空间：−109≤x≤109，操作空间：1≤N≤105

// 输入格式
// 第一行包含整数 N，表示操作数量。
// 接下来 N 行，每行包含一个操作指令，操作指令为 I x，Q x 中的一种。

// 输出格式
// 对于每个询问指令 Q x，输出一个询问结果，如果 x 在集合中出现过，则输出 Yes，否则输出 No。

// 输入样例：
// 5
// I 1
// I 2
// I 3
// Q 2
// Q 5

// 输出样例：
// Yes
// No

#include <cstring>
#include <iostream>
using namespace std;
const int N = 200003, null = 0x3f3f3f3f; 
//搜索出200000后的第一个质数为200003；0x3f3f3f3f是一个大于10^9的数，作为null

int h[N];

int find(int x) //如果x在哈希表中，返回k的下标；如果x不在哈希表中，返回他应该存储的位置
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x) //坑位上有人 且 !=x 
    {
        t ++ ;
        if (t == N) t = 0;//到达哈希表末尾，返回开头
    }
    return t;
}

int main()
{
    memset(h, 0x3f, sizeof h);//memset按字节，int有四个字节，因此得到0x 3f3f3f3f
    // 常用：memset(h, 0, sizeof h); 每一位都是0
        // memset(h, -1, sizeof h); -1的二进制是全1，每一位都是1
    int n;
    scanf("%d", &n);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I') h[find(x)] = x;
        else
        {
            if (h[find(x)] == null) puts("No");
            else puts("Yes");
        }
    }

    return 0;
}
