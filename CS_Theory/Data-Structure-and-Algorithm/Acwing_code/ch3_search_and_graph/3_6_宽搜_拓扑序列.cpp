// 给定一个 n 个点 m 条边的有向图，点的编号是 1 到 n，图中可能存在重边和自环。
// 请输出任意一个该有向图的拓扑序列，如果拓扑序列不存在，则输出 −1。
// 若一个由图中所有点构成的序列 A 满足：对于图中的每条边 (x,y)，x 在 A 中都出现在 y 之前，则称 A 是该图的一个拓扑序列。
 
// 输入格式
// 第一行包含两个整数 n 和 m。
// 接下来 m 行，每行包含两个整数 x 和 y，表示存在一条从点 x 到点 y 的有向边 (x,y)。

// 输出格式
// 共一行，如果存在拓扑序列，则输出任意一个合法的拓扑序列即可。否则输出 −1。

// 输入样例：
// 3 3
// 1 2
// 2 3
// 1 3

// 输出样例：
// 1 2 3

#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 100010;

int n, m;
int h[N], e[N], ne[N], idx;
int d[N];//入度 
int q[N];

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

bool topsort()
{
    int hh = 0, tt = -1;
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];//找到出边
            d[j] --;
            if (d[j] == 0)
                q[ ++ tt] = j;
        }
    }
    return tt == n - 1;
    //True说明队列里一共进了n个点，说明所有点都进来过，是有向无环图；False有环
    //队列里的内容就是拓扑序
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);

    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);

        d[b] ++ ;
    }

    if (!topsort()) puts("-1");
    else
    {
        for (int i = 0; i < n; i ++ ) printf("%d ", q[i]);
        puts("");
    }
    return 0;
}