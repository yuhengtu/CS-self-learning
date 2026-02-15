// 给定一个 n 个点 m 条边的有向图，图中可能存在重边和自环。
// 所有边的长度都是 1，点的编号为 1∼n。
// 请你求出 1 号点到 n 号点的最短距离，如果从 1 号点无法走到 n 号点，输出 −1。

// 输入格式
// 第一行包含两个整数 n 和 m。
// 接下来 m 行，每行包含两个整数 a 和 b，表示存在一条从 a 走到 b 的长度为 1 的边。

// 输出格式
// 输出一个整数，表示 1 号点到 n 号点的最短距离。

// 输入样例：
// 4 5
// 1 2
// 2 3
// 3 4
// 1 3
// 1 4

// 输出样例：
// 1

#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;
const int N = 100010;

int n, m;//n个点 m条边
int h[N], e[N], ne[N], idx;//邻接表
int d[N];//距离

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

int bfs()
{
    memset(d, -1, sizeof d);
    queue<int> q;
    d[1] = 0;//开始只有第一个点 
    q.push(1);

    while (q.size())
    {
        int t = q.front();
        q.pop();

        for (int i = h[t ]; i != -1; i = ne[i])
        {
            int j = e[i];//当前点可以到的点  
            if (d[j] == -1)
            {
                d[j] = d[t] + 1;
                q.push(j);
            }
        }
    }
    return d[n];
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
    }

    cout << bfs() << endl;
    return 0;
}
