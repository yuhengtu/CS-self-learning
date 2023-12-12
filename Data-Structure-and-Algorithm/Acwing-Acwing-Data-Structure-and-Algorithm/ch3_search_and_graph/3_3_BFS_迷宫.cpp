// 给定一个 n×m 的二维整数数组，用来表示一个迷宫
// 数组中只包含 0 或 1，其中 0 表示可以走的路，1 表示不可通过的墙壁。
// 最初，有一个人位于左上角 (1,1) 处，已知该人每次可以向上、下、左、右任意一个方向移动一个位置。
// 请问，该人从左上角移动至右下角 (n,m) 处，至少需要移动多少次。
// 数据保证 (1,1) 处和 (n,m) 处的数字为 0，且一定至少存在一条通路。

// 输入格式
// 第一行包含两个整数 n 和 m。
// 接下来 n 行，每行包含 m 个整数（0 或 1），表示完整的二维数组迷宫。

// 输出格式
// 输出一个整数，表示从左上角移动至右下角的最少移动次数。

// 输入样例：
// 5 5
// 0 1 0 0 0
// 0 1 0 1 0
// 0 0 0 0 0
// 0 1 1 1 0
// 0 0 0 1 0

// 输出样例：
// 8

#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;
typedef pair<int, int> PII;

const int N = 110;

int n, m;
int g[N][N], d[N][N];//g存地图，d存每个点到起点的距离

// 法1:手写一个队列
PII q[N*N], Prev[N][N];//Prev记录路径

int bfs()
{
    int hh = 0, tt = 0;
    q[0] = {0,0};

    memset(d, -1, sizeof d);//初始-1表示没走过
    d[0][0] = 0;//0表示走过

    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};//表示上下左右四个（x，y）向量

    while (hh <= tt)
    {
        auto t = q[hh ++];//读取并弹出

        for (int i = 0; i < 4; i ++ )
        {
            int x = t.first + dx[i], y = t.second + dy[i];

            if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 && d[x][y] == -1)
            //在边界内且点没有走过,只有第一次搜到才算
            {
                d[x][y] = d[t.first][t.second] + 1;
                Prev[x][y] = t;
                q[++ tt] = {x,y};//搜索成功并加入
            }
        }
    }

    int x = n-1, y = m-1;
    while(x||y)
    {
        cout << x << " " << y << endl;
        auto t = Prev[x][y];
        x = t.first, y = t.second;
    }

    return d[n - 1][m - 1];//右下角点的距离
}

// // 法2:STL
// int bfs()
// {
//     queue<PII> q;

//     memset(d, -1, sizeof d);
//     d[0][0] = 0;
//     q.push({0, 0});

//     int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};

//     while (q.size())
//     {
//         auto t = q.front();
//         q.pop();

//         for (int i = 0; i < 4; i ++ )
//         {
//             int x = t.first + dx[i], y = t.second + dy[i];

//             if (x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 && d[x][y] == -1)
//             {
//                 d[x][y] = d[t.first][t.second] + 1;
//                 q.push({x, y});
//             }
//         }
//     }

//     return d[n - 1][m - 1];
// }

int main()
{
    cin >> n >> m;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            cin >> g[i][j];//读入地图

    cout << bfs() << endl;
    return 0;
}

