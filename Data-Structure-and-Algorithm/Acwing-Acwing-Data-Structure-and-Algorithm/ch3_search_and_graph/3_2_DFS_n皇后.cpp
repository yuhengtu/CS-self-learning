// n−皇后问题是指将 n 个皇后放在 n×n 的国际象棋棋盘上，使得皇后不能相互攻击到，即任意两个皇后都不能处于同一行、同一列或同一斜线上。
// 现在给定整数 n，请你输出所有的满足条件的棋子摆法。

// 输入格式
// 共一行，包含整数 n。

// 输出格式
// 每个解决方案占 n 行，每行输出一个长度为 n 的字符串，用来表示完整的棋盘状态。
// 其中 . 表示某一个位置的方格状态为空，Q 表示某一个位置的方格上摆着皇后。
// 每个方案输出完成后，输出一个空行。
// 注意：行末不能有多余空格。
// 输出方案的顺序任意，只要不重复且没有遗漏即可。

// 输入样例：
// 4
// 输出样例：
// 只有两种可能性
// .Q..
// ...Q
// Q...
// ..Q.

// ..Q.
// Q...
// ...Q
// .Q..

// 第一种搜索顺序
// 每一行的皇后放在哪 _1_, _3_, 第一行皇后在第1列，第二行皇后在第3列
// 全排列思路，判断不合法直接剪枝（剪掉这个树枝）并回溯
#include <iostream>
using namespace std;
const int N = 20;
 
int n;
char g[N][N];//记录最终方案
bool col[N], dg[N * 2], udg[N * 2];//列 对角线 反对角线

void dfs(int u)
{
    if (u == n)
    {
        for (int i = 0; i < n; i ++ ) puts(g[i]);//输出g
        puts("");
        return;
    }

    for (int i = 0; i < n; i ++ )
        if (!col[i] && !dg[u + i] && !udg[n - u + i])//如果不冲突
        {
            g[u][i] = 'Q';
            col[i] = dg[u + i] = udg[n - u + i] = true;//设bool
            dfs(u + 1);

            //恢复现场
            col[i] = dg[u + i] = udg[n - u + i] = false;
            g[u][i] = '.';
        }
}

int main()
{
    cin >> n;
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
            g[i][j] = '.';

    dfs(0);
    return 0;
}



// 第二种搜索顺序
// 每个格子 放或不放两个分支
// s表示已经放置的皇后数量
#include <iostream>
using namespace std;
const int N = 10;

int n;
bool row[N], col[N], dg[N * 2], udg[N * 2];
char g[N][N];

void dfs(int x, int y, int s)
{
    if (s > n) return;
    if (y == n) y = 0, x ++ ; //下一行 

    if (x == n)
    {
        if (s == n)
        {
            for (int i = 0; i < n; i ++ ) puts(g[i]);
            puts("");
        }
        return;
    }

    //不放皇后
    g[x][y] = '.';
    dfs(x, y + 1, s);

    //放皇后
    if (!row[x] && !col[y] && !dg[x + y] && !udg[x - y + n])
    {
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = true;
        g[x][y] = 'Q';
        dfs(x, y + 1, s + 1);

        //恢复现场
        g[x][y] = '.';
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = false;
    }
}

int main()
{
    cin >> n;
    dfs(0, 0, 0);
    return 0;
}



