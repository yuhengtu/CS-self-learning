// 给定一个整数 n，请你按照字典序将所有的排列方法输出。

// 输入样例：
// 3

// 输出样例：
// 1 2 3
// 1 3 2
// 2 1 3
// 2 3 1
// 3 1 2
// 3 2 1

#include <iostream>
using namespace std;
const int N = 10;

int n;
int path[N];
bool st[N];//存某一数是否用过

void dfs(int u)
{
    if (u == n)
    {
        for (int i = 0; i < n; i ++ ) printf("%d ", path[i]);
        puts("");
        return;
    }

    for (int i = 1; i <= n; i ++ )
        if (!st[i])//如果数字没用过
        {
            path[u] = i;
            st[i] = true;
            dfs(u + 1);

            //恢复现场
            // path[u] = 0;没必要写，会被覆盖
            st[i] = false;
        }
}

int main()
{
    scanf("%d", &n);
    dfs(0); 
    return 0;
}

// 移位优化
// int n;
// int path[N];

// void dfs(int u, int state)//state存某一数是否用过
// {
//     if (u == n)
//     {
//         for (int i = 0; i < n; i ++ ) printf("%d ", path[i]);
//         puts("");
//         return;
//     }

//     for (int i = 0; i < n; i ++ )
//         if (!(state >> i & 1))//如果没用过
//         {
//             path[u] = i + 1;
//             dfs(u + 1, state + (1 << i));
//         }
// }

// int main()
// {
//     scanf("%d", &n);
//     dfs(0, 0);
//     return 0;
// }
