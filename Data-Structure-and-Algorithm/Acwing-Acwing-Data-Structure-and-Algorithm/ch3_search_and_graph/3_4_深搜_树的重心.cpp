// 给定一颗树，树中包含 n 个结点（编号 1∼n）和 n−1 条无向边。
// 请你找到树的重心，并输出将重心删除后，剩余各个连通块中点数的最大值。
// 重心定义：重心是指树中的一个结点，如果将这个点删除后，剩余各个连通块中点数的最大值最小，那么这个节点被称为树的重心。

// 输入格式 
// 第一行包含整数 n，表示树的结点数。
// 接下来 n−1 行，每行包含两个整数 a 和 b，表示点 a 和点 b 之间存在一条边。

// 输出格式 
// 输出一个整数 m，表示将重心删除后，剩余各个连通块中点数的最大值。
// 重心可能不唯一，但最小的最大值唯一

// 输入样例 
// 9 
// 1 2 
// 1 7 
// 1 4 
// 2 8 
// 2 5 
// 4 3 
// 3 9 
// 4 6 

// 输出样例： 
// 4

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int N = 100010, M = N * 2;

int n;
int h[N], e[M], ne[M], idx;//h存链表头，e存值
int ans = N;//全局答案
bool st[N];

void add(int a, int b) {
  e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

// 基本模板
// int dfs(int u) {
//   st[u] = true;  //标记已搜过

//   for (int i = h[u]; i != -1; i = ne[i]) {
//     int j = e[i];
//     if (!st[j]) dfs(j);
//   }
// }

    int dfs(int u) {
      st[u] = true;  //标记已搜过

      int size = 0, sum = 0;//size存当前子树大小；sum存删除该点后，剩下连通块的点数最大值

      for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (st[j]) continue;

        int s = dfs(j);//表示子树大小
        size = max(size, s);
        sum += s;//当前子树是以u为根节点的一部分
      }

      size = max(size, n - sum - 1);//该点头上的连通块
      ans = min(ans, size);

      return sum + 1;
    }

    int main() {
      scanf("%d", &n);

      memset(h, -1, sizeof h);

      for (int i = 0; i < n - 1; i++) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);//无向边
      }

      dfs(1);
      // dfs(n);也可以，从哪个点开始都一样

      printf("%d\n", ans);

      return 0;
    }
