// 输入一个 n 行 m 列的整数矩阵，再输入 q 个操作，每个操作包含五个整数 x1,y1,x2,y2,c
// 其中 (x1,y1) 和 (x2,y2) 表示一个子矩阵的左上角坐标和右下角坐标。
// 每个操作都要将选中的子矩阵中的每个元素的值加上 c。

// 输入格式
// 第一行包含整数 n,m,q。
// 接下来 n 行，每行包含 m 个整数，表示整数矩阵。
// 接下来 q 行，每行包含 5 个整数 x1,y1,x2,y2,c，表示一个操作。

// 输出格式
// 共 n 行，每行 m 个整数，表示所有操作进行完毕后的最终矩阵。

// 输入样例：
// 3 4 3
// 1 2 2 1
// 3 2 2 1
// 1 1 1 1
// 1 1 2 2 1
// 1 3 2 3 2
// 3 1 3 4 1
// 输出样例：
// 2 3 4 1
// 4 3 4 1
// 2 2 2 2

#include <iostream>
using namespace std;
const int N = 100010;

int n, m, q;
int a[N][N], b[N][N];

void insert(int x1, int y1, int x2, int y2, int c)
{
    b[x1][y1] += c;
    b[x2 + 1][y1] -= c;
    b[x1][y2 + 1] -= c;
    b[x2 + 1][y2 + 1] += c;
}

int main()
{
    scanf("%d%d%d", &n, &m, &q);
    // 读入 a
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            scanf("%d", &a[i][j]);
    // 初始化 b
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            insert(i, j, i, j, a[i][j]);

    // +c操作
    while (q -- )
    {
        int x1, y1, x2, y2, c;
        cin >> x1 >> y1 >> x2 >> y2 >> c;
        insert(x1, y1, x2, y2, c);
    }

    // A -> B
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            b[i][j] += b[i - 1][j] + b[i][j - 1] - b[i - 1][j - 1];

    for (int i = 1; i <= n; i ++ )
    {
        for (int j = 1; j <= m; j ++ ) printf("%d ", b[i][j]);
        puts(""); //即endl
    }

    return 0;
}
