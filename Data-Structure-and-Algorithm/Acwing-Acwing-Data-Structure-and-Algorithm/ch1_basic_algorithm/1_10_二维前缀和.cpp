// 输入一个 n 行 m 列的整数矩阵，再输入 q 个询问，
// 每个询问包含四个整数 x1,y1,x2,y2，表示一个子矩阵的左上角坐标和右下角坐标。
// 对于每个询问输出子矩阵中所有数的和。

// 输入格式
// 第一行包含三个整数 n，m，q。
// 接下来 n 行，每行包含 m 个整数，表示整数矩阵。
// 接下来 q 行，每行包含四个整数 x1,y1,x2,y2，表示一组询问。

// 输出格式
// 共 q 行，每行输出一个询问的结果。

// 输入样例：
// 3 4 3
// 1 7 2 4
// 3 6 2 8
// 2 1 2 3
// 1 1 2 2
// 2 1 3 4
// 1 3 3 4
// 输出样例：
// 17
// 27
// 21

#include <iostream>
using namespace std;
const int N = 100010;

int n, m, q;
int s[N][N];

int main()
{
    scanf("%d%d%d", &n, &m, &q);

    //读入原矩阵
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            scanf("%d", &s[i][j]);

    //计算前缀和
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1];

    //计算子矩阵块 的和
    while (q -- )
    {
        int x1, y1, x2, y2;
        scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
        printf("%d\n", s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1]);
    }

    return 0;
}
