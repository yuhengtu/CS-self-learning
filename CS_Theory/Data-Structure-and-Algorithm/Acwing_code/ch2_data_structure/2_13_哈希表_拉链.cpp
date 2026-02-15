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
const int N = 100003; // 100000之后的第一个质数

int h[N], e[N], ne[N], idx;

void insert(int x)
{
    int k = (x % N + N) % N; //C++中，负数 mod N余数是负数，+N再mod N，余数必定是正数
    //同单链表 e[idx] = x; ne[idx] = head; head = idx; idx++;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx;// 插入到第一个位置
    idx ++;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
        if (e[i] == x)
            return true;

    return false;
}

int main()
{
    int n;
    scanf("%d", &n);

    memset(h, -1, sizeof h);// 先把所有槽清空，-1表示空指针，在cstring库里

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);

        if (*op == 'I') insert(x);
        else
        {
            if (find(x)) puts("Yes");
            else puts("No");
        }
    }

    return 0;
}
