// 维护一个字符串集合，支持两种操作：
// I x 向集合中插入一个字符串 x；
// Q x 询问一个字符串在集合中出现了多少次。
// 共有 N 个操作，所有输入的字符串总长度不超过 105，字符串仅包含小写英文字母。

// 输入格式
// 第一行包含整数 N，表示操作数。
// 接下来 N 行，每行包含一个操作指令，指令为 I x 或 Q x 中的一种。

// 输出格式
// 对于每个询问指令 Q x，都要输出一个整数作为结果，表示 x 在集合中出现的次数。
// 每个结果占一行。

// 输入样例：
// 5
// I abc
// Q abc
// Q ab
// I ab
// Q ab

// 输出样例：
// 1
// 0
// 1

#include <iostream>
using namespace std;
const int N = 100010;

int son[N][26], cnt[N], idx;
//最多26个分支，cnt存当前点结尾单词个数，idx用于读入
//下标是0的点，既是根结点，又是空节点（没有子节点也会指向0）
char str[N];

void insert(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i++)  //str[i]是结尾\0，判断是否走到结尾
    {
      int u = str[i] - 'a'; //a-z -> 0-25
      if (!son[p][u]) son[p][u] = ++idx; //如果p结点不存在u这个儿子，创建之
      // ++idx 给每个结点赋一个唯一的值
      p = son[p][u]; //走到子节点上，子节点的值即idx
    }
    cnt[p] ++ ;
}

int query(char *str) //返回字符串出现多少次
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];  //走到子节点上
    }
    return cnt[p];
}

int main()
{
    int n;
    scanf("%d", &n);
    while (n -- )
    {
        char op[2];
        scanf("%s%s", op, str);
        if (*op == 'I') insert(str);
        else printf("%d\n", query(str));
    }

    return 0;
}
