// 假定有一个无限长的数轴，数轴上每个坐标上的数都是 0。
// 现在，我们首先进行 n 次操作，每次操作将某一位置 x 上的数加 c。
// 接下来，进行 m 次询问，每个询问包含两个整数 l 和 r，你需要求出在区间 [l,r] 之间的所有数的和。

// 输入格式
// 第一行包含两个整数 n 和 m。
// 接下来 n 行，每行包含两个整数 x 和 c。
// 再接下来 m 行，每行包含两个整数 l 和 r。

// 输出格式
// 共 m 行，每行输出一个询问中所求的区间内数字和。

// 输入样例：
// 3 3
// 1 2
// 3 6
// 7 5
// 1 3
// 4 6
// 7 8
// 输出样例：
// 8
// 0
// 5

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef pair<int, int> PII;

const int N = 300010;// 插入用到1个坐标，查询用到两个坐标，每个范围10^5 

int n, m;
int a[N], s[N]; 

vector<int> alls; // 待离散化的数组
vector<PII> add, query; // 用pair存取包含两个数的操作，分别是插入和查询

int find(int x)//在alls中查找
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}

int main()
{
    //读入插入操作
    cin >> n >> m;
    for (int i = 0; i < n; i ++ )
    {
        int x, c;
        cin >> x >> c;
        add.push_back({x, c});

        alls.push_back(x);
    }

    // 读入查询操作
    for (int i = 0; i < m; i ++ )
    {
        int l, r;
        cin >> l >> r;
        query.push_back({l, r});

        alls.push_back(l);
        alls.push_back(r);
    }

    // add包含（x，c）
    // query包含 （l，r）
    // 去重排序 alls，alls包含 x，l，r
    sort(alls.begin(), alls.end());
    alls.erase(unique(alls.begin(), alls.end()), alls.end());

    // 处理插入
    for (auto item : add)
    {
        int x = find(item.first);// find函数是在alls中查找 x 离散化后映射到的自然数
        a[x] += item.second; //值存到a数组
    }

    // 计算前缀和
    for (int i = 1; i <= alls.size(); i ++ ) s[i] = s[i - 1] + a[i];

    // 处理询问
    for (auto item : query)
    {
        int l = find(item.first), r = find(item.second);//左右区间离散化后的值
        cout << s[r] - s[l - 1] << endl;
    }

    return 0;
}
