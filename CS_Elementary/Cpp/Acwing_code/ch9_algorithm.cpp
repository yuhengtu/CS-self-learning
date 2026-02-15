//与 或 非～ 异或XOR^ 二进制 按位算 程序员计算器
//cout<< (a^b)<<endl; 
//<<左移  >>右移 移出不要 /2^k

//53252   个位序号为0
// 求x的第k位数字 x >> k & 1
// lowbit(x) = x & -x，返回x的最后一位1  eg：1101000  返回1000 
//计算机用补码存负数  -a和~a+1一样
//reverse翻转 非常重要

#include <algorithm>
// 翻转一个vector：reverse(a.begin(), a.end());
// 翻转一个数组，元素存放在下标0 ~ n-1：reverse(a , a + n );

// unique去重 去掉相邻重复元素
// 返回去重（只去掉相邻的相同元素）之后的尾迭代器（或指针）（去重之后末尾元素的下一个位置）】p
// 把一个vector去重：int m = unique(a.begin(), a.end()) – a.begin();
// 把一个数组去重，元素存放在下标1 ~ n：int m = unique(a + 1, a + n + 1) – (a + 1);

//random_shuffle（头，尾+1）随机数据  
//随机种子默认是0 要srand传入时间作为随机种子

//sort（头，尾+1）从小到大排序；从大到小排序：sort(头，尾+1，greater<int>())
//排结构体 自定义cmp函数代替greater 也可重载小于大于号 不推荐
#include <vector>   // 头文件
#include <iostream>
#include <algorithm>   // 头文件
using namespace std;
struct Rec
{
    int x,y;
}a[5];
bool cmp(Rec a, Rec b)
{
    return a.x < b.x;
}
int main()
{
    for(int i = 0; i < 5; i++)
    {
        a[i].x=-i;
        a[i].y=i;
    }
    for(int i=0;i<5;i++) printf("(%d,%d)",a[i].x,a[i].y);
    cout<<endl;

    sort(a,a+5,cmp);

    for(int i=0;i<5;i++) printf("(%d,%d)",a[i].x,a[i].y);
    cout<<endl;

    return 0;
}

// lower_bound/upper_bound 二分 必须已经从小到大排好序
// lower_bound的第三个参数传x，二分查找，返回指向第一个大于等于x的元素的迭代器（指针）
// int* p=lower_bound(begin，end，比较的值）
// upper_bound同理
// 在有序int数组（元素存放在下标1 ~ n）中查找大于等于x的最小整数的下标：
// int i = lower_bound(a + 1, a + 1 + n, x) - a;  
// 在有序vector<int>中查找小于等于x的最大整数（假设一定存在）：
// int y = *--upper_bound(a.begin(), a.end(), x);
// int t = upper_bound(a.begin(), a.end(), x) - a.begin();