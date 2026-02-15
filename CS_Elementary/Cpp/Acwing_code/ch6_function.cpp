//函数形参 不改变实参
//int f(int &x) 引用

// 数组形参
// 在函数中对数组中的值的修改，会影响函数外面的数组。
// 尽管形式不同，但这三个print函数是等价的
// void print(int *a) {/* … */}
// void print(int a[]) {/* … */}
// void print(int a[10]) {/* … */}
// 多维数组中，除了第一维之外，其余维度的大小必须指定
// void print(int (*a)[10]) {/* … */}
// void print(int a[][10]) {/* … */}

//void类函数，return;  类似break
void swap(int &v1, int &v2)
{
    // 如果两个值相等，则不需要交换，直接退出
    if (v1 == v2)
        return;
    // 如果程序执行到了这里，说明还需要继续完成某些功能

    int tmp = v2;
    v2 = v1;
    v1 = tmp;
    // 此处无须显示的return语句
}

//递归
#include <iostream>
using namespace std;
int fact(int n)
{
    if (n <= 1) return 1;
    return n * fact(n - 1);
}
int main()
{
    int n;
    cin >> n;
    cout << fact(n) << endl;
    return 0;
}