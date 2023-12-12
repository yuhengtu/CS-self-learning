// 给定一个长度为 N 的整数数列，输出每个数左边第一个比它小的数，如果不存在则输出 −1。

// 输入格式 
// 第一行包含整数 N，表示数列长度。
// 第二行包含 N 个整数，表示整数数列。

// 输出格式 
// 共一行，包含 N 个整数，其中第 i 个数表示第 i 个数的左边第一个比它小的数，如果不存在则输出 −1

// 输入样例： 
// 5 
// 3 4 2 7 5 
// 输出样例： -1 3 -1 2 2

#include <iostream>
using namespace std;
const int N = 100010;

int stk[N], tt;

int main() {
  int n;
  cin >> n;
  while (n--) {
    int x;
    scanf("%d", &x);
    
    while (tt && stk[tt] >= x) tt--;
    if (!tt)
      printf("-1 ");
    else
      printf("%d ", stk[tt]);
    stk[++tt] = x;
  }

  return 0;
}
