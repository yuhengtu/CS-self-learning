// 输出带空格的字母串，输入：abc def ghi

#include <iostream>
#include <string.h>
using namespace std;

int main()
{
    char str [1000];
    gets(str);

    int n = strlen(str);

    for (int i = 0; i < n; i++)
    {
        int j = i;
        while (j < n && str[j] != ' ') j++; // 找到下一个空格
        
        for (int k = i;k < j; k++) cout << str[k];
        cout << endl;

        i = j;
    }
    return 0;
}