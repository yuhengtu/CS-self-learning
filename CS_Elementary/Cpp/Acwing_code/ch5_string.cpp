0//常用ASCII值：'A'- 'Z'是65 ~ 90，'a' - 'z'是97 - 122，0 - 9是 48 - 57。
//字符串就是字符数组加上结束符'\0'
#include <iostream>
#include <string.h>//.h 字符数组
// (1) strlen(str)，求字符串的长度
// (2) strcmp(a, b)，比较两个字符串的大小，a < b返回-1，a == b返回0，a > b返回1。这里的比较方式是字典序！
// (3) strcpy(a, b)，将字符串b复制给从a开始的字符数组。
//      char a[100] = "hello world!";
//     // 注意：下述for循环每次均会执行strlen(a)，运行效率较低，最好将strlen(a)用一个变量存下来
//     for (int i = 0; i < strlen(a); i ++ )
//         cout << a[i] << endl;
#include <string>//可变长的字符序列
    // string s1;              // 默认初始化，s1是一个空字符串
    // string s2 = s1;         // s2是s1的副本，注意s2只是与s1的值相同，并不指向同一段地址
    // string s3 = "hiya";     // s3是该字符串字面值的副本
    // string s4(10, 'c');     // s4的内容是 "cccccccccc"
    // cin >> s1 >> s2;
    // cout << s1 << s2 << endl;
    // printf(“%s”, s.c_str());
//使用getline读取一整行
// string s;
// getline(cin, s);
// cout << s << endl;
// string s1(10, 'c'), s2;     // s1的内容是 cccccccccc；s2是一个空字符串
// s1 = s2;                    // 赋值：用s2的副本替换s1的副本
//                             // 此时s1和s2都是空字符串
using namespace std;
int main()
{
    char a1[] = {'C', '+', '+'};            // 列表初始化，没有空字符
    char a2[] = {'C', '+', '+', '\0'};      // 列表初始化，含有显示的空字符
    char a3[] = "C++";                      // 自动添加表示字符串结尾的空字符
    //char a4[6] = "Daniel";                  // 错误：没有空间可以存放空字符

    // char str[100];
    // cin >> str;             // 输入字符串时，遇到空格或者回车就会停止
    // cout << str << endl;    // 输出字符串时，遇到空格或者回车不会停止，遇到'\0'时停止
    // printf("%s\n", str);

//读入一行字符串，包括空格：
    char str1[100];
    fgets(str1, 100, stdin);  // gets函数在新版C++中被移除了，因为不安全。˜
                             // 可以用fgets代替，但注意fgets不会删除行末的回车字符
    cout << str1 << endl;

    string s1, s2 = "abc";
    cout << s1.empty() << endl;
    cout << s2.empty() << endl;
    cout << s2.size() << endl;
    s1 = "hello";
    s2 = "world";   
    string s3 = s1 + ", " + s2 + '\n';
    cout << s3 << endl;
    string s6 = s1 + ", " + "world";  // 正确，每个加法运算都有一个运算符是string
    //string s7 = "hello" + ", " + s2;  // 错误：不能把字面值直接相加，运算是从左到右进行的

    string s = "hello world";
    for (char c: s) cout << c << endl;//用char c遍历s
    for (char& c: s) c = 'a';//别名 两者完全相同 引用
    cout << s << endl;

    return 0;
}
