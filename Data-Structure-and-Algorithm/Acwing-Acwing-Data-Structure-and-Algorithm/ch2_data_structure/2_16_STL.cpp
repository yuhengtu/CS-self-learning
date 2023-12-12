// C++ reference
//STL容器 已讲string
#include <bitset>
#include <deque>
#include <iostream>
#include <map>    //写法特殊
#include <queue>  //头文件queue主要包括循环队列queue和优先队列priority_queue两个容器。
#include <set>  //set和multiset两个容器，分别是“有序集合”和“有序多重集合”,不可重复和可重复
#include <stack>
#include <string>         //可变长的字符序列
#include <unordered_map>  //效率高 cpp11
#include <unordered_set>  //无序，没有lower_bound/upper_bound，所有都是o(1)
#include <vector>

using namespace std;
int main() {
  //vector 自动改变长度的数组 倍增

  // vector<int> a;
  // vector<int> a(10); //长度为10
  // vector<int> a(10, -3); //长度为10,10个-3
  // vector<int> a[10];  //10个vector组成数组

  vector<int> a(
      {1,
       2,
       3});  //C++11 相当于一个长度动态变化的int数组 结尾进出o(1) 开头进出o(n)
  vector<int> b[233];  // 相当于第一维长233，第二位长度动态变化的int数组
  //struct rec{…};
  //vector<rec> c;      // 自定义的结构体类型也可以保存在vector中

  a.size();   //元素个数 O(1),有变量存，所有STL都有
  a.empty();  //是否为空 O(1)，所有STL都有
  a.clear();  //除了队列 优先队列 栈之外都有

  a.front();       //第一个元素
  a.back();        //最后一个元素
  a.push_back(4);  //在最后加上一个元素为4
  a.pop_back();    //删除最后一个数

  //迭代器  即指针  一般不用
  vector<int>::iterator it = a.begin();  //*it->a[0]
  //a.end()最后一个的后一个（左闭右开)
  //it+2->a[2]

  //三种遍历输出
  for (int i = 0; i < a.size(); i++) cout << a[i] << endl;
  for (vector<int>::iterator it = a.begin(); it != a.end(); it++)
    cout << *it << endl;
  for (auto it = a.begin(); it != a.end(); it++) cout << *it << endl;
  //C++11 范围遍历
  for (int x : a) cout << x << "";
  cout << endl;

  // 自带比较运算
  vector<int> a(4, 3), b(3, 4);  //四个3 < 三个4，按字典序比较
  if (a < b) puts("a<b");

  pair<int, string> p;  //可以比较运算 first不同比second 字典序
  // p.first;//第一个元素
  // p.second;//第二个元素
  //需要排序的放第一个，不需要排序的放第二个，整个pair排序；可以存>2个东西
  p = {3, "tyh"};
  cout << p.first << " " << p.second << endl;
  p = make_pair(4, "abc");  //cpp99

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

  //size()/length() 长度
  cout << s3.substr(1, 2) << endl;  //返回下标1开始长度2的子串
  cout << s3.substr(1, 2) << endl;  //返回下标1开始的所有
  printf("%s\n", s3.c_str());  //返回存储a的起始地址,打印整个字符串

  string s = "hello world";
  for (char c : s) cout << c << endl;  //用char c遍历s
  for (char& c : s) c = 'a';           //别名 两者完全相同 引用
  cout << s << endl;

  queue<int> q;  //循环队列 先进先出 只能队尾插入队头弹出
  //struct rec{…}; queue<rec> q;    //大根堆结构体rec中必须重载小于号
  q.push(1);         // 从队尾插入
  q.pop();           // 从队头弹出
  q.front();         // 返回队头元素
  q.back();          // 返回队尾元素
  q = queue<int>();  // 没有clear,通过重新构造清空

  priority_queue<int> k;  // 大根堆  优先队列  //结构体中必须重载小于号
  k.push(1);              // 把元素插入堆
  k.pop();                // 删除堆顶元素
  k.top();                // 查询堆顶元素（最大值）
  //小根堆
  // 法1:插入负数 k.push(-x);
  // 法2:
  priority_queue<int, vector<int>, greater<int> >
      qqq;  // 小根堆   //结构体中必须重载大于号
  priority_queue<pair<int, int> > qq;  //pair  二元组n

  //栈 先进后出  只能栈顶出入
  stack<int> stk;
  stk.push(1);
  stk.top();
  stk.pop();

  //双端队列 头尾都可进可出  都是o(1) 运行效率慢,比数组慢几倍,用的不多
  deque<int> aa;  //加强版vector
  // aa[0]              // 随机访问
  // begin/end       // 返回deque的头/尾迭代器
  // front/back      // 队头/队尾元素
  // push_back       // 从队尾入队
  // push_front      // 从队头入队
  // pop_back        // 从队尾出队
  // pop_front       // 从队头出队
  // clear           // 清空队列

  //set
  set<int> s;
  //struct rec{…}; set<rec> s;  // 结构体rec中必须定义小于号
  multiset<double> ss;

  //size/empty/clear
  //set<int>::iterator it； 可以++ --  s.begin()  s.end()
  //s.insert(x)插入，O(logn)
  //s.find(x)查找等于x的元素,并返回迭代器。若不存在返回s.end() O(logn)
  //s.count(x)返回等于x的元素个数，时间复杂度为 O(k+logn)，其中 k为元素x的个数。

  //lower_bound/upper_bound
  //这两个函数的用法与find类似，但查找的条件略有不同，时间复杂度为 O(logn)
  //s.lower_bound(x)查找 大于等于x的元素中最小的一个，并返回迭代器。
  //s.upper_bound(x)查找 大于x的元素中最小的一个，并返回迭代器。

  //设it是一个迭代器，s.erase(it)删除迭代器it指向的元素，时间复杂度为 O(logn)
  //设x是一个元素，s.erase(x)删除所有等于x的元素，时间复杂度为 O(k+logn)，其中 k是被删除的元素个数。

  // map 两个东西映射
  // map<key_type, value_type> name;
  // map<long long, bool> vis;//
  // map<string, int> hash;
  // map<pair<int, int>, vector<int>> test;
  map<string, int> aaa;  //类似数组 字典
  aaa["tyh"] = 1;
  cout << aaa["tyh"] << endl;  //O(logn)

  map<string, vector<int> > bbb;  //类似数组 字典
  bbb["tyh"] = vector<int>({1, 2, 3, 4});
  cout << bbb["tyh"][2] << endl;
  //size/empty/clear 除了这三个基本都是O(logn)
  //begin/end ++ -- O(logn)
  //insert,输入pair/ erase,输入x或迭代器/ find
  //lower_bound/upper_bound
  aaa.insert({"a", {}});  //二元组

  unordered_set<int> sss;
  unordered_map<string, int> aaaa;
  // unordered_multiset,unordered_multimap
  // 这四个与上面类似,但curd大多是O(1)
  // 无序,不支持lower_bound/upper_bound,也不支持迭代器++--

  // bool每个存1字节,1024个要1KB;bitset压到128bit,节省8倍内存
  bitset<1000> aaaa;  //长度为1k的01串
  // ~aaaa,&,|,^,<<,>>,==,!=,[]取出某一位
  aaaa.count();  //返回1的个数
  //any判断是否至少有一个1/none判断是否全为0
  //set()所有位置1,set(k,v)第k位置1
  aaaa.set(3);  //设置第三位为1
  // reset,reset(k,v)
  aaaa.reset(3);  //设置第三位为0
  // flip()全部取反,~;flip(k)第k位取反

  return 0;
}
