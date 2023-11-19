//STL容器 已讲string  
//vector 自动改变长度的数组 倍增
#include <vector>   // 头文件
#include <iostream>
#include <queue>//头文件queue主要包括循环队列queue和优先队列priority_queue两个容器。
#include <stack>
#include <deque>
#include <set> //set和multiset两个容器，分别是“有序集合”和“有序多重集合”,不可重复和可重复
#include <unordered_set>//无序，没有lower_bound/upper_bound，所有都是o(1)
#include <map>//写法特殊
#include <unordered_map>//效率高 cpp11
#include <bitset>

using namespace std;
int main()
{
    vector<int> a({1,2,3});      //C++11 相当于一个长度动态变化的int数组 结尾进出o(1) 开头进出o(n) 自带比较运算
    vector<int> b[233]; // 相当于第一维长233，第二位长度动态变化的int数组
    //struct rec{…};
    //vector<rec> c;      // 自定义的结构体类型也可以保存在vector中
    a.size();
    a.clear();//除了队列 优先队列 栈之外都有
    a.empty();
    a.front();//第一个元素
    a.back();//最后一个元素
    a.push_back(4);//在最后加上一个元素为4
    a.pop_back();//删除最后一个严肃
    //迭代器  即指针  一般不用
    vector<int>::iterator it=a.begin();//*it->a[0]
    //a.end()最后一个的后一个（左闭右开)
    //it+2->a[2]  

    //遍历输出
    for (int i = 0; i < a.size(); i ++)
        cout << a[i] << endl;
    for (vector<int>::iterator it = a.begin(); it != a.end(); it ++)
        cout << *it << endl;
    for (auto it = a.begin(); it != a.end(); it ++)
        cout << *it << endl;
    for (int x:a)  cout<<x<<"";
    cout<<endl;

    queue<int> q;//循环队列 先进先出 只能队尾插入队头弹出
    //struct rec{…}; queue<rec> q;    //大根堆结构体rec中必须重载小于号
    q.push(1);    // 从队尾插入
    q.pop();    // 从队头弹出
    q.front();   // 返回队头元素
    q.back();    // 返回队尾元素
    q = queue<int>();// 清空

    priority_queue<int> k;                              // 大根堆  优先队列  //结构体中必须重载小于号
    k.push(1);    // 把元素插入堆，不一定队尾
    k.pop();   // 删除堆顶元素
    k.top();    // 查询堆顶元素（最大值）
    priority_queue<int, vector<int>, greater<int> > qqq;   // 小根堆   //结构体中必须重载大于号
    priority_queue<pair<int, int> >qq;//pair  二元组n

//栈 先进后出  只能栈顶出入
    stack<int> stk;
    stk.push(1);
    stk.top();
    stk.pop(); 

//双端队列 头尾都可进可出  都是o(1) 运行效率慢
    deque<int> aa;
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
//s.insert(x)把一个元素x插入到集合s中，时间复杂度为 O(logn)
//在set中，若元素已存在，则不会重复插入该元素，对集合的状态无影响。
//s.find(x)在集合s中查找等于x的元素，并返回指向该元素的迭代器。若不存在，则返回s.end()。时间复杂度为 O(logn)
//lower_bound/upper_bound
//这两个函数的用法与find类似，但查找的条件略有不同，时间复杂度为 O(logn)
//s.lower_bound(x)查找 大于等于x的元素中最小的一个，并返回指向该元素的迭代器。
//s.upper_bound(x)查找 大于x的元素中最小的一个，并返回指向该元素的迭代器。
//设it是一个迭代器，s.erase(it)从s中删除迭代器it指向的元素，时间复杂度为 O(logn)
//设x是一个元素，s.erase(x)从s中删除所有等于x的元素，时间复杂度为 O(k+logn)，其中 k是被删除的元素个数。
//s.count(x)返回集合s中等于x的元素个数，时间复杂度为 O(k+logn)，其中 k为元素x的个数。

// map<key_type, value_type> name;
// map<long long, bool> vis;//
// map<string, int> hash;
// map<pair<int, int>, vector<int>> test;
map<string, int> aaa;//类似数组 字典
aaa["tyh"]=1;
cout<<aaa["tyh"]<<endl;

map<string, vector<int> > bbb;//类似数组 字典
bbb["tyh"]=vector<int>({1,2,3,4});
cout<<bbb["tyh"][2]<<endl;
//size/empty/clear/begin/end
//insert/erase  
aaa.insert({"a",{}});//二元组 

unordered_set<int> sss;
unordered_map<string, int> aaaa;

bitset<1000> aaaa;//长度为1k的01串 
aaaa.count();//返回1的个数
aaaa.set(3);//设置第三位为1
aaaa.reset(3);//设置第三位为0 

pair <int,string> p;//可以比较运算 first不同比second
p={3,"tyh"};
cout<<p.first<<" "<<p.second<<endl;
p = make_pair(4,"abc");//cpp99

   return 0;
}
