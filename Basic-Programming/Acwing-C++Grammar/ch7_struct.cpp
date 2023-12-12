    // Person c, persons[100];
    // c.name = "yxc";      // 正确！访问公有变量
    // c.age = 18;          // 错误！访问私有变量
    // c.set_age(18);       // 正确！set_age()是共有成员变量

//结构体和类的作用是一样的。不同点在于类默认是private，结构体默认是public。
//struct Person{}person_a, person_b, persons[100];

//引用和指针类似，相当于给变量起了个别名。
    // int a = 10;
    // int *p = &a;
    // *p += 5;

    //数组名是一种特殊的指针。指针可以做运算
    // int a[5] = {1, 2, 3, 4, 5};
    // for (int i = 0; i < 5; i ++ )
    //     cout << *(a + i) << endl;

    //链表
#include <iostream>
using namespace std;
struct Node
{
    int val;//value
    Node* next;//next指针
    Node(int _val):val(_val),next(NULL){}//构造函数
} ;
int main()
{
    Node* p = new Node(1);//建立一个Node结构体变量，通过构造函数初始化val为1；next为NULL，通过new返回该变量地址，赋给p
    //动态开辟一段空间，地址给p
    Node* q = new Node(2);
    Node* o = new Node(3);

    //调用成员变量
    //Node a = Node(1);  a.next
    //Node *a = new Node(1);  a->next
    p->next=q;
    q->next=o;
    Node* head=p;

//遍历
    for (Node* i = head; i; i = i->next)  //第二个i即i!=0或i!=NULL
        cout << i->val <<endl;

    return 0;
}

// struct Node
// {
//     int val;//value
//     Node* next;
// } *head;
// int main()
// {
//     for (int i = 1; i <= 5; i ++ )
//     {
//         Node* p = new Node();
//         p->val = i;
//         p->next = head;
//         head = p;
//     }
//     for (Node* p = head; p; p = p->next)
//         cout << p->val << ' ';
//     cout << endl;
//     return 0;
// }
