// class default to private，struct default to public
#include <iostream>
#include <cstdio>
#include <string>
#include <memory>

using namespace std;

class Student 
{
    private:
        string name_;
        int age_;
    
    public:
        // constructor
        Student(string n = "Unknown", int a = 0) : name_(n), age_(a) {}
        // Student(string n = "Unknown", int a = 0) {
        //     name = n;
        //     age = a;
        // } // equivalent
    
        void print() {
            cout << name_ << " " << age_ << endl;
        }
} s1("Alice", 18), s2, students[3];

struct Node
{
    int val;
    Node* next;

    ~Node() {delete next;} // destructor
} *head = nullptr;
// Node* head = nullptr;
// head is a pointer to Node object, and it is initialized to nullptr

struct Node_smart {
    int val;
    unique_ptr<Node_smart> next;
};

int main() {
    // 1. class
    s1.print();
    s2.print();

    students[0] = Student("Bob", 19);
    students[1] = Student("Carl", 20);
    
    students[0].print();
    students[1].print();
    students[2].print();
    printf("\n");
    
    // 2. struct, with classic pointer
    for (int i = 1; i <= 5; i ++)
    {
        Node* p = new Node(); 
        // new enable the Node object to survive beyond each loop iteration
        // Node* p: p point to the new Node object
        p->val = i; // same as (*p).val = i;
        p->next = head;
        head = p;
    }
    
    // 5 → 4 → 3 → 2 → 1 → nullptr
    // ↑
    // head

    for (Node* p = head; p; p = p->next)
        cout << p->val << ' ';
    cout << endl;
    
    // delete
    delete head; // call destructor
    // same to the following code if we do not write the destructor ~Node()
    // Node* p = head;
    // while (p) {
    //     Node* next = p->next;
    //     delete p;
    //     p = next;
    // }
    // head = nullptr;
    printf("\n");
    
    // 3. smart pointer
    unique_ptr<Node_smart> head_smart = nullptr;

    for (int i = 1; i <= 5; i++) {
        auto p = make_unique<Node_smart>();
        p->val = i;
        p->next = move(head_smart);
        head_smart = move(p);
    }

    for (Node_smart* p = head_smart.get(); p; p = p->next.get())
        cout << p->val << ' ';
    cout << endl;

    return 0;
}