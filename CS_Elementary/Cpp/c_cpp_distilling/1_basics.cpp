#include <iostream>
#include <cstdio> // for printf
#include <string.h> // for C-char-array strlen/strcmp/strcpy
#include <string> // for Cpp-char-array std::string
#include <array> // for Cpp-number-array std::array
#include <vector> // for Cpp-number-array std::vector

using namespace std; // standard namespace, avoid writing std::count, std::endl, etc., Cpp-only

const double PI = 3.1415926; // C-styple: #define PI (3.1415926)

int main() {
    // 1. type
    int a = 3;
    float b = 3.005;
    double c = 3.006;
    char d = 'd';
    bool e = true; // Cpp-only

    // ASCII values (-128 ~ 127):
    // - 'A'–'Z' are 65–90
    // - 'a'–'z' are 97–122
    // - '0'–'9' are 48–57
    cout << int('A') << endl;
    
    // 2. printf (printf is C-style, cout is Cpp-Style)
    printf("%d\n", a);
    printf("%.2f\n", b);
    printf("%.2lf\n", c);
    printf("%c\n", d);
    printf("%d\n\n", e);
    
    // 3. if; and &&, or ||, not !
    if (a > 5) // use int in compare, avoid float/double because of precision issue
    {
        cout << "a > 5\n" << endl;
    }
    else if (a < 5) 
    {
        cout << "a < 5\n" << endl;
    }
    else 
    {
        cout << "a = 5\n" << endl;
    }
    
    // 4. while
    int i = 0;
    while (i < 3)
    {
        cout << i << endl;
        i ++ ;
    }
    printf("\n");
    
    // 5. do while, do at least once, rarely used
    int j = 1;
    do
    {
        cout << j << endl;
    } while (j < 1);
    printf("\n");
    
    // 6. for
    for (int i = 0; i < 3; i ++ )
    {
        cout << i << endl;
    }
    printf("\n");
    
    // 7. break
    for (int i = 0; i < 100; i ++ )
    {
        cout << i << endl;
        if (i == 2) break;
    }
    printf("\n");
    
    // 8. continue
    for (int i = 0; i <= 6; i ++ )
    {
        if (i % 2 == 0) continue;
        cout << i << endl;
    }
    printf("\n");
    
    // 9. C-number/char-array, fixed length, decay to pointer
    // number
    int arr[] = {0, 1, 2};
    cout << arr[0] << ' ' << arr[1] << ' ' << arr[2] << endl;
    int arr2d[2][3] = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << arr2d[i][j] << " ";
        }
        cout << endl;
    }

    // char, has terminator '\0'
    char c1[] = "Hello World", c2[] = "Hi World";
    printf("%s\n", c1);
    cout << strlen(c1) << endl; // len
    cout << strcmp(c1, c2) << endl; // compare
    strcpy(c1, c2); // copy
    strcat(c1, "!"); // concatenate
    cout << c1 << endl;
    printf("\n");

    // 10. Cpp-number-array std::array, fixed length, no decay to pointer
    array<int, 3> arr_cpp = {0, 1, 2};
    cout << arr_cpp[0] << ' ' << arr_cpp[1] << ' ' << arr_cpp[2] << endl;
    cout << arr_cpp.size() << endl;
    array<array<int, 3>, 2> arr2d_cpp = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << arr2d_cpp[i][j] << " ";
        }
        cout << endl;
    }
    printf("\n");

    // 11. Cpp-number-array std::vector, dynamic length, no decay to pointer
    vector<int> v = {0, 1, 2};
    cout << v[0] << ' ' << v[1] << ' ' << v[2] << endl;
    cout << v.size() << endl;
    vector<vector<int>> v2d(2, vector<int>(3, 0));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << v2d[i][j] << " ";
        }
        cout << endl;
    }
    printf("\n");

    // 12. Cpp-char-array std::string, dynamic length, no decay to pointer
    string s1 = "Hello World", s2;
    printf("%s\n", s1.c_str());
    cout << s1.size() << endl;
    s2 = s1 + "!";
    cout << s2 << endl;
    printf("\n");
    
    // 13. reference &: another name for the same variable, Cpp-only
    int x = 1;
    int &y = x;
    y += 1;
    cout << x << endl;
    printf("\n");
    
    // 14. pointer p [store the address of / point to] varibale z
    int z = 1;
    int *p = &z; // & is address operator; * is dereference operator
    *p += 1;
    cout << z << endl;
    
    // C-number/char-array is the pointer to first element
    // recall: int arr[] = {0, 1, 2};
    cout << *arr << endl; // arr[0] = *arr
    cout << *(arr + 2) << endl; // arr[2] = *(arr+2)
    printf("\n");

    // const pointer
    const int *p1 = &z; // can change the pointer (p1 = &x; ok); cannot modify *p1 (*p1 = 2; error)
    int *const p2 = &z; // cannot change the pointer (p2 = &x; error); can modify *p2 (*p2 = 2; ok)
    printf("\n");

    // 15. new & delete, used when we want the variable to survive beyond the a local scope
    // the following code is not a good example, just for grammar
    int* heap_p = new int(5); // C: int* heap_p = (int*)malloc(sizeof(int)); *heap_p = 5;
    delete heap_p; // free heap memory; C: free(heap_p);
    heap_p = nullptr; // avoid dangling pointer; C: heap_p = NULL;
    // 3 kinds of places a variable can live
    // - local variable: inside {} block, live in stack, automatically destroyed after the local scope ends
    //   stackoverflow: reach the limit of stack memory, the program crash
    // - global variable: outside all {} blocks, live in static storage, automatically destroyed after the program ends
    // - heap variable: created by new, live in heap, manually destroyed by delete; if never deleted, it will cause memory leak (waste memory)
    //   this heap is completely different from the heap in data structure

    // new & delete is usually used for array, not for single variable
    int* heap_arr = new int[3]{0, 1, 2}; // C: int* heap_arr = (int*)malloc(sizeof(int)*3); memcpy(heap_arr, arr, sizeof(int)*3);
    delete[] heap_arr; // C: free(heap_arr);
    heap_arr = nullptr; // C: heap_arr = NULL;

    // 16. switch, keep executing until hitting break
    int m = 2;
    switch (m) {
        case 1:
            cout << "one" << endl;
            break;
        case 2:
            cout << "two" << endl;
            break;
        default:
            cout << "other" << endl;
    }

    // 17. never use goto

    return 0; // in Unix/Linux, 0 means the program ends successfully
}

