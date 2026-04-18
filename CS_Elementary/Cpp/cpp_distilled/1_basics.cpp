#include <iostream>
#include <cstdio> // for printf
#include <string.h> // for strlen, strcmp, strcpy
#include <string> // for string

using namespace std;

const double pi = 3.1415926;

int main() {
    // 1. type
    int a = 3;
    float b = 3.005;
    double c = 3.006;
    char d = 'd';
    bool e = true;
    
    // 2. printf
    printf("%d\n", a);
    printf("%.2f\n", b);
    printf("%.2lf\n", c);
    printf("%c\n", d);
    printf("%d\n\n", e);
    
    // 3. if; and &&, or ||, not !
    if (a > 5) 
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
    
    // 5. do while, do at least once
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
    
    // 9. number array
    int arr[] = {0, 1, 2};
    cout << arr[0] << ' ' << arr[1] << ' ' << arr[2] << endl;
    int arr2d[2][3] = {0};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << arr2d[i][j] << " ";
        }
        cout << endl;
    }
    printf("\n");
    
    //10. ASCII values (-128 ~ 127):
    // - 'A'–'Z' are 65–90
    // - 'a'–'z' are 97–122
    // - '0'–'9' are 48–57
    cout << int('A') << endl;
    printf("\n");
    
    // 11. C-string: char array with fixed length, has terminator '\0'
    char c1[] = "Hello World", c2[] = "Hi World";
    printf("%s\n", c1);
    cout << strlen(c1) << endl; // size = 11 + 1 ('\0')
    cout << strcmp(c1, c2) << endl; // compare

    strcpy(c1, c2); // copy
    cout << c1 << endl;
    printf("\n");
    
    // 12. (Cpp-)string: char array with non-fixed length
    string s1 = "Hello World", s2;
    cout << s1 << endl;
    printf("%s\n", s1.c_str());
    cout << s1.size() << endl;
    s2 = s1 + "!";
    cout << s2 << endl;
    printf("\n");
    
    // 13. reference &: another name for the same variable
    int x = 1;
    int &y = x;
    y += 1;
    cout << x << endl;
    printf("\n");
    
    // 14. pointer p [store the address of / point to] varibale z
    int z = 1;
    int *p = &z;
    *p += 1;
    cout << z << endl;
    
    // array is the pointer to first element
    // recall: int arr[] = {0, 1, 2};
    cout << *arr << endl;
    cout << *(arr + 2) << endl;
    printf("\n");

    // 15. new & delete, used when we want the variable to survive beyond the a local scope
    // the following code is not a good example, just for grammar
    int* heap_p = new int(5);
    delete heap_p;
    heap_p = nullptr;
    // 3 kinds of places a variable can live
    // - local variable: inside {} block, live in stack, automatically destroyed after the local scope ends
    // - global variable: outside all {} blocks, live in the global storage area, automatically destroyed after the program ends
    // - heap variable: created by new, live in the heap, manually destroyed by delete; if never deleted, it will cause memory leak (waste memory)

    // 16. switch
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

    return 0;
}
