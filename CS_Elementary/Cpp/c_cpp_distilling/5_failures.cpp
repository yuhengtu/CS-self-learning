#include <stdio.h>

// 1. dangling pointer
char *return_c_char_array() {
    char str[] = "hello";
    return str;  // str is local and decay to pointer, destroyed after fn ends, return a dangling pointer
}

// 2. array bound
int* foo = new int[100];
for (int i = 0; i <= 100; ++i) 
{
    foo[i] = 0; // out of bound for i = 100
}