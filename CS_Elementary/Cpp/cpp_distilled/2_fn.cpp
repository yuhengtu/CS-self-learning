// Local variable is limited inside the function
// If global and local variables share a name, the local one overrides

#include <iostream>
#include <cstdio>

using namespace std;

// do not modify value: pass by value
int var_plus_one_value(int var)
{
    var += 1;
    return var;
}

// modify value: pass by reference
void var_plus_one_reference(int &var)
{
    var += 1;
    return; // optional
}

// modify value: pass by pointer
void var_plus_one_pointer(int *var)
{
    *var += 1;
    // return; // optional
}

// recall: array is the pointer to first element
// thus, without &, the fn still modify the array
void set_first_element_to_one(int arr1d[])
// void set_first_element_to_one(int *arr1d) // equivalent
{
    arr1d[0] = 1;
}
// for multi-dim array, all dims except the first must be specified
// void fn_name(int (*arr_3d)[3][4])
// void fn_name(int arr_3d[][3][4])

int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

int main()
{
    // 1. pass by value / reference / pointer
    int x = 1;
    int y = var_plus_one_value(x);
    cout << x << endl;
    cout << y << endl;

    var_plus_one_reference(x);
    cout << x << endl;

    var_plus_one_pointer(&x);
    cout << x << endl;

    int arr[] = {0, 1, 2};
    set_first_element_to_one(arr);
    cout << arr[0] << endl;
    printf("\n");

    // 2. function overloading: same name, different parameter types
    cout << add(1, 2) << endl;
    cout << add(1.5, 2.5) << endl;

    return 0;
}