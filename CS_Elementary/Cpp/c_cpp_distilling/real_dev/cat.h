#ifndef CAT_H
#define CAT_H

const double pi = 3.1415926;

class Cat {
private:
    int age_;

public:
    Cat(int age);
    int show_age() const; // const function, not allowed to modify variables
};

#endif