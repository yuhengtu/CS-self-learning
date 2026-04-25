#include "real_dev.h"

Cat::Cat(int age) : age_(age) {}

int Cat::show_age() const {
    return age_;
}
