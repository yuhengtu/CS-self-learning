// 求浮点数三次方根

#include<iostream>
#include<cstdio>
using namespace std;
const int N =1e6+10;
double x;

int main()
{
    scanf("%lf", &x);
    double l=-10000, r=10000;//不可l=0，不可l=-x,r=x , 不知x正负
    while(r-l>1e-8)
    {
        double mid= (l+r)/2;
        if(mid*mid*mid >= x) r=mid;
        else l=mid;
    }
    printf("%lf\n",l);
    return 0;
    
}

