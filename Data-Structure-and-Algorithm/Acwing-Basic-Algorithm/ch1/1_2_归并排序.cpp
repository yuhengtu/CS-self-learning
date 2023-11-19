//输入样例：
//5
//3 1 2 4 5
//输出样例：
//1 2 3 4 5

#include<iostream>
#include<cstdio>
using namespace std;
const int N =1e6+10;
int n, q[N],temp[N];
void merge_sort(int *, int, int);

int main()
{
    scanf("%d", &n);
    
    for (int i=0;i<n;i++)  scanf("%d", &q[i]);
    merge_sort(q,0,n-1);
    for (int i=0;i<n;i++)  printf("%d ", q[i]);
    
    return 0;
}

void merge_sort(int q[], int l, int r)
{
    if(l>=r) return;
    
    int mid = r+l >>1;
    int k=0, i=l, j =mid+1;
    merge_sort(q, l, mid);merge_sort(q, mid+1, r);
    
    while(i<=mid && j<=r)
    {
        if(q[i]<=q[j]) temp[k++]=q[i++];
        else temp[k++]=q[j++];
    }
    while(i<=mid) temp[k++]=q[i++];
    while(j<=r) temp[k++]=q[j++];
    for(int i=l,j=0; i<=r; i++,j++) q[i]=temp[j];
    
}
