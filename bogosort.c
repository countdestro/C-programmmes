#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<stdbool.h>
bool is_sorted(int *a,int n)
{
    int i;
    for(int i=1;i<n;i++)
    {
        if(a[i-1]>a[i])
        {
            return false;

        }
    }
    return true;
}
void shuffle(int *a,int n)
{
    for(int i=0;i<n;i++)
    {
        int j=rand() %n;
        int temp=a[i];
        a[i]=a[j];
        a[j]=temp;
    }
}
void bogo_sort(int *a,int n)
{
    while(!is_sorted(a,n))
    {
        shuffle(a,n);

    }
}
 void main()
 {
    int n,i;    
    printf("enter the size of array:\n");
    scanf("%d",&n);
    int a[n];
    printf("enter the values of array:\n");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    srand(time(0));
    bogo_sort(a,n);
    printf("Sorted array:\n"); 
    for (int i = 0; i < n; i++) 
    {
         printf("%d ", a[i]); 
    }

 }