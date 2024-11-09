#include<stdio.h>
void main()
{
    int a[100],i,j,n,temp;
    printf("enter the number of elements\n");
    scanf("%d",&n);
    printf("enter the elements of arrays\n");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("the array before sorting\n");
    for(i=0;i<n;i++)
    {
        printf("%d\t",a[i]);
    }
    for(i=0;i<n-1;i++)
    for(j=1;j<n-i;j++)
    {
        if(a[j]<a[j-1])
        {
            temp=a[j];
            a[j]=a[j-1];
            a[j-1]=temp;
        }
    }
    printf("\nthe array after sorting\n");
    for(j=0;j<n;j++)
    {
  printf("%d\t",a[j]);
}}