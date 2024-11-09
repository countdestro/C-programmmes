#include<stdio.h>
void main()
{
    int a[100],i,j,pos,n,temp;
    printf("enter the no of element\n");
    scanf("%d",&n);
    printf("enter the elements of array \n");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("the elements before sorting \n");
    for(i=0;i<n;i++)
    {
        printf("%d\t",a[i]);
    }
    for(i=0;i<n;i++)
    {
        pos=i;
        for(j=i+1;j<n;j++)
         if(a[pos]>a[j])
            pos=j;
            temp=a[i];
            a[i]=a[pos];
            a[pos]=temp;
         
        
    }
    printf("\nthe elements after sorting:\n");
    for(i=0;i<n;i++)
    {
        printf("%d\t",a[i]);
    }    
}