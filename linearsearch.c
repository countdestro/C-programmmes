#include<stdio.h>
void main()
{
    int a[100],i,n,key,f=0;
    printf("enter the size of array \n");
    scanf("%d",&n);
    printf("enter the elements of array \n");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf(" enter the no to be searched \n");
    scanf("%d",&key);
    for(i=0;i<n;i++)
    {
        if(a[i]==key)
        {
            f=1;
            break;
        }
    }
    if(f==1)
    {
        printf("the no %d is present at %d \n",key,i+1);
    }
    else
    {
        printf("number not found \n");
    }

}