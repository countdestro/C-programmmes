#include<stdio.h>
void main()
{
    int  a[100],i,n,key,low,mid,f=0,high;
    printf("enter the size of array \n");
    scanf("%d",&n);
    printf("enter array values \n");
    for(i=0;i<n;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("enter the element to be searched \n");
    scanf("%d",&key);
    low=0;
    high=n-1;
    while(low<=high)
    {
        mid=(high+low)/2;
        if(a[mid]==key)
        {
          f=1;
          break;  
        }
        else
        {
            low=low+1;
            high=high-1;
        }
    }
    if(f==1)
    {
        printf("element %d found at %d \n",key,mid);
    }
    else
    {
        printf("element not found \n");
    }
}