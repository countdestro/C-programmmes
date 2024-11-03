#include<stdio.h>
void main()
{
    int f[100] ,i,n;
    printf("enter the number values you need fibonacci no \n");
    scanf("%d",&n);
    f[0]=0;
    f[1]=1;
    for(i=2;i<n;i++)
    {
        f[i]=f[i-1]+f[i-2];
    }
     printf("the fibonacci no for first %d \n",n);
     for(i=2;i<n;i++)
     {
        printf("%d\t",f[i]);
     }
}
 
