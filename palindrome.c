#include<stdio.h>
void main()
{
    int n,rn=0,r,on;
    printf("enter the number to be checked \n");
    scanf("%d",&n);
    on=n;
    while(n !=0)
    {
        r=n%10;
        rn=rn*10+r;
        n/=10;
    }
    if(on==rn)
    {
        printf("the no %d is a palindrome\n",on);
        
    }
    else
    {
        printf("the no %d is not a palindrome \n",on);
    }
}

