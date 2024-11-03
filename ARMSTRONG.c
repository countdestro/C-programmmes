#include<stdio.h>
#include<math.h>
int main()
{
    long long int x,sum,temp,n,digit,z=0;
    printf("enter upto which armstrong no you want \n");
    scanf("%d",&z);
    for(x=10;x<z;x++)
    {
        sum=0;
        temp=x;
        n=0;
        while(temp>0)
        {
            n++;
            temp/=10;
        }
        temp=x;
        while(temp>0)
        {
            digit=temp%10;
            sum+=pow(digit,n);
            temp/=10;
        }
        if(sum==x)
        {
            printf("%d is armstrong number\n",x);
        }

    }
    return(0);
}