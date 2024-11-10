#include<stdio.h>
#include<string.h>
void main()
{
    char str1[100],str2[100];
    int i,len,flag=0,j;
    printf("enter the first string\n");
    gets(str1);
    printf("enter the second string\n");
    gets(str2);
    len=strlen(str1);
    j=strlen(str2);
    if(len==j)
    {
    for(i=0;i<len;i++)
    {
        if(str2[i]!=str1[i])
        {
        
            flag=1;
        }   
        
    }
    if(flag==0)
    {
        printf("the strings are equal\n");
    }
    else
    {
        printf("the strings are not equal\n");
    }
    }
    else
    printf("the strings are not equal\n");
}   