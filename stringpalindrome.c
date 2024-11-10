#include<stdio.h>
#include<string.h>
void main()
{
    char str[100],str1[100];
    int i,j,flag=0,n;
    printf("enter your string\n");
    gets(str);
    n=strlen(str);
    for(j=0,i=n-1;i>=0;i--,j++)
    {
        str1[j]=str[i];

    }
    str1[j]='\0';
    
    printf("reverse of string is\n");
    puts(str1);
    for(i=0;i<n;i++)
    {
        if(str[i]!=str1[i])
        {
            flag=1;
            break;
        }
   }
   if(flag==1)
   {
    printf("string is not palindrome\n");
   }
   else
   {
    printf("string is a palindrome\n");
   }
}
