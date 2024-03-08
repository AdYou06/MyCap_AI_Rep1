# Task: Write a Python program to print all positive numbers in a range

#taking input and storing as a list
range1 = input("Enter the elements of the list(space-separated): ")
values = range1.split( )
list1 = [int(x) for x in values]

#treatment of the data
output = [x for x in list1 if x > 0] 
        
        
print("Input:", list1)
print("Output:" , output)
