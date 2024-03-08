# Function to generate Fibonacci sequence
def fibonacci(n):
    fib_sequence = [0, 1]

    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

    return fib_sequence

# Get user input for the number of Fibonacci numbers to generate
n = int(input("Enter the number of Fibonacci numbers to generate: "))

# Generate and display the Fibonacci sequence
result = fibonacci(n)
print(f"The first {n} Fibonacci numbers are: {result}")
