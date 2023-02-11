# Import the GNFS function
from GNFS_Integer_Factorization_RSA import gnfs

# Call the GNFS function with the number you want to factorize
result = gnfs(123456789)

# Print the result, which should be the two prime factors of the input number
print(result)