# Import the GNFS function
from GNFS_Integer_Factorization_RSA import gnfs
from Generate_n import generate_rsa_keys

# Call the GNFS function with the number you want to factorize
result = gnfs(n)

# Print the result, which should be the two prime factors of the input number
print(result)