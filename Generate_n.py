import random
from math import gcd

def generate_rsa_keys(key_size=2048):
    print("Generating RSA keys of size", key_size)
    # Generate two large prime numbers, p and q
    p = generate_large_prime(key_size // 2)
    q = generate_large_prime(key_size // 2)
    
    print("p:", p)
    print("q:", q)
    
    # Calculate n, which is used as the modulus for both the public and private keys
    n = p * q
    print("n:", n)
    
    # Calculate the totient of n, denoted as phi(n)
    totient = (p - 1) * (q - 1)
    print("totient:", totient)
    
    # Choose a public key exponent, e, such that 1 < e < phi(n) and gcd(e, phi(n)) = 1
    e = 65537
    while gcd(e, totient) != 1:
        e = random.randint(2, totient - 1)
    
    print("e:", e)
    
    # Calculate the private key exponent, d, such that d * e = 1 (mod phi(n))
    d = mod_inverse(e, totient)
    print("d:", d)
    
    # Return the public and private keys as tuples (e, n) and (d, n) respectively
    return (e, n), (d, n)

def generate_large_prime(key_size):
    print("Generating large prime of size", key_size)
    # Generate a random number of the desired key size
    number = random.randint(2**(key_size - 1), 2**key_size - 1)
    
    # Test the number for primality and return it if it is prime
    return get_next_prime(number)

def get_next_prime(number):
    print("Getting next prime starting from", number)
    # Increment the number until it is prime
    while not is_prime(number):
        number += 1
    
    return number

def is_prime(number):
    print("Checking if", number, "is prime")
    # Test the number for primality by checking for divisors up to its square root
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    
    return True

def mod_inverse(a, m):
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None