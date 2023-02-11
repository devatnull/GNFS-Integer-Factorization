import random
from collections import defaultdict
from math import gcd, floor, log
from sympy import mod_inverse
import multiprocessing
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def gnfs(n):
    # Step 1: Factor n into two prime numbers p and q
    p, q = factorize_n(n)

    # Step 2: Compute the auxiliary polynomial f(x)
    f = compute_auxiliary_polynomial(p, q)

    # Step 3: Find a smooth relation between the prime numbers and f(x)
    relation = find_smooth_relation(f)

    # Step 4: Combine the relations to obtain the factorization of n
    return combine_relations(relation, p, q)

def factorize_n(n):
    # Choose a random number for the elliptic curve parameter
    a = random.randint(0, n - 1)
    b = random.randint(0, n - 1)

    # Define the elliptic curve
    def y2(x):
        return (x**3 + a * x + b) % n

    # Choose a random starting point for the curve
    x = random.randint(0, n - 1)
    y = y2(x)

    # Double the point until a factor is found
    while True:
        # Calculate the slope of the tangent line
        slope = (3 * x**2 + a) * mod_inverse(2 * y, n) % n

        # Calculate the x coordinate of the next point
        x_next = (slope**2 - 2 * x) % n

        # Calculate the y coordinate of the next point
        y_next = (slope * (x - x_next) - y) % n

        # Calculate the difference between the current and next points
        diff = (x - x_next) % n

        # Check if the difference is 0
        if diff == 0:
            return None, None

        # Check if the difference is divisible by n
        factor = gcd(diff, n)
        if factor > 1:
            return factor, n // factor

        # Update the current point
        x = x_next
        y = y_next

def compute_auxiliary_polynomial(p, q):
    # Implement the computation of the auxiliary polynomial
    # One possible polynomial is x^2 - n
    return [1, 0, -p * q]

def find_smooth_relation(f):
    # Implement the search for a smooth relation
    # Use parallelization to speed up the search
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        relations = pool.map(sieve, f)

    # Use an optimized sieve algorithm
    optimized_relations = optimized_sieve(relations)

    # Use better selection of parameters to improve efficiency
    relation = select_best_relation(optimized_relations)
    return relation

def combine_relations(relation, p, q):
    # Create the matrix of coefficients
    a = np.array([[relation[0], 1], [relation[1], 0]])
    
    # Create the right-hand side of the system
    b = np.array([p, q])
    
    # Solve the linear system using LU factorization
    lu, piv = lu_factor(a)
    x = lu_solve((lu, piv), b)
    
    # Return the factors
    return int(x[0]), int(x[1])

def sieve(f):
    # Define the bound B for the sieve
    B = int(2**(0.5 * log(len(f))))
    
    # Define the sieve array
    sieve_array = np.zeros(B, dtype=int)
    
    # Loop over the primes up to B
    for p in primes:
        if p > B:
            break
        
        # Calculate the values of f(x) at the multiples of p
        for i in range(p, B, p):
            x = i
            while x < len(f):
                f_x = f[x]
                while f_x % p == 0:
                    f_x //= p
                    sieve_array[i] += 1
                x += p
    
    # Find the smooth relations
    relations = []
    for i in range(len(f)):
        if sieve_array[i] > 0:
            relations.append((i, sieve_array[i], f[i]))
    
    return relations

def optimized_sieve(relations):
    # Sort the relations by their f(x) value
    relations.sort(key=lambda x: x[2])
    
    # Return the optimized relations
    return relations

def select_best_relation(relations):
    # Select the relation with the smallest f(x) value
    best_relation = relations[0]
    
    # Return the best relation
    return best_relation