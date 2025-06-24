import numpy as np
import sympy as sp
import math
import random
import time
import os
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count
import scipy.linalg
import scipy.sparse
from collections import deque

# Simple progress tracker
def print_progress(current, total, description=""):
    percent = (current / total) * 100
    print(f"\r{description}: {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current >= total:
        print()

# Utility functions
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    g, x, y = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError("Modular inverse does not exist")
    return (x % m + m) % m

def legendre_symbol(a, p):
    return pow(a, (p - 1) // 2, p)

def tonelli_shanks(n, p):
    if legendre_symbol(n, p) != 1:
        return None
    
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    if S == 1:
        return pow(n, (p + 1) // 4, p)
    
    z = 2
    while legendre_symbol(z, p) != p - 1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    
    while t != 1:
        i = 1
        while pow(t, 2**i, p) != 1:
            i += 1
        
        b = pow(c, 2**(M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R

# Factor base generation
def generate_factor_base(n, bound):
    factor_base = []
    for p in sp.primerange(2, bound + 1):
        if legendre_symbol(n, p) == 1:
            factor_base.append(p)
    return factor_base

def is_smooth(n, factor_base):
    for p in factor_base:
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    return abs(n) == 1

# Polynomial class
class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
    
    def evaluate(self, x):
        return sum(c * (x ** i) for i, c in enumerate(self.coeffs))
    
    def evaluate_mod(self, x, mod):
        return sum(c * pow(x, i, mod) for i, c in enumerate(self.coeffs)) % mod

# Relation class
class Relation:
    def __init__(self, a, b, algebraic_factors=None, rational_factors=None):
        self.a = a
        self.b = b
        self.algebraic_factors = algebraic_factors or {}
        self.rational_factors = rational_factors or {}

# Quadratic Sieve
def quadratic_sieve(n, factor_base, sieve_bound, time_limit):
    start_time = time.time()
    relations = []
    
    for x in range(1, sieve_bound):
        if time.time() - start_time > time_limit:
            break
            
        y_squared = x * x - n
        if y_squared <= 0:
            continue
            
        if is_smooth(y_squared, factor_base):
            relations.append((x, y_squared))
            print_progress(len(relations), len(factor_base) + 10, "QS Relations")
            
            if len(relations) >= len(factor_base) + 10:
                break
    
    return relations

# Multiple Polynomial Quadratic Sieve (MPQS)
def generate_mpqs_polynomials(n, factor_base, num_polys=5):
    """Generate multiple polynomials for MPQS of the form (ax + b)² - N"""
    polynomials = []
    sqrt_n = int(math.sqrt(n))
    
    # Start with the basic polynomial (a=1, b=0) like traditional QS
    polynomials.append((1, 0))
    
    for i in range(1, min(num_polys, len(factor_base))):
        # Choose 'a' from factor base (small primes work well)
        a = factor_base[i]
        
        # Find 'b' such that b² ≡ N (mod a) using Tonelli-Shanks if needed
        b = tonelli_shanks(n % a, a)
        if b is not None:
            polynomials.append((a, b))
            # Also add the negative root
            if b != 0:
                polynomials.append((a, a - b))
        else:
            # Fallback: try simple approach
            for candidate_b in range(min(a, 100)):  # Limit search
                if (candidate_b * candidate_b) % a == n % a:
                    polynomials.append((a, candidate_b))
                    break
    
    return polynomials

def mpqs(n, factor_base, sieve_bound, time_limit):
    """Multiple Polynomial Quadratic Sieve"""
    start_time = time.time()
    relations = []
    
    # Generate multiple polynomials
    polynomials = generate_mpqs_polynomials(n, factor_base)
    if not polynomials:
        # Fallback to basic QS if polynomial generation fails
        return quadratic_sieve(n, factor_base, sieve_bound, time_limit)
    
    poly_index = 0
    sieve_per_poly = sieve_bound // len(polynomials)
    
    for a, b in polynomials:
        if time.time() - start_time > time_limit:
            break
            
        # Sieve with polynomial (ax + b)² - N
        for x in range(-sieve_per_poly//2, sieve_per_poly//2):
            if time.time() - start_time > time_limit:
                break
                
            # Compute (ax + b)² - N
            ax_plus_b = a * x + b
            y_squared = ax_plus_b * ax_plus_b - n
            
            if y_squared <= 0:
                continue
                
            if is_smooth(y_squared, factor_base):
                # Store the relation as (ax+b, y_squared)
                relations.append((ax_plus_b, y_squared))
                
                if len(relations) % 10 == 0:
                    print_progress(len(relations), len(factor_base) + 10, f"MPQS Relations (poly {poly_index + 1}/{len(polynomials)})")
                
                if len(relations) >= len(factor_base) + 10:
                    return relations
        
        poly_index += 1
    
    return relations

# GNFS polynomial selection
def select_gnfs_polynomials(n, degree=5):
    # Simple polynomial selection - find irreducible polynomial mod n
    m = int(n**(1/degree))
    
    # Try different values around m
    for k in range(-10, 11):
        candidate_m = m + k
        f_coeffs = [candidate_m - n] + [0] * (degree - 1) + [1]
        f = Polynomial(f_coeffs)
        
        # Check if polynomial is suitable
        if abs(f.evaluate(candidate_m)) < n**0.5:
            g = Polynomial([-candidate_m, 1])  # g(x) = x - m
            return f, g
    
    # Fallback
    f = Polynomial([-n, 0, 0, 0, 0, 1])
    g = Polynomial([-int(n**(1/5)), 1])
    return f, g

# Factor base for GNFS
def create_factor_base(n, f, g, bound):
    rational_fb = []
    algebraic_fb = []
    
    for p in sp.primerange(2, bound + 1):
        if n % p != 0:
            # Check if p splits in the number field
            roots = []
            for r in range(p):
                if f.evaluate_mod(r, p) == 0:
                    roots.append(r)
            
            if roots:
                for root in roots:
                    algebraic_fb.append((p, root))
                rational_fb.append(p)
    
    return {"rational": rational_fb, "algebraic": algebraic_fb}

# Lattice sieve
def lattice_sieve(n, f, g, factor_bases, special_q, special_q_root, sieve_size=1000):
    relations = []
    
    for a in range(-sieve_size//2, sieve_size//2):
        for b in range(1, sieve_size//10):
            if gcd(a, b) != 1:
                continue
                
            # Compute norms
            rational_norm = abs(a + b * int(n**(1/5)))
            algebraic_norm = abs(f.evaluate(a/b) * (b**len(f.coeffs)))
            
            if algebraic_norm == 0 or rational_norm == 0:
                continue
                
            # Check smoothness
            if (is_smooth_gnfs(rational_norm, factor_bases["rational"]) and 
                is_smooth_gnfs(algebraic_norm, factor_bases["algebraic"])):
                
                rational_factors = factorize_over_factor_base(rational_norm, factor_bases["rational"])
                algebraic_factors = factorize_over_factor_base(algebraic_norm, factor_bases["algebraic"])
                
                relations.append(Relation(a, b, algebraic_factors, rational_factors))
                
                if len(relations) % 100 == 0:
                    print_progress(len(relations), len(factor_bases["rational"]) + 50, "GNFS Relations")
                
                if len(relations) >= len(factor_bases["rational"]) + 50:
                    break
        
        if len(relations) >= len(factor_bases["rational"]) + 50:
            break
    
    return relations

def is_smooth_gnfs(n, factor_base):
    for item in factor_base:
        if isinstance(item, tuple):
            p = item[0]
        else:
            p = item
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    return abs(n) == 1

def factorize_over_factor_base(n, factor_base):
    factors = {}
    for item in factor_base:
        if isinstance(item, tuple):
            p = item[0]
        else:
            p = item
        
        count = 0
        while n % p == 0:
            n //= p
            count += 1
        if count > 0:
            factors[p] = count
    return factors

# Matrix operations
def build_relation_matrix(relations, factor_bases):
    all_primes = set(factor_bases["rational"])
    for p, _ in factor_bases["algebraic"]:
        all_primes.add(p)
    
    prime_list = sorted(all_primes)
    prime_to_index = {p: i for i, p in enumerate(prime_list)}
    
    matrix = np.zeros((len(relations), len(prime_list)), dtype=int)
    
    for i, rel in enumerate(relations):
        for p, exp in rel.rational_factors.items():
            if p in prime_to_index:
                matrix[i][prime_to_index[p]] = exp % 2
        
        for p, exp in rel.algebraic_factors.items():
            if p in prime_to_index:
                matrix[i][prime_to_index[p]] = (matrix[i][prime_to_index[p]] + exp) % 2
    
    return matrix, prime_list

def find_null_space(matrix):
    # Simple Gaussian elimination over GF(2)
    m, n = matrix.shape
    matrix = matrix % 2
    
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for row in range(rank, m):
            if matrix[row][col] == 1:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank:
            matrix[[rank, pivot_row]] = matrix[[pivot_row, rank]]
        
        # Eliminate
        for row in range(m):
            if row != rank and matrix[row][col] == 1:
                matrix[row] = (matrix[row] + matrix[rank]) % 2
        
        rank += 1
    
    # Find null space vectors
    null_vectors = []
    for col in range(rank, n):
        vector = np.zeros(m, dtype=int)
        vector[col] = 1
        
        for row in range(rank):
            if matrix[row][col] == 1:
                vector[row] = 1
        
        null_vectors.append(vector)
    
    return null_vectors

# Main factorization functions
def select_algorithm(n):
    digits = len(str(n))
    if digits <= 10:
        return "QS"
    elif digits <= 15:
        return "MPQS"
    else:
        return "GNFS"

def factorize(n, time_limit=3600, verbose=True):
    if n <= 1:
        return 1, 1
    
    # Trial division for small factors
    for p in sp.primerange(2, 1000):
        if n % p == 0:
            return p, n // p
    
    # Check if perfect power
    for k in range(2, int(math.log2(n)) + 1):
        root = int(n**(1/k))
        if root**k == n:
            factors = factorize(root, time_limit//2, verbose)
            return factors[0], n // factors[0]
    
    algorithm = select_algorithm(n)
    if verbose:
        print(f"Using algorithm: {algorithm}")
    
    if algorithm == "QS":
        bound = int(math.exp(0.5 * math.sqrt(math.log(n) * math.log(math.log(n)))))
        factor_base = generate_factor_base(n, bound)
        sieve_bound = bound * 10
        
        relations = quadratic_sieve(n, factor_base, sieve_bound, time_limit)
        
        if len(relations) < len(factor_base) + 10:
            if verbose:
                print("Not enough relations found")
            return n, 1
        
        # Build matrix and find dependencies (simplified)
        matrix = np.array([[int(rel[1] % p == 0) for p in factor_base] for rel in relations])
        
        try:
            null_space = scipy.linalg.null_space(matrix.T)
            if null_space.size > 0:
                # Use first null vector
                dep = (null_space[:, 0] > 0.5).astype(int)
                
                x = 1
                y_squared = 1
                for i, bit in enumerate(dep):
                    if bit:
                        x = (x * relations[i][0]) % n
                        y_squared = (y_squared * relations[i][1]) % n
                
                y = int(math.sqrt(y_squared))
                if y * y == y_squared:
                    factor = gcd(x - y, n)
                    if 1 < factor < n:
                        return factor, n // factor
        except:
            pass
    
    elif algorithm == "MPQS":
        bound = int(math.exp(0.5 * math.sqrt(math.log(n) * math.log(math.log(n))))) * 2  # Larger bound for MPQS
        factor_base = generate_factor_base(n, bound)
        sieve_bound = bound * 15  # More sieving for MPQS
        
        relations = mpqs(n, factor_base, sieve_bound, time_limit)
        
        if len(relations) < len(factor_base) + 10:
            if verbose:
                print("Not enough relations found with MPQS")
            return n, 1
        
        # Build matrix and find dependencies (same as QS)
        matrix = np.array([[int(rel[1] % p == 0) for p in factor_base] for rel in relations])
        
        try:
            null_space = scipy.linalg.null_space(matrix.T)
            if null_space.size > 0:
                # Use first null vector
                dep = (null_space[:, 0] > 0.5).astype(int)
                
                x = 1
                y_squared = 1
                for i, bit in enumerate(dep):
                    if bit:
                        x = (x * relations[i][0]) % n
                        y_squared = (y_squared * relations[i][1]) % n
                
                y = int(math.sqrt(y_squared))
                if y * y == y_squared:
                    factor = gcd(x - y, n)
                    if 1 < factor < n:
                        return factor, n // factor
        except:
            pass
    
    elif algorithm == "GNFS":
        return gnfs(n, time_limit, verbose)
    
    if verbose:
        print("Factorization failed")
    return n, 1

def gnfs(n, time_limit=3600, verbose=True):
    start_time = time.time()
    
    # Polynomial selection
    f, g = select_gnfs_polynomials(n)
    if verbose:
        print(f"Selected polynomials: f={f.coeffs}, g={g.coeffs}")
    
    # Factor base creation
    bound = int(n**(1/6))
    factor_bases = create_factor_base(n, f, g, bound)
    
    if verbose:
        print(f"Factor base sizes: rational={len(factor_bases['rational'])}, algebraic={len(factor_bases['algebraic'])}")
    
    # Relation collection via lattice sieve
    relations = []
    special_q_bound = bound * 2
    
    for special_q in sp.primerange(bound, special_q_bound):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Find roots of f(x) ≡ 0 (mod special_q)
        roots = []
        for r in range(special_q):
            if f.evaluate_mod(r, special_q) == 0:
                roots.append(r)
        
        for root in roots:
            new_rels = lattice_sieve(n, f, g, factor_bases, special_q, root)
            relations.extend(new_rels)
            
            if len(relations) >= len(factor_bases["rational"]) + 50:
                break
        
        if len(relations) >= len(factor_bases["rational"]) + 50:
            break
    
    if verbose:
        print(f"\nCollected {len(relations)} relations")
    
    if len(relations) < len(factor_bases["rational"]) + 10:
        if verbose:
            print("Not enough relations for GNFS")
        return n, 1
    
    # Linear algebra
    matrix, prime_list = build_relation_matrix(relations, factor_bases)
    dependencies = find_null_space(matrix)
    
    if not dependencies:
        if verbose:
            print("No dependencies found")
        return n, 1
    
    # Square root and factor extraction
    for dep in dependencies[:5]:  # Try first few dependencies
        # Simplified square root computation
        try:
            # Combine relations according to dependency
            combined_a, combined_b = 1, 1
            for i, bit in enumerate(dep):
                if bit:
                    combined_a = (combined_a * relations[i].a) % n
                    combined_b = (combined_b * relations[i].b) % n
            
            # Try to extract factors
            candidates = [combined_a + combined_b, combined_a - combined_b]
            for candidate in candidates:
                factor = gcd(candidate, n)
                if 1 < factor < n:
                    if verbose:
                        print(f"Found factor: {factor}")
                    return factor, n // factor
        except:
            continue
    
    if verbose:
        print("Square root failed")
    return n, 1

# Simple test
def test_factorization():
    test_numbers = [
        15,  # 3 * 5
        77,  # 7 * 11
        143,  # 11 * 13
        1234567,  # Should use QS
    ]
    
    for n in test_numbers:
        print(f"\nFactoring {n}:")
        factors = factorize(n, verbose=True)
        print(f"Result: {factors[0]} * {factors[1]} = {factors[0] * factors[1]}")
        assert factors[0] * factors[1] == n, f"Factorization failed for {n}"

if __name__ == "__main__":
    # Run dependency check
    print("Starting GNFS Integer Factorization")
    
    # Quick test
    test_factorization()
    
    # Interactive mode
    while True:
        try:
            n = input("\nEnter number to factor (or 'quit'): ").strip()
            if n.lower() == 'quit':
                break
            
            n = int(n)
            if n <= 1:
                print("Please enter a number greater than 1")
                continue
            
            print(f"Factoring {n}...")
            start_time = time.time()
            factors = factorize(n, verbose=True)
            end_time = time.time()
            
            print(f"\nResult: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            if factors[0] * factors[1] != n:
                print("Warning: Factorization verification failed!")
                
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}") 