import numpy as np
import sympy as sp
import math
import logging
import random
import time
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import gmpy2
from gmpy2 import mpz, isqrt
from sympy import primerange, sieve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_legendre(args):
    n_mpz, p = args
    p_mpz = mpz(p)
    return p if gmpy2.legendre(n_mpz, p_mpz) == 1 else None

def generate_factor_base(n: int, bound: int) -> List[int]:
    n_mpz = mpz(n)
    
    digits = len(str(n))
    max_fb_size = min(10**6, int(digits * math.log(digits)))
    
    logger.info(f"Generating primes up to {bound} with max factor base size: {max_fb_size}")
    
    factor_base = []
    segment_size = 10**6
    
    with Pool(processes=cpu_count()) as pool:
        for start in range(3, bound, segment_size):
            end = min(start + segment_size, bound)
            primes = list(primerange(start, end))
            
            chunk_results = pool.map(check_legendre, [(n_mpz, p) for p in primes])
            factor_base.extend(p for p in chunk_results if p is not None)
            
            if len(factor_base) >= max_fb_size:
                factor_base = factor_base[:max_fb_size]
                break
    
    logger.info(f"Generated factor base with {len(factor_base)} primes")
    return factor_base

def tonelli_shanks(n: int, p: int) -> int:
    assert gmpy2.legendre(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        return pow(n, (p + 1) // 4, p)
    for z in range(2, p):
        if p - 1 == gmpy2.legendre(z, p):
            break
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return r

def is_smooth(n: int, factor_base: List[int]) -> bool:
    n = abs(n)
    small_primes = factor_base[:100]
    
    for p in small_primes:
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    
    if n > 1:
        return gmpy2.is_prime(n) and n in factor_base
    
    return True

def calculate_bound(n: float) -> int:
    log_n = math.log(n)
    log_log_n = math.log(log_n)
    return int(math.exp(math.sqrt(log_n * log_log_n)))

def quadratic_sieve(n: int, factor_base: List[int], sieve_bound: int, time_limit: float) -> List[Tuple[int, int]]:
    smooth_relations = []
    start_time = time.time()
    last_report_time = start_time
    
    segment_size = min(10**6, sieve_bound)
    num_segments = (2 * sieve_bound + segment_size - 1) // segment_size
    
    for segment in range(num_segments):
        if time.time() - start_time > time_limit:
            logger.info("Time limit reached in QS. Stopping early.")
            break
        
        start = segment * segment_size - sieve_bound
        end = min((segment + 1) * segment_size - sieve_bound, sieve_bound)
        
        sieve_array = np.zeros(end - start, dtype=np.int32)
        x_values = np.arange(start, end)
        y_values = x_values * x_values - n
        
        for p in factor_base:
            p_mpz = mpz(p)
            try:
                r = tonelli_shanks(n, p)
                x1 = (r - start) % p
                x2 = (-r - start) % p
                while x1 < len(sieve_array):
                    sieve_array[x1] += int(gmpy2.log2(p))
                    x1 += p
                while x2 < len(sieve_array):
                    sieve_array[x2] += int(gmpy2.log2(p))
                    x2 += p
            except Exception as e:
                logger.debug(f"Skipping prime {p} due to error: {str(e)}")
                continue
        
        threshold = int(gmpy2.log2(sieve_bound * sieve_bound))
        smooth_indices = np.where(sieve_array >= threshold)[0]
        
        for i in smooth_indices:
            x = start + i
            y = int(y_values[i])
            if is_smooth(abs(y), factor_base):
                smooth_relations.append((x, y))
            
            if len(smooth_relations) > len(factor_base) + 100:
                logger.info(f"Found sufficient smooth relations: {len(smooth_relations)}. Stopping early.")
                return smooth_relations
        
        current_time = time.time()
        if current_time - last_report_time > 5:
            progress = (segment + 1) / num_segments * 100
            logger.info(f"QS progress: {progress:.2f}%")
            last_report_time = current_time
    
    return smooth_relations

def mpqs(n: int, factor_base: List[int], sieve_bound: int, time_limit: float) -> List[Tuple[int, int]]:
    smooth_relations = []
    start_time = time.time()
    last_report_time = start_time
    last_progress_time = start_time
    
    digits = len(str(n))
    sieve_bound = min(sieve_bound, 10**8)  # Increased sieve bound for larger numbers
    segment_size = min(10**6, sieve_bound // 10)  # Increased segment size
    
    num_segments = (2 * sieve_bound + segment_size - 1) // segment_size
    
    a = isqrt(2 * n)
    k = 1
    
    for segment in range(num_segments):
        current_time = time.time()
        if current_time - start_time > time_limit:
            logger.info(f"Time limit reached in MPQS after {time_limit:.2f} seconds. Stopping early.")
            break
        
        start = segment * segment_size - sieve_bound
        end = min((segment + 1) * segment_size - sieve_bound, sieve_bound)
        
        sieve_array = np.zeros(end - start, dtype=np.int32)
        x_values = np.arange(start, end)
        
        b = isqrt(k * n)
        c = b * b - k * n
        
        y_values = a * a * x_values * x_values + 2 * a * b * x_values + c
        
        for p in factor_base:
            p_mpz = mpz(p)
            try:
                if gmpy2.legendre(n, p_mpz) == 1:
                    r = tonelli_shanks(n, p)
                    x1 = (r - start) % p
                    x2 = (-r - start) % p
                    while x1 < len(sieve_array):
                        sieve_array[x1] += int(gmpy2.log2(p))
                        x1 += p
                    while x2 < len(sieve_array):
                        sieve_array[x2] += int(gmpy2.log2(p))
                        x2 += p
            except Exception as e:
                logger.debug(f"Skipping prime {p} due to error: {str(e)}")
                continue
        
        threshold = int(gmpy2.log2(sieve_bound * sieve_bound))
        smooth_indices = np.where(sieve_array >= threshold)[0]
        
        for i in smooth_indices:
            x = start + i
            y = int(y_values[i])
            if is_smooth(abs(y), factor_base):
                smooth_relations.append((x, y))
        
        if current_time - last_report_time > 5:
            progress = (segment + 1) / num_segments * 100
            logger.info(f"MPQS progress: {progress:.2f}% | Smooth relations: {len(smooth_relations)} | Current polynomial: k={k}")
            last_report_time = current_time
        
        if len(smooth_relations) > 0:
            last_progress_time = current_time
        elif current_time - last_progress_time > 60:
            logger.info(f"No progress made for 1 minute. Switching polynomial. k={k} -> k={k+1}")
            k += 1
            last_progress_time = current_time
        
        if len(smooth_relations) > len(factor_base) + 100:
            logger.info(f"Found sufficient smooth relations: {len(smooth_relations)}. Stopping early.")
            break
    
    return smooth_relations

def gnfs(n: int, factor_base: List[int], sieve_bound: int, time_limit: float) -> List[Tuple[int, int]]:
    logger.info("GNFS not fully implemented yet. Using MPQS as a fallback.")
    return mpqs(n, factor_base, sieve_bound, time_limit)

def select_algorithm(n: int) -> str:
    digits = len(str(n))
    if digits <= 30:
        return "QS"
    elif 30 < digits < 100:
        return "MPQS"
    else:
        return "GNFS"

def factorize(n: int, time_limit: float = 3600) -> Tuple[int, int]:
    algorithm = select_algorithm(n)
    logger.info(f"Selected algorithm: {algorithm}")
    
    bound = calculate_bound(float(n))
    factor_base = generate_factor_base(n, bound)
    
    sieve_bound = int(pow(n, 1/3))  # Increased sieve bound
    smooth_relations = []
    
    try:
        if algorithm == "QS":
            relations = quadratic_sieve(n, factor_base, sieve_bound, time_limit)
        elif algorithm == "MPQS":
            relations = mpqs(n, factor_base, sieve_bound, time_limit)
        else:  # GNFS
            relations = gnfs(n, factor_base, sieve_bound, time_limit)
        
        smooth_relations.extend(relations)
    except Exception as e:
        logger.error(f"Error in {algorithm}: {str(e)}")
    
    logger.info(f"Found {len(smooth_relations)} smooth relations")
    
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i, n // i
    
    logger.info("No factors found using trial division. The number might be prime or require more advanced techniques.")
    return 1, n

def main():
    try:
        test_numbers = [
            10403,  # Small number (QS)
            10**30 + 87,  # Medium number (MPQS)
            1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139,  # Large number (GNFS)
        ]
        
        for n in test_numbers:
            logger.info(f"Starting factorization for n = {n}")
            start_time = time.time()
            p, q = factorize(n, time_limit=600)  # 10 minutes time limit
            end_time = time.time()
            
            if p != 1 and q != n:
                logger.info(f"Factors found: {p} and {q}")
            else:
                logger.info("No non-trivial factors found.")
            
            logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
            logger.info("------------------------")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Stack trace:")

if __name__ == "__main__":
    main()
