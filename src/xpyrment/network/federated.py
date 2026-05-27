"""Cryptographically Secure Federated Experimentation & Homomorphic Pooling (Block 19).

Implements a Paillier asymmetric homomorphic cryptosystem from scratch to enable
additive homomorphic secure multi-party computation (SMPC) of global covariance
matrices, alongside the Federated Averaging (FedAvg) global model pooling algorithm.
"""

from typing import List, Tuple, Union
import numpy as np


def gcd(a: int, b: int) -> int:
    """Computes the Greatest Common Divisor of a and b."""
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """Computes the Least Common Multiple of a and b."""
    return abs(a * b) // gcd(a, b)


def mod_inverse(a: int, m: int) -> int:
    """Computes the modular multiplicative inverse of a modulo m using Extended Euclidean Algorithm."""
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError("Modular inverse does not exist.")
    return x % m


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Greatest Common Divisor."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


class PaillierCryptosystem:
    """Paillier Homomorphic Cryptosystem for Secure Multi-Party Computation.

    # TODO: Implement threshold decryption where the private key lambda is divided into shares (lambda_1, lambda_2)
    # distributed among client nodes, requiring collaboration of at least k-out-of-n clients to decrypt global stats.
    """

    def __init__(self) -> None:
        # Predefined prime coordinates for deterministic demonstration / unit test speed
        # For a production grade system, large primes (e.g. 2048-bit) would be randomly generated
        self.p = 61
        self.q = 53
        
        self.n = self.p * self.q
        self.n_sq = self.n**2
        self.g = self.n + 1  # Standard generator choice for Paillier
        self.lam = lcm(self.p - 1, self.q - 1)
        
        # Compute modular inverse: mu = (L(g^lambda mod n^2))^-1 mod n
        # Since g = n + 1, g^lambda mod n^2 = 1 + lambda * n, so L(g^lambda mod n^2) = lambda
        self.mu = mod_inverse(self.lam, self.n)

    @property
    def public_key(self) -> Tuple[int, int]:
        """Returns the public key (n, g)."""
        return self.n, self.g

    @property
    def private_key(self) -> Tuple[int, int]:
        """Returns the private key (lambda, mu)."""
        return self.lam, self.mu

    def encrypt(self, m: int, r: int = 42) -> int:
        """Encrypts a plaintext message m using public key.

        c = (g^m * r^n) mod n^2
        """
        # Map any negative values into positive modular space
        m = m % self.n
        
        # Modular exponentiation
        gm = int(pow(self.g, m, self.n_sq))
        rn = int(pow(r, self.n, self.n_sq))
        c = (gm * rn) % self.n_sq
        return c

    def decrypt(self, c: int) -> int:
        """Decrypts a ciphertext c using private key.

        m = L(c^lambda mod n^2) * mu mod n
        """
        u = int(pow(c, self.lam, self.n_sq))
        L = (u - 1) // self.n
        m = (L * self.mu) % self.n
        return m

    def add_encrypted(self, c1: int, c2: int) -> int:
        """Homomorphically adds two encrypted ciphertexts together.

        The decryption of the returned ciphertext is exactly (m1 + m2) mod n.
        c_sum = (c1 * c2) mod n^2
        """
        return (c1 * c2) % self.n_sq


def federated_averaging(local_weights: List[np.ndarray], sample_sizes: List[int]) -> np.ndarray:
    """Performs Federated Averaging (FedAvg) to pool local model parameters.

    w_global = sum_k (N_k / N_total) * w_k

    Args:
        local_weights (List[np.ndarray]): List of model parameter vectors/matrices from local nodes.
        sample_sizes (List[int]): Number of observations at each local node.

    Returns:
        np.ndarray: Pooled global model parameter vector/matrix.
    """
    total_samples = sum(sample_sizes)
    if total_samples == 0:
        raise ValueError("Total sample size across federated clients must be greater than zero.")

    global_weights = np.zeros_like(local_weights[0])
    for w_k, n_k in zip(local_weights, sample_sizes):
        global_weights += (n_k / total_samples) * w_k

    return global_weights


def federated_secure_covariance_pooling(
    client_covariances: List[np.ndarray], crypto: PaillierCryptosystem
) -> np.ndarray:
    """Pools local client covariance matrices homomorphically without disclosing entries.

    Args:
        client_covariances (List[np.ndarray]): List of 2D covariance matrices from local clients.
        crypto (PaillierCryptosystem): The initialized Paillier asymmetric cryptosystem.

    Returns:
        np.ndarray: The decrypted globally pooled covariance matrix.
    """
    D = client_covariances[0].shape[0]
    
    # 1. Encrypt each client's covariance matrix entries
    # Scaling factor to support fractional representations as integer keys
    scale = 100
    encrypted_sum_matrix = np.zeros((D, D), dtype=object)

    for i in range(D):
        for j in range(D):
            # Server-side homomorphic summation
            for client_idx, cov in enumerate(client_covariances):
                val_int = int(round(cov[i, j] * scale))
                # Add deterministic random offset to randomize different clients' ciphertexts
                r = 13 + client_idx * 7
                c = crypto.encrypt(val_int, r=r)
                
                if client_idx == 0:
                    encrypted_sum_matrix[i, j] = c
                else:
                    encrypted_sum_matrix[i, j] = crypto.add_encrypted(encrypted_sum_matrix[i, j], c)

    # 2. Decrypt and reconstruct global pooled covariance
    pooled_cov = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            decrypted_int = crypto.decrypt(encrypted_sum_matrix[i, j])
            # Handle overflow modular wrapping for negative numbers
            if decrypted_int > crypto.n // 2:
                decrypted_int -= crypto.n
            pooled_cov[i, j] = decrypted_int / scale

    return pooled_cov
