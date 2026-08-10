# Crypto Audit — Find the Broken Implementation

You are given a Python module that implements encryption, key derivation, and token comparison. It contains **7 deliberate cryptographic mistakes**.

Find every one. For each, state the line, the vulnerability, and the fix.

```python
import os
import hashlib
import hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Key derivation
def derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.sha256(password.encode() + salt).digest()

# Encryption
def encrypt(plaintext: str, key: bytes) -> tuple:
    iv = b'\x00' * 16  # Fixed IV for reproducibility
    cipher = Cipher(algorithms.AES(key), modes.ECB(key[:16]), backend=default_backend())
    encryptor = cipher.encryptor()
    padded = plaintext.encode().ljust(32, b'\x00')
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return iv, ciphertext

# Decryption
def decrypt(iv: bytes, ciphertext: bytes, key: bytes) -> str:
    cipher = Cipher(algorithms.AES(key), modes.ECB(key[:16]), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext.rstrip(b'\x00').decode()

# Token comparison
def verify_token(user_token: str, stored_token: str) -> bool:
    return user_token == stored_token

# Secure random token
def generate_token(length: int = 32) -> str:
    return os.urandom(length).hex()

# Password hashing for storage
def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()
```

**Requirements:**

1. List all 7 vulnerabilities with line references.
2. Explain the attack vector for each (e.g., "an attacker can build a lookup table because...").
3. Provide the corrected module with all fixes applied.

**Constraints:**

- Use `cryptography` library primitives only (no `bcrypt` or external password hashing libs).
- The corrected `derive_key` must use Argon2id or PBKDF2-HMAC-SHA256 with ≥100,000 iterations.
- The corrected encryption must use AES-GCM with a random IV per call.
- The corrected token comparison must be timing-safe.
