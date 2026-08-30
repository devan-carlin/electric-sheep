# Output: top-8 × 09-auth-security

## Prompt
# Auth Security — Design Decisions

You're designing an auth service. Answer each in 1-2 bullet points:

1. **Password hashing:** bcrypt vs argon2 vs scrypt? Which do you pick and why? What cost factor?
2. **JWT signing:** HS256 vs RS256 vs EdDSA? Tradeoffs for a microservice architecture?
3. **Refresh tokens:** How do you implement rotation? How do you detect token theft (session hijacking)?
4. **Account lockout:** After N failed attempts, lock for M minutes. What are N and M? How do you prevent lockout DoS?
5. **Token revocation:** JWTs are stateless — how do you support "logout all devices"? Short-lived JWT + refresh token DB check? Token blacklist in Redis?
6. **Rate limiting:** Login endpoint: 5 attempts/min/IP. Registration: 3 attempts/hour/IP. Where do you store counters? Redis? In-memory? What about distributed deployments?

## Response
Here's a thinking process:

1.  **Analyze User Input:**
   - **Topic:** Auth Security Design Decisions
   - **Format:** 6 specific questions, each requiring 1-2 bullet points per answer.
   - **Questions:**
     1. Password hashing: bcrypt vs argon2 vs scrypt? Pick one, why? Cost factor?
     2. JWT signing: HS256 vs RS256 vs EdDSA? Tradeoffs for microservices?
     3. Refresh tokens: Rotation implementation? Detect token theft/session hijacking?
     4. Account lockout: N failed attempts, lock for M minutes. What are N and M? Prevent lockout DoS?
     5. Token revocation: Stateless JWTs vs "logout all devices". Options: short-lived JWT + refresh token DB check? Token blacklist in Redis?
     6. Rate limiting: Login (5/min/IP), Registration (3/hour/IP). Where store counters? Redis vs in-memory? Distributed deployments?
   - **Constraints:** 1-2 bullet points per answer.

2.  **Deconstruct & Draft Answers (Mental Refinement):**

   *Q1: Password hashing*
   - Pick: Argon2id (winner of Password Hashing Competition)
   - Why: Memory-hard, resistant to GPU/ASIC attacks, configurable memory/time/CPU costs
   - Cost: 64MB memory, 3 iterations, 1 thread (or similar modern defaults), tuned to ~100ms on target hardware
   - Format: 1-2 bullets
   - Draft: 
     • **Argon2id** – Winner of the Password Hashing Competition; it’s memory-hard and highly resistant to GPU/ASIC cracking while remaining CPU-efficient for legitimate verification.
     • **Cost:** 64 MB memory, 3 iterations, 1 thread (tuned to ~100ms on target servers), with automatic fallback to lower costs if hardware constraints change.

   *Q2: JWT signing*
   - Pick: EdDSA (Ed25519)
   - Why: Stronger security than RSA/ECDSA, smaller signatures, faster verification, no padding oracle vulnerabilities, ideal for microservices
   - Tradeoffs: Requires asymmetric key management (public key distribution), but avoids shared secret risks of HS256 and key size/per

## Metadata
- **Top-k**: 8
- **Elapsed**: 6149ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
