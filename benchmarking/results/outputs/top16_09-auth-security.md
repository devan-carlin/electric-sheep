# Output: top-16 × 09-auth-security

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
Here's a thinking thinking sequence

1.  **Deconstruct and Analyze each question:**
    *   *Goal:* Answer each question in 1-2 bullet points.
    *   *Context:* Designing an auth service, microservice architecture implied (especially for JWT and rate limiting).

    *   *Question 1: Password hashing.*
        *   *Options:* bcrypt, argon2, scrypt.
        *   *Decision:* Argon2 is generally the current gold standard (winner of Password Hashing Competition). It's resistant to GPU attacks and memory-hard.
        *   *Why:* Security, memory-hardness.
        *   *Cost factor:* Needs to be high enough to slow down attackers but low enough to not impact UX. Usually ~100ms. Argon2id is preferred.
        *   *Drafting:* Argon2id. Memory-hard, resistant to GPU attacks. Cost: ~100ms latency (e.g., `time_cost=3`, `memory_cost=65536`).

    *   *Question 2: JWT signing.*
        *   *Options:* HS256, RS256, EdDSA.
        *   *Context:* Microservice architecture.
        *   *Tradeoffs:* HS256 requires shared secret (hard to distribute securely across services). RS256 uses asymmetric keys (public key can be shared, private kept secure), good for distributed systems. EdDSA is newer, faster, smaller signatures, but RS256 is more widely supported in existing libraries/ecosystems.
        *   *Decision:* RS256 is the safest bet for microservices due to key distribution. EdDSA is a strong contender for performance, but RS256 is the standard answer for "microservice architecture" tradeoffs regarding key management.
        *   *Drafting:* RS256. Asymmetric keys allow public key distribution to all microservices without sharing secrets; private key stays secure in auth service. EdDSA is faster/smaller but RS256 offers better ecosystem compatibility and key isolation.

    *   *Question 3: Refresh tokens.*
        *   *Sub-questions:* Rotation? Detection of theft?
        *   *Rotation:* Revoke old, issue new. Store in DB.
        *   *Detection:* Device fingerprinting, IP tracking, or "one active session

## Metadata
- **Top-k**: 16
- **Elapsed**: 6489ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
