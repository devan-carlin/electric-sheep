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
   - **Format:** Answer each of 6 questions in 1-2 bullet points
   - **Questions:**
     1. Password hashing: bcrypt vs argon2 vs scrypt? Pick one, why, cost factor?
     2. JWT signing: HS256 vs RS256 vs EdDSA? Tradeoffs for microservices?
     3. Refresh tokens: Rotation implementation? Detect token theft?
     4. Account lockout: N failed attempts, lock M minutes. What N/M? Prevent lockout DoS?
     5. Token revocation: Stateless JWTs vs "logout all"? Options: short-lived + refresh DB check, blacklist in Redis?
     6. Rate limiting: Login 5/min/IP, Reg 3/hour/IP. Where store counters? Redis vs in-memory? Distributed deployments?
   - **Constraint:** 1-2 bullet points per question.

2.  **Evaluate Each Question & Formulate Answers (Mental Draft):**
   - *Q1: Password hashing.* Modern consensus: Argon2id is the winner (OWASP, NIST). It's memory-hard, resistant to GPU/ASIC attacks. Pick Argon2id. Cost: time=2, memory=64MB, parallelism=4 (or similar OWASP-recommended defaults). bcrypt is legacy, scrypt is outdated.
   - *Q2: JWT signing.* HS256 needs shared secret (bad for microservices). RS256/EdDSA use asymmetric keys. EdDSA (Ed25519) is faster, smaller, and cryptographically stronger than RSA. Tradeoff: EdDSA requires key distribution via JWKS, but no secret sharing. Pick EdDSA.
   - *Q3: Refresh tokens.* Rotation: issue new refresh token on use, invalidate old one atomically. Detect theft: compare token fingerprint (hash of token + IP/UserAgent) or use device binding; flag mismatched context.
   - *Q4: Account lockout.* N=5, M=15-30 mins is standard. Prevent DoS: lock by username, not IP; use exponential backoff instead of hard lock; or rely on CAPTCHA/rate limiting rather than account lockout (modern auth prefers sliding windows

## Metadata
- **Top-k**: 8
- **Elapsed**: 6091ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
