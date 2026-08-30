# Output: top-64 × 09-auth-security

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

1.  **Analyze the User Request:**
   - The user is asking for design decisions for an auth service.
   - Each question needs to be answered in 1-2 bullet points.
   - Topics: Password hashing, JWT signing, Refresh tokens, Account lockout, Token revocation, Rate limiting.
   - Context: "microservice architecture" implied in JWT question, distributed deployments implied in rate limiting.

2.  **Break down each question & formulate concise answers (1-2 bullets):**

   *1. Password hashing:*
   - Options: bcrypt, argon2, scrypt.
   - Pick: Argon2 (modern standard, memory-hard, configurable).
   - Why: Better resistance to GPU attacks, configurable cost/memory parameters, widely adopted by industry standards (OWASP, RFC).
   - Cost factor: Adjustable via `cost` parameter (e.g., 12-14), balances security vs latency.

   *2. JWT signing:*
   - Options: HS256, RS256, EdDSA.
   - Tradeoffs for microservice: HS256 is simplest/shared secret, but requires secret distribution across services. RS256/EdDSA is asymmetric, better for distributed/microservice (public key shared, private key kept in auth service only, easier to revoke/verify across boundaries).
   - Pick: EdDSA (or RS256) for microservice architecture.
   - Why: Asymmetric signing allows auth service to hold private key, other services verify with public key. Supports token revocation without secret leakage. EdDSA is faster, smaller payload.

   *3. Refresh tokens:*
   - Rotation: Implement via "refresh token rotation" (new refresh token issued upon refresh, old one invalidated).
   - Detection theft: Use session fingerprinting (IP, device, timestamp, user-agent) + refresh token rotation + short-lived refresh tokens. If mismatch detected, invalidate.
   - Bullet 1: Rotation: Issu new refresh token upon refresh, invalidate old one.
   - Bullet 2: Theft detection: Session fingerprinting + rotation + short-lived refresh tokens; mismatch triggers invalidation.

   *4. Account lockout:*
   - N & M: N=5, M=15 minutes (standard industry practice).
   - Prevent DoS: Use IP-based rate limiting

## Metadata
- **Top-k**: 64
- **Elapsed**: 8332ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
