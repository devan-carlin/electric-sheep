# Output: top-32 × 09-auth-security

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
   - Context: Microservice architecture implied (especially for JWT and rate limiting).

2.  **Break Down Each Question & Formulate Answers:**

   *1. Password hashing: bcrypt vs argon2 vs scrypt? Which do you pick and why? What cost factor?*
   - Need to pick one, justify, and specify cost factor.
   - Argon2 is generally recommended by OWASP and modern standards. It's memory-hard, resistant to GPU attacks, and has configurable parameters.
   - Bcrypt is legacy but still widely used. Scrypt is memory-hard but less standardized.
   - Pick: Argon2id.
   - Why: Memory-hard, resistant to hardware acceleration attacks, configurable time/memory parameters, OWASP recommended.
   - Cost factor: ~10-12 iterations, 64MB memory, ~100ms latency (adjustable based on hardware/server load).

   *2. JWT signing: HS256 vs RS256 vs EdDSA? Tradeoffs for a microservice architecture?*
   - Need to pick one, justify tradeoffs for microservices.
   - HS256: symmetric, fast, simple, but requires secret sharing across services. Risk of secret leakage.
   - RS256/EdDSA: asymmetric, public key can be shared, private key kept secure. Better for distributed/microservices. EdDSA is faster and smaller than RSA.
   - Pick: EdDSA (or RS256 if legacy compatibility needed, but EdDSA is modern). I'll go with EdDSA.
   - Why: Asymmetric signing allows public key distribution across microservices without sharing secrets. EdDSA is faster, smaller payload, and resistant to side-channel attacks. Tradeoff: requires key management infrastructure (key rotation, public key distribution), but eliminates secret leakage risk.

   *3. Refresh tokens: How do you implement rotation? How do you detect token theft (session hijacking)?*
   - Need to explain rotation mechanism and theft detection.
   - Rotation: Use a "parent-child

## Metadata
- **Top-k**: 32
- **Elapsed**: 7020ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
