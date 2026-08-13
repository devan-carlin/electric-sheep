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
Thinking Process:

1.  **Deconstruct the Prompt:**
    *   Goal: Design an auth service.
    *   Constraint: Answer each in 1-2 bullet points.
    *   Topics:
        1.  Password hashing (bcrypt vs argon2 vs scrypt, pick, why, cost).
        2.  JWT signing (HS256 vs RS256 vs EdDSA, tradeoffs microservice).
        3.  Refresh tokens (rotation, theft detection).
        4.  Account lockout (N, M, DoS prevention).
        5.  Token revocation (logout all, stateless JWT vs stateful check).
        6.  Rate limiting (Login/Reg limits, storage, distributed).

2.  **Draft Answers per Topic (Mental Check against constraints):**
    *   *Topic 1: Hashing.*
        *   Pick: Argon2id. Why: Memory-hard, best against GPU/ASIC attacks, winner of Password Hashing Competition.
        *   Cost: Time cost 2-3 seconds, memory 64MB.
        *   *Bullet 1:* Argon2id; memory-hard algorithm wins PHC, superior GPU resistance.
        *   *Bullet 2:* Time cost 2-3s, memory 64MB; balances security vs latency.
        *   *Result:* 2 bullets. Good.

    *   *Topic 2: JWT Signing.*
        *   HS256: Shared secret, simple, but key management hard in microservices.
        *   RS256: Asymmetric, good for key rotation, but larger tokens, CPU heavy.
        *   EdDSA: Fast, small, but less standard support? Actually EdDSA is great.
        *   Tradeoff Microservice: RS256 allows independent key management per service, no shared secret leakage. Or EdDSA for speed?
        *   *Decision:* RS256 is the classic answer for microservices due to key isolation. EdDSA is modern but RS256 is the "tradeoff" answer usually expected.
        *   *Bullet 1:* RS256; asymmetric keys isolate signing secrets per service, enable independent key rotation without re-issuing tokens.
        *   *Bullet 2:* HS256 avoids

## Metadata
- **Top-k**: 16
- **Elapsed**: 6645ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
