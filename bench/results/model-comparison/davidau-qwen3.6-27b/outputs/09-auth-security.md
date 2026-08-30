# Output: davidau-qwen3.6-27b × 09-auth-security

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
Here's a thinking process that leads to the suggested answers:

1.  **Analyze the Request:**
    *   **Topic:** Auth Security Design Decisions.
    *   **Format:** 1-2 bullet points per question.
    *   **Questions:**
        1.  Password Hashing (bcrypt vs. argon2 vs. scrypt).
        2.  JWT Signing (HS256 vs. RS256 vs. EdDSA).
        3.  Refresh Tokens (Rotation & Theft Detection).
        4.  Account Lockout (Parameters & DoS prevention).
        5.  Token Revocation (Logout all devices).
        6.  Rate Limiting (Storage & Distribution).

2.  **Drafting Answers (Iterative Refinement):**

    *   **Q1: Password Hashing**
        *   *Options:* bcrypt (standard, CPU bound), scrypt (memory bound), argon2 (winner of Password Hashing Competition, memory + CPU bound, side-channel resistant).
        *   *Decision:* Argon2id is the modern gold standard.
        *   *Why:* Resists GPU/ASIC attacks better than bcrypt/scrypt due to memory hardness.
        *   *Cost:* Needs to be tuned to ~250-500ms on target hardware.
        *   *Drafting:* Pick Argon2id for superior resistance against GPU/ASIC cracking via memory hardness. Tune parameters (memory, iterations, parallelism) to target a ~250-500ms verification time on your specific infrastructure.

    *   **Q2: JWT Signing**
        *   *Options:* HS256 (symmetric, fast, shared secret), RS256 (asymmetric, slower, public/private key), EdDSA (asymmetric, faster/smaller than RSA).
        *   *Context:* Microservices.
        *   *Tradeoff:* HS256 requires sharing a secret across all services (blast radius if leaked). RS256/EdDSA allow services to verify with a public key without knowing the signing secret.
        *   *Decision:* RS256 or EdDSA. EdDSA is technically superior (smaller keys/tokens, faster), but RS256 is more ubiquitous. Let's go with RS256 for compatibility or EdDSA for performance

## Metadata
- **Model**: davidau-qwen3.6-27b
- **Elapsed**: 16248ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
- **Gen tok/s**: 31.5
