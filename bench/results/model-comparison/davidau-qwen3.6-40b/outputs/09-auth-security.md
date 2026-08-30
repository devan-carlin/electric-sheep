# Output: davidau-qwen3.6-40b × 09-auth-security

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
This is a technical interview-style question about authentication security design. I need to give concise, practical answers that show real-world experience and understanding of tradeoffs.

Let me think through each:

1. **Password hashing:** 
- Argon2 is the winner (won Password Hashing Competition 2015), memory-hard, resistant to GPU/ASIC attacks
- bcrypt is still widely supported but older, less memory-hard
- scrypt is good but Argon2 supersedes it
- Cost factor: depends on algorithm - for Argon2id, aim for ~500ms hash time, typically memory=64MB, iterations=3, parallelism=1
- Practical note: bcrypt cost 12-14 is still common due to ubiquity

2. **JWT signing:**
- HS256: fast, symmetric, shared secret - good for single org, no key distribution needed
- RS256: asymmetric, slower, good for cross-org, public key distribution
- EdDSA: asymmetric, fast, small keys/signatures, modern choice
- For microservices: HS256 if all services trust each other and share secret; EdDSA if need asymmetric verification or cross-org
- Tradeoff: HS256 = speed but key sharing; asymmetric = key distribution overhead but better isolation

3. **Refresh tokens:**
- Rotation: issue new refresh token with each use, invalidate old one (or allow short grace period)
- Detection: track token fingerprints/hashes, detect concurrent use from different locations/tokens
- If old token used after new issued → likely theft, revoke all tokens for user
- Store token hash + metadata (IP, user-agent) in DB

4. **Account lockout:**
- N=5, M=15-30 minutes is common
- But lockout DoS is real - attackers can lock out legitimate users
- Better: exponential backoff instead of hard lockout
- Or lockout only after suspicious patterns (multiple IPs, etc.)
- Some orgs skip lockout entirely and use other signals
- Store attempts in Redis with TTL

5. **Token revocation:**
- JWTs are stateless but you can add stateful checks
- Short-lived JWT (5-15 min) + refresh token checked against DB on sensitive ops
- Redis blacklist for JWTs: store jti with TTL = remaining lifetime
- Better: version number per

## Metadata
- **Model**: davidau-qwen3.6-40b
- **Elapsed**: 24119ms
- **Prompt tokens**: 225
- **Completion tokens**: 512
- **Total tokens**: 737
- **Gen tok/s**: 21.2
