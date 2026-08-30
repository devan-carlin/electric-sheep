# Build an OAuth 2.0 / OIDC Provider

Implement a minimal but correct OAuth 2.0 authorization server with OpenID Connect support in a single Python file using FastAPI.

**Required endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/authorize` | GET | Authorization code consent page |
| `/token` | POST | Token issuance (authorization code + client credentials grant) |
| `/userinfo` | GET | UserInfo endpoint (returns claims) |
| `/.well-known/openid-configuration` | GET | OIDC discovery document |
| `/.well-known/jwks.json` | GET | Public JWK set for token verification |

**Required features:**

1. **Authorization code flow with PKCE** — `code_challenge` (S256) and `code_verifier` validation.
2. **JWT access tokens** — RS256 signed, with `iss`, `sub`, `aud`, `exp`, `iat`, `scope` claims.
3. **Refresh token rotation** — each refresh invalidates the previous token; single-use.
4. **Scope validation** — `openid`, `profile`, `email`; reject unknown scopes.
5. **State parameter** — required for `/authorize`, echoed back in callback URL.
6. **Client credential validation** — `client_id` + `client_secret` for `/token` endpoint.
7. **Token introspection** — `/token/introspect` returns active/inactive status.

**Constraints:**

- Single file, no external OAuth libraries (no `authlib`, `oauthlib`).
- Use `cryptography` for RSA key generation and JWT signing.
- In-memory storage for clients, codes, and tokens (no database).
- Include a `/register` endpoint that creates a test client with a random secret.
- Tokens must expire: access token 15 minutes, refresh token 7 days.
- Reject replayed authorization codes.
- Validate `redirect_uri` matches registered URI exactly.

**Deliverable:**

Complete FastAPI application. Include a `curl` example for the full authorization code + PKCE flow at the bottom of the file.
