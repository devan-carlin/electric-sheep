# Strict Output Compliance — Messy Log to JSON Parser

**Target Capability:** Deterministic formatting, edge-case handling, zero conversational fluff.

Small and medium local models frequently fail when asked to adhere *strictly* to a JSON schema without adding explanatory text like *"Here is your JSON:"* or leaking invalid syntax when encountering weird input data.

---

## Prompt

```
You are a data extraction engine. Your ONLY output must be valid, raw JSON. Do NOT include any explanatory text, markdown formatting, or conversational filler before or after the JSON.

Extract the following information from the messy log data below into this exact schema:

{
  "entries": [
    {
      "timestamp": "ISO 8601 UTC string",
      "level": "ERROR | WARN | INFO | DEBUG",
      "source": "string or null",
      "message": "string",
      "user_id": "string or null",
      "has_pii": true | false
    }
  ]
}

Rules:
- Standardize all timestamps to ISO 8601 UTC.
- If a field is missing, use null (not an empty string).
- If a message contains double quotes, escape them properly.
- Set has_pii to true if the message contains an email address, phone number, or name.

Here is the raw log data:

[2025-03-14 08:23:11] INFO auth-service: User login successful for user_id=USR-4492
[Mar 14 08:25:03 UTC] WARN payment-gw: Retry attempt 2 for txn TXN-881 — timeout after 30s
2025-03-14T08:27:45Z ERROR db-primary: Connection pool exhausted (active=50, max=50)
[14/03/2025 08:30:12] INFO email-svc: Sent welcome email to dc@example.com from user "New User"
ERROR 08:31:55 unknown: Segfault in module worker_3 — core dumped
[2025-03-14 08:33:01 UTC+5] WARN auth-service: Failed login attempt for user_id=USR-1103 (IP: 192.168.1.42)
Mar 14 08:35:22 INFO cdn-edge: Cache miss for /assets/logo.png — fetching origin
2025-03-14T08:37:00Z ERROR payment-gw: Refund failed for TXN-881 — "insufficient reserve" — user_id=USR-4492
[08:39:11] DEBUG worker_3: Heartbeat OK (uptime: 4h 12m)
2025-03-14 08:41:33 UTC WARN auth-service: Password reset requested for jane.doe@corp.net
```
