# Tool Use & Agentic Planning — Simulated Function Calling

**Target Capability:** Multi-step orchestration and structured payload generation.

If you want to use your local LLM for agentic workflows (like AutoGen, LangChain, or Ollama function calling), it needs to know *when* and *how* to trigger tools.

---

## Prompt

```
You are an AI agent with access to the following 4 tools. Each tool is a REST API endpoint that accepts JSON input and returns JSON output.

### Available Tools

1. **get_user_balance(user_id: string)**
   - Returns: {"user_id": string, "balance": number, "currency": "USD"}
   - Fetches the current account balance for a user.

2. **transfer_funds(from_id: string, to_id: string, amount: number)**
   - Returns: {"transaction_id": string, "status": "success" | "failed", "message": string}
   - Transfers funds between two user accounts. Fails if insufficient balance.

3. **send_email(to_email: string, subject: string, body: string)**
   - Returns: {"message_id": string, "status": "sent"}
   - Sends an email notification.

4. **get_user_email(user_id: string)**
   - Returns: {"user_id": string, "email": string}
   - Looks up a user's email address.

### User Request

"Check if User A (user_id: usr_1001) has enough money to send $50 to User B (user_id: usr_2002). If they do, transfer it and email User B a confirmation. If not, email User A about low funds."

### Your Task

Output a sequential JSON execution plan — an array of tool calls in the exact order they should be executed. Each step must include:
- "tool": the function name
- "params": the exact parameters to pass
- "condition": optional, describes when this step should execute (e.g., "if balance >= 50")

Your output must be ONLY valid JSON. No explanation, no markdown, no conversational text.

Example format:
[
  {"step": 1, "tool": "get_user_balance", "params": {"user_id": "usr_1001"}, "condition": null},
  {"step": 2, "tool": "transfer_funds", "params": {"from_id": "usr_1001", "to_id": "usr_2002", "amount": 50}, "condition": "balance >= 50"},
  ...
]
```
