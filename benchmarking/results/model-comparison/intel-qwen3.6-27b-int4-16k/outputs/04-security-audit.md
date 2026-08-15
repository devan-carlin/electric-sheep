# Output: intel-qwen3.6-27b-int4-16k × 04-security-audit

## Prompt
# Vulnerability Audit — Find the Bugs

This Flask app has 5 security vulnerabilities. List each one (type + line + attack vector):

```python
import os, sqlite3, hashlib
from flask import Flask, request, jsonify, send_file
import jwt

app = Flask(__name__)
app.secret_key = 'dev-secret-key-12345'
JWT_SECRET = 'jwt-secret-production-key'

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username, password, email = data.get('username',''), data.get('password',''), data.get('email','')
    password_hash = hashlib.md5(password.encode()).hexdigest()  # LINE A
    db = sqlite3.connect('users.db')
    db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")  # LINE B
    db.commit(); db.close()
    return jsonify({'message': 'User created'}), 201

@app.route('/api/profile', methods=['GET'])
def get_profile():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    db = sqlite3.connect('users.db')
    user = db.execute("SELECT * FROM users WHERE id = ?", (payload['user_id'],)).fetchone()
    db.close()
    return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})  # LINE C

@app.route('/api/users/search', methods=['GET'])
def search_users():
    q = request.args.get('q', '')
    role = request.args.get('role', '')
    sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)  # LINE D
    if role: sql += f" AND role = '{role}'"  # LINE E
    db = sqlite3.connect('users.db')
    return jsonify([dict(r) for r in db.execute(sql).fetchall()])

@app.route('/api/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join('/tmp/uploads', filename))  # LINE F
```

For each vulnerability, give: (1) CVE category, (2) which line, (3) one-sentence attack vector.
    token = auth.replace('Bearer ', '')

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except:
        return jsonify({'error': 'Unauthorized'}), 401

    # Check admin role
    if payload.get('role') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    db = get_db()
    users = db.execute("SELECT id, username, email, role, created_at, last_login FROM users").fetchall()
    db.close()

    return jsonify([{
        'id': u['id'],
        'username': u['username'],
        'email': u['email'],
        'role': u['role'],
        'created_at': u['created_at']
    } for u in users])

@app.route('/api/debug/stack', methods=['GET'])
def debug_stack():
    """Debug endpoint for development — shows current stack trace"""
    import traceback
    return jsonify({'stack': traceback.format_stack()})

@app.route('/api/error-test', methods=['GET'])
def error_test():
    """Intentionally raises an error for testing"""
    raise ValueError("This is a test error")

@app.errorhandler(500)
def handle_500(error):
    return jsonify({
        'error': 'Internal server error',
        'details': str(error),
        'type': type(error).__name__,
        'module': type(error).__module__
    }), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Requirements:**

1. Identify all five vulnerabilities (label each by category)
2. For each vulnerability:
   - Quote the exact code
   - Describe the attack vector (what an attacker would do)
   - Rate severity (Critical/High/Medium/Low)
3. Provide a fully patched version of the application
4. Include a security checklist of additional improvements (beyond the 5 bugs)

**Constraints:**

- Use `werkzeug.security` for password hashing (bcrypt or scrypt)
- Use parameterized queries everywhere
- Validate and sanitize file uploads (extension whitelist, safe filename)
- Remove or protect debug endpoints
- Use environment variables for secrets

Produce the analysis and patched code. No placeholders, no TODOs.

## Reasoning
Here's a thinking process:

1.  **Understand User Request:**
   - The user provides a Flask application code snippet with 5 security vulnerabilities.
   - They want me to identify each vulnerability with: (1) CVE category/type, (2) line reference, (3) one-sentence attack vector.
   - They also want: exact code quote, attack vector description, severity rating for each.
   - Then, provide a fully patched version of the application.
   - Finally, include a security checklist of additional improvements.
   - Constraints: Use `werkzeug.security` for password hashing, parameterized queries everywhere, validate/sanitize file uploads, remove/protect debug endpoints, use environment variables for secrets.
   - No placeholders, no TODOs.

2.  **Analyze the Provided Code for Vulnerabilities:**
   Let's go through the code line by line to find the 5 vulnerabilities.

   *Code Snippet 1:*
   ```python
   import os, sqlite3, hashlib
   from flask import Flask, request, jsonify, send_file
   import jwt

   app = Flask(__name__)
   app.secret_key = 'dev-secret-key-12345'
   JWT_SECRET = 'jwt-secret-production-key'
   ```
   - Hardcoded secrets (`app.secret_key`, `JWT_SECRET`). This is a vulnerability (CWE-798: Use of Hard-coded Credentials). But the prompt says "5 security vulnerabilities" and specifically points to lines A, B, C, D, E, F in the first block. Let's check those.

   *Line A:* `password_hash = hashlib.md5(password.encode()).hexdigest()`
   - Weak cryptographic hash for passwords. MD5 is broken and unsuitable for password storage. (CWE-916: Use of Password Hash With Insufficient Computational Effort / CWE-328: Use of Weak Hash)

   *Line B:* `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
   - SQL Injection via string formatting. (CWE-89: SQL Injection)

   *Line C:* `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
   - Information Disclosure / Sensitive Data Exposure. Returning password hash to the client. (CWE-200: Exposure of Sensitive Information to an Unauthorized Actor)

   *Line D:* `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)`
   - SQL Injection via string formatting. (CWE-89: SQL Injection)

   *Line E:* `if role: sql += f" AND role = '{role}'"`
   - SQL Injection via string formatting. (CWE-89: SQL Injection) - Actually, D and E are both SQLi in the same function, but they might count as one or two. The prompt says 5 vulnerabilities total. Let's count: A (Weak Hash), B (SQLi), C (Info Disclosure), D/E (SQLi), F (Path Traversal). That's 5 distinct categories/instances. Wait, D and E are both SQLi in the same route. I'll group them or treat D as the main SQLi for that route. Actually, the prompt says "5 security vulnerabilities". I'll list:
     1. Weak Password Hashing (Line A)
     2. SQL Injection (Line B)
     3. Sensitive Data Exposure (Line C)
     4. SQL Injection (Line D/E)
     5. Path Traversal (Line F)
     Plus hardcoded secrets, but I'll stick to the 5 explicitly marked or obvious ones. The prompt says "This Flask app has 5 security vulnerabilities." I'll map them to the lines provided.

   *Line F:* `return send_file(os.path.join('/tmp/uploads', filename))`
   - Path Traversal / Directory Traversal. `filename` comes from URL path, not sanitized. (CWE-22: Improper Limitation of a Pathname to a Restricted Directory)

   Let's verify the exact lines and categories:
   1. **Weak Cryptographic Hash** (Line A) - CWE-328/CWE-916
   2. **SQL Injection** (Line B) - CWE-89
   3. **Sensitive Data Exposure** (Line C) - CWE-200
   4. **SQL Injection** (Line D/E) - CWE-89
   5. **Path Traversal** (Line F) - CWE-22

   Wait, the prompt also includes a second block of code with `/api/admin/users`, `/api/debug/stack`, `/api/error-test`, and error handler. The prompt says "This Flask app has 5 security vulnerabilities." It might be referring to the first block's marked lines, or maybe the debug endpoint is one? But the prompt explicitly says "List each one (type + line + attack vector)" and gives lines A-F. I'll stick to A, B, C, D, F as the 5. Actually, D and E are both SQLi, but I'll treat D as the primary SQLi for that route. Or maybe the 5 are: Weak Hash, SQLi (B), Info Disclosure, SQLi (D/E), Path Traversal. That's 5. I'll list them clearly.

   Let's draft the vulnerability analysis:
   1. **Weak Password Hashing** (Line A)
      - Code: `password_hash = hashlib.md5(password.encode()).hexdigest()`
      - Attack Vector: Attacker intercepts or steals the database and uses rainbow tables or brute-force to instantly crack MD5 hashes, recovering plaintext passwords.
      - Severity: Critical
   2. **SQL Injection** (Line B)
      - Code: `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
      - Attack Vector: Attacker supplies a malicious username/email containing SQL metacharacters (e.g., `admin'--`) to manipulate the INSERT statement, bypassing authentication or corrupting the database.
      - Severity: Critical
   3. **Sensitive Data Exposure** (Line C)
      - Code: `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
      - Attack Vector: Attacker authenticates with a valid token and calls the profile endpoint to retrieve the password hash, which can then be cracked offline to compromise the account.
      - Severity: High
   4. **SQL Injection** (Line D/E)
      - Code: `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)` / `sql += f" AND role = '{role}'"`
      - Attack Vector: Attacker crafts the `q` or `role` query parameter with SQL syntax (e.g., `q=%' OR 1=1--`) to dump the entire users table or extract sensitive data.
      - Severity: Critical
   5. **Path Traversal** (Line F)
      - Code: `return send_file(os.path.join('/tmp/uploads', filename))`
      - Attack Vector: Attacker requests `/api/download/../../../etc/passwd` to traverse directories and read arbitrary files from the server's filesystem.
      - Severity: Critical

   Wait, the prompt says "CVE category". I'll use CWE/CVE categories like CWE-328, CWE-89, CWE-200, CWE-22. I'll format as requested.

   Now, I need to produce a fully patched version. Constraints:
   - Use `werkzeug.security` for password hashing (bcrypt or scrypt)
   - Use parameterized queries everywhere
   - Validate and sanitize file uploads (extension whitelist, safe filename)
   - Remove or protect debug endpoints
   - Use environment variables for secrets
   - No placeholders, no TODOs.

   I'll rewrite the entire app properly. I'll need to handle the database setup, JWT, routes, etc. I'll make it complete and functional.

   Structure of patched code:
   - Imports: `os`, `sqlite3`, `hashlib` (maybe not needed), `werkzeug.security`, `flask`, `jwt`, `secrets`, `re`, `os.path`
   - Config: `app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))`, `JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))`
   - DB helper: `get_db()` function with parameterized queries.
   - `/api/register`: Use `werkzeug.security.generate_password_hash(password, method='scrypt')`, parameterized INSERT.
   - `/api/profile`: JWT decode with proper error handling, parameterized SELECT, exclude password_hash from response.
   - `/api/users/search`: Parameterized LIKE query, sanitize/validate inputs.
   - `/api/download/<path:filename>`: Validate filename, prevent traversal, check extension whitelist, use `os.path.basename` or `pathlib`, safe join.
   - Remove `/api/debug/stack` and `/api/error-test`.
   - Error handler: Don't leak details.
   - Add `init_db()` for completeness.

   Let's draft the patched code carefully.

   ```python
   import os
   import sqlite3
   import secrets
   import re
   from flask import Flask, request, jsonify, send_file
   from werkzeug.security import generate_password_hash, check_password_hash
   import jwt

   app = Flask(__name__)
   app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
   JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))

   UPLOAD_DIR = '/tmp/uploads'
   ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt'}

   def get_db():
       db = sqlite3.connect('users.db')
       db.row_factory = sqlite3.Row
       return db

   def init_db():
       db = get_db()
       db.execute('''CREATE TABLE IF NOT EXISTS users (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           username TEXT UNIQUE NOT NULL,
           password_hash TEXT NOT NULL,
           email TEXT,
           role TEXT DEFAULT 'user'
       )''')
       db.commit()
       db.close()

   @app.route('/api/register', methods=['POST'])
   def register():
       data = request.get_json()
       if not data:
           return jsonify({'error': 'Invalid JSON'}), 400

       username = data.get('username', '').strip()
       password = data.get('password', '')
       email = data.get('email', '').strip()

       if not username or not password:
           return jsonify({'error': 'Username and password are required'}), 400

       password_hash = generate_password_hash(password, method='scrypt')

       db = get_db()
       try:
           db.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                      (username, password_hash, email))
           db.commit()
       except sqlite3.IntegrityError:
           return jsonify({'error': 'Username already exists'}), 409
       finally:
           db.close()

       return jsonify({'message': 'User created'}), 201

   @app.route('/api/profile', methods=['GET'])
   def get_profile():
       auth_header = request.headers.get('Authorization', '')
       if not auth_header.startswith('Bearer '):
           return jsonify({'error': 'Unauthorized'}), 401

       token = auth_header.replace('Bearer ', '')
       try:
           payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
       except jwt.InvalidTokenError:
           return jsonify({'error': 'Unauthorized'}), 401

       db = get_db()
       user = db.execute("SELECT id, username, email, role FROM users WHERE id = ?",
                         (payload.get('user_id'),)).fetchone()
       db.close()

       if not user:
           return jsonify({'error': 'User not found'}), 404

       return jsonify({
           'id': user['id'],
           'username': user['username'],
           'email': user['email'],
           'role': user['role']
       })

   @app.route('/api/users/search', methods=['GET'])
   def search_users():
       q = request.args.get('q', '').strip()
       role = request.args.get('role', '').strip()

       # Basic input validation to prevent obvious injection attempts
       if not re.match(r'^[a-zA-Z0-9_\- ]+$', q):
           return jsonify({'error': 'Invalid search query'}), 400
       if role and not re.match(r'^[a-zA-Z0-9_\- ]+$', role):
           return jsonify({'error': 'Invalid role parameter'}), 400

       db = get_db()
       if role:
           users = db.execute("SELECT id, username, email FROM users WHERE username LIKE ? AND role = ?",
                              (f'%{q}%', role)).fetchall()
       else:
           users = db.execute("SELECT id, username, email FROM users WHERE username LIKE ?",
                              (f'%{q}%',)).fetchall()
       db.close()

       return jsonify([dict(u) for u in users])

   @app.route('/api/download/<path:filename>')
   def download_file(filename):
       # Prevent path traversal
       safe_filename = os.path.basename(filename)
       if not safe_filename or safe_filename.startswith('.'):
           return jsonify({'error': 'Invalid filename'}), 400

       # Validate extension
       ext = safe_filename.rsplit('.', 1)[-1].lower()
       if ext not in ALLOWED_EXTENSIONS:
           return jsonify({'error': 'File type not allowed'}), 400

       file_path = os.path.join(UPLOAD_DIR, safe_filename)

       # Ensure the resolved path is within UPLOAD_DIR
       real_path = os.path.realpath(file_path)
       if not real_path.startswith(os.path.realpath(UPLOAD_DIR)):
           return jsonify({'error': 'Access denied'}), 403

       if not os.path.isfile(real_path):
           return jsonify({'error': 'File not found'}), 404

       return send_file(real_path, as_attachment=True)

   @app.errorhandler(500)
   def handle_500(error):
       return jsonify({'error': 'Internal server error'}), 500

   if __name__ == '__main__':
       init_db()
       app.run(host='0.0.0.0', port=5000)
   ```

   Wait, I need to make sure I address all constraints:
   - `werkzeug.security` for password hashing: Used `generate_password_hash(password, method='scrypt')`
   - Parameterized queries everywhere: Used `?` placeholders in all SQL.
   - Validate and sanitize file uploads: Used `os.path.basename`, extension whitelist, `os.path.realpath` check.
   - Remove or protect debug endpoints: Removed `/api/debug/stack` and `/api/error-test`.
   - Use environment variables for secrets: Used `os.environ.get(...)` with fallbacks.
   - No placeholders, no TODOs: Checked.

   I should also add a proper JWT creation endpoint or at least note that the profile expects a token with `user_id`. The original code didn't show login, but that's fine. I'll keep it focused on the provided routes.

   Let's refine the vulnerability analysis to match the prompt's exact request:
   "For each vulnerability, give: (1) CVE category, (2) which line, (3) one-sentence attack vector."
   Also: "Quote the exact code", "Describe the attack vector", "Rate severity".

   I'll structure it clearly.

   Vulnerability 1: Weak Password Hashing
   - Category: CWE-328 / CWE-916 (Use of Weak Hash / Insufficient Computational Effort)
   - Line: A
   - Code: `password_hash = hashlib.md5(password.encode()).hexdigest()`
   - Attack Vector: An attacker steals the database and uses rainbow tables or GPU-accelerated brute-force to instantly recover plaintext passwords from the unsalted MD5 hashes.
   - Severity: Critical

   Vulnerability 2: SQL Injection (Registration)
   - Category: CWE-89 (SQL Injection)
   - Line: B
   - Code: `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
   - Attack Vector: An attacker supplies a username containing SQL metacharacters (e.g., `admin'--`) to manipulate the INSERT statement, bypassing validation or corrupting the database schema.
   - Severity: Critical

   Vulnerability 3: Sensitive Data Exposure
   - Category: CWE-200 (Exposure of Sensitive Information)
   - Line: C
   - Code: `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
   - Attack Vector: An attacker authenticates with a valid JWT and calls the profile endpoint to retrieve the password hash, which is then cracked offline to compromise the account.
   - Severity: High

   Vulnerability 4: SQL Injection (Search)
   - Category: CWE-89 (SQL Injection)
   - Line: D/E
   - Code: `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)` / `sql += f" AND role = '{role}'"`
   - Attack Vector: An attacker crafts the `q` or `role` query parameters with SQL syntax (e.g., `q=%' OR 1=1--`) to dump the entire users table or extract sensitive data.
   - Severity: Critical

   Vulnerability 5: Path Traversal
   - Category: CWE-22 (Improper Limitation of a Pathname to a Restricted Directory)
   - Line: F
   - Code: `return send_file(os.path.join('/tmp/uploads', filename))`
   - Attack Vector: An attacker requests `/api/download/../../../etc/passwd` to traverse directories and read arbitrary sensitive files from the server's filesystem.
   - Severity: Critical

   Now, check the patched code against constraints. I'll make sure it's complete and production-ready style. I'll add comments where necessary but keep it clean. I'll also ensure the `init_db` and `get_db` are robust. I'll use `pathlib` or `os.path` consistently. I'll make sure the JWT decode handles errors properly. I'll remove the debug endpoints as requested.

   One thing: The prompt says "Use `werkzeug.security` for password hashing (bcrypt or scrypt)". I used scrypt. That's fine.
   "Validate and sanitize file uploads (extension whitelist, safe filename)" - I did.
   "Remove or protect debug endpoints" - I removed them.
   "Use environment variables for secrets" - I did.
   "Parameterized queries everywhere" - I did.

   I should also add a login route to make the JWT flow complete? The prompt doesn't require it, but it's good practice. I'll stick to the provided routes to avoid scope creep, but I'll make sure the code is self-contained. I'll add a simple `/api/login` route to generate tokens, otherwise the profile endpoint can't be tested. Actually, the prompt says "fully patched version of the application", so I'll include a basic login route for completeness, or just note that JWT generation is assumed. I'll add a minimal login route to make it functional.

   Let's draft the final response carefully. I'll structure it exactly as requested.
   I'll double-check the line references. The prompt has:
   LINE A: `password_hash = hashlib.md5(password.encode()).hexdigest()`
   LINE B: `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
   LINE C: `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
   LINE D: `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)`
   LINE E: `if role: sql += f" AND role = '{role}'"`
   LINE F: `return send_file(os.path.join('/tmp/uploads', filename))`

   All match. I'll present the analysis first, then the patched code, then the checklist.

   Checklist items (beyond the 5 bugs):
   - Rate limiting on auth/register endpoints
   - HTTPS enforcement / HSTS
   - Content Security Policy headers
   - Proper JWT expiration & rotation
   - Input validation/sanitization for all endpoints
   - Logging without sensitive data
   - Dependency vulnerability scanning
   - Database connection pooling / context managers
   - CORS configuration
   - Health check endpoint

   I'll format it neatly.

   One minor thing: The prompt says "CVE category". I'll use CWE/CVE mapping like "CWE-89 / CVE-2023-XXXX (SQL Injection)" but since specific CVEs aren't assigned to generic flaws, I'll use the standard CWE category which is the industry standard for vulnerability classification, and note it as requested. I'll phrase it as "CWE-89 (SQL Injection)" etc.

   Let's write it out. 
   I'll make sure the patched code is complete, runnable, and follows all constraints. I'll use `os.environ.get` with secure fallbacks. I'll use `werkzeug.security`. I'll parameterize all SQL. I'll sanitize file paths. I'll remove debug endpoints. I'll add a security checklist.

   All steps verified. Proceeding. 
   Output matches the response. 
   Self-Correction/Refinement during thought:
   - The prompt says "CVE category". I'll explicitly state the CWE/CVE category as requested.
   - I'll ensure the patched code doesn't have any `TODO` or placeholders.
   - I'll make sure the file download validation is robust: `os.path.realpath` check, extension whitelist, basename extraction.
   - I'll add a simple `/api/login` route to make the JWT flow work, as it's necessary for a "fully patched version".
   - I'll ensure the error handler doesn't leak stack traces.
   - I'll use `sqlite3.Row` for dict-like access.
   - I'll add proper imports.

   Everything looks solid. I'll generate the response. 
   Structure:
   1. Vulnerability Analysis (5 items)
   2. Fully Patched Code
   3. Security Checklist
   All constraints met. No fluff. Direct and professional.✅
   Output matches the final response. 
   Proceeds. 
   [Output Generation] -> *See response.* 
   Self-Correction/Verification during generation prep:
   - Check line references: The prompt explicitly marks A, B, C, D, E, F. I'll reference them exactly.
   - Check constraints: `werkzeug.security` used? Yes. Parameterized queries? Yes. File validation? Yes. Debug endpoints removed? Yes. Env vars for secrets? Yes.
   - Check format: (1) CVE category, (2) line, (3) one-sentence attack vector. Then quote, describe, rate severity. I'll combine these as requested.
   - All good.
   Output matches response. 
   [Done.] 
   *Self-Correction/Note during thought*: The prompt says "CVE category". I will use the standard CWE/CVE classification format (e.g., CWE-89 / SQL Injection) since CVEs are specific instances, but I'll label it as requested. I'll make sure the format strictly follows: `(1) CVE category, (2) which line, (3) one-sentence attack vector.` then the additional details.

   I'll produce it now. 
   [Output Generation] -> *Proceeds.* 
   *(Done.)*

## Response


### 🔍 Vulnerability Analysis

| # | CVE/CWE Category | Line | One-Sentence Attack Vector |
|---|------------------|------|----------------------------|
| 1 | **CWE-328 / CWE-916** (Weak Password Hash) | A | Attacker steals the database and uses rainbow tables or GPU brute-force to instantly recover plaintext passwords from unsalted MD5 hashes. |
| 2 | **CWE-89** (SQL Injection) | B | Attacker supplies a username containing SQL metacharacters (e.g., `admin'--`) to manipulate the `INSERT` statement, bypassing validation or corrupting the database. |
| 3 | **CWE-200** (Sensitive Data Exposure) | C | Attacker authenticates with a valid JWT and calls the profile endpoint to retrieve the password hash, which is then cracked offline to compromise the account. |
| 4 | **CWE-89** (SQL Injection) | D/E | Attacker crafts the `q` or `role` query parameters with SQL syntax (e.g., `q=%' OR 1=1--`) to dump the entire users table or extract sensitive data. |
| 5 | **CWE-22** (Path Traversal) | F | Attacker requests `/api/download/../../../etc/passwd` to traverse directories and read arbitrary sensitive files from the server's filesystem. |

**Detailed Breakdown:**
1. **Weak Password Hashing** (Line A)
   - *Code:* `password_hash = hashlib.md5(password.encode()).hexdigest()`
   - *Attack Vector:* Database breach → offline hash cracking → credential compromise.
   - *Severity:* Critical
2. **SQL Injection** (Line B)
   - *Code:* `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
   - *Attack Vector:* Malicious input in registration payload → database manipulation or authentication bypass.
   - *Severity:* Critical
3. **Sensitive Data Exposure** (Line C)
   - *Code:* `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
   - *Attack Vector:* Valid token → profile endpoint → hash exfiltration → offline cracking.
   - *Severity:* High
4. **SQL Injection** (Line D/E)
   - *Code:* `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)` / `sql += f" AND role = '{role}'"`
   - *Attack Vector:* Crafted query parameters → full table dump or data exfiltration.
   - *Severity:* Critical
5. **Path Traversal** (Line F)
   - *Code:* `return send_file(os.path.join('/tmp/uploads', filename))`
   - *Attack Vector:* Directory traversal in URL path → arbitrary file read (e.g., `/etc/passwd`, `.env`, source code).
   - *Severity:* Critical

---

### 🛡️ Fully Patched Application

```python
import os
import sqlite3
import secrets
import re
from flask import Flask, request, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import jwt

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt'}

def get_db():
    db = sqlite3.connect('users.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT,
        role TEXT DEFAULT 'user'
    )''')
    db.commit()
    db.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    username = data.get('username', '')
    password = data.get('password', '')
    
    db = get_db()
    user = db.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,)).fetchone()
    db.close()
    
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = jwt.encode({'user_id': user['id'], 'role': user['role']}, JWT_SECRET, algorithm='HS256')
    return jsonify({'token': token})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    password_hash = generate_password_hash(password, method='scrypt')

    db = get_db()
    try:
        db.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                   (username, password_hash, email))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409
    finally:
        db.close()

    return jsonify({'message': 'User created'}), 201

@app.route('/api/profile', methods=['GET'])
def get_profile():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401

    token = auth_header.replace('Bearer ', '')
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Unauthorized'}), 401

    db = get_db()
    user = db.execute("SELECT id, username, email, role FROM users WHERE id = ?",
                      (payload.get('user_id'),)).fetchone()
    db.close()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'id': user['id'],
        'username': user['username'],
        'email': user['email'],
        'role': user['role']
    })

@app.route('/api/users/search', methods=['GET'])
def search_users():
    q = request.args.get('q', '').strip()
    role = request.args.get('role', '').strip()

    if not re.match(r'^[a-zA-Z0-9_\- ]+$', q):
        return jsonify({'error': 'Invalid search query'}), 400
    if role and not re.match(r'^[a-zA-Z0-9_\- ]+$', role):
        return jsonify({'error': 'Invalid role parameter'}), 400

    db = get_db()
    if role:
        users = db.execute("SELECT id, username, email FROM users WHERE username LIKE ? AND role = ?",
                           (f'%{q}%', role)).fetchall()
    else:
        users = db.execute("SELECT id, username, email FROM users WHERE username LIKE ?",
                           (f'%{q}%',)).fetchall()
    db.close()

    return jsonify([dict(u) for u in users])

@app.route('/api/download/<path:filename>')
def download_file(filename):
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename.startswith('.'):
        return jsonify({'error': 'Invalid filename'}), 400

    ext = safe_filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'File type not allowed'}), 400

    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    real_path = os.path.realpath(file_path)
    
    if not real_path.startswith(os.path.realpath(UPLOAD_DIR)):
        return jsonify({'error': 'Access denied'}), 403

    if not os.path.isfile(real_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(real_path, as_attachment=True)

@app.errorhandler(500)
def handle_500(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
```

---

### ✅ Security Checklist (Beyond the 5 Bugs)

| Area | Recommendation |
|------|----------------|
| **Authentication** | Implement JWT expiration (`exp` claim), token rotation, and secure cookie storage (`HttpOnly`, `Secure`, `SameSite=Strict`). |
| **Rate Limiting** | Add `Flask-Limiter` to `/api/login`, `/api/register`, and `/api/profile` to prevent brute-force and credential stuffing. |
| **Transport Security** | Enforce HTTPS in production; configure HSTS, TLS 1.2+, and disable weak ciphers. |
| **Input Validation** | Apply strict schema validation (e.g., `marshmallow` or `pydantic`) to all JSON payloads and query parameters. |
| **Headers** | Set security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy`, `Referrer-Policy`. |
| **Logging** | Use structured logging; never log passwords, tokens, or full request bodies. Rotate logs securely. |
| **Dependencies** | Run `pip-audit` or `safety` regularly; pin exact versions in `requirements.txt`; use `pip install --no-deps` for critical libs if needed. |
| **Database** | Use connection pooling (`sqlite3` is file-locked; consider PostgreSQL/MySQL for production); enable WAL mode; backup regularly. |
| **CORS** | Configure explicit `Access-Control-Allow-Origin` instead of `*`; restrict to known frontend domains. |
| **Environment** | Never commit `.env` files; use secret managers (AWS Secrets Manager, HashiCorp Vault, or Kubernetes Secrets) in production. |

## Metadata
- **Model**: intel-qwen3.6-27b-int4-16k
- **Elapsed**: 161494ms
- **Prompt tokens**: 1078
- **Completion tokens**: 7762
- **Total tokens**: 8840
- **Gen tok/s**: 48.0
