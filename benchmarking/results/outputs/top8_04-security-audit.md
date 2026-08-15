# Output: top-8 × 04-security-audit

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

## Response
Here's a thinking process:

1.  **Understand User Goal:** The user wants a vulnerability audit of a provided Flask application. They specifically ask for 5 security vulnerabilities, each with: type/category, line number/reference, attack vector, severity, and exact code quote. Then, they want a fully patched version of the app and a security checklist. Constraints are specified: use `werkzeug.security` for passwords, parameterized queries, validate/sanitize file uploads, remove/protect debug endpoints, use env vars for secrets. No placeholders/TODOs.

2.  **Analyze the Provided Code:**
    I'll go through the code line by line to identify vulnerabilities.

    *Line A:* `password_hash = hashlib.md5(password.encode()).hexdigest()`
    - Vulnerability: Weak password hashing (MD5 is broken, fast, vulnerable to rainbow tables/brute force).
    - Category: Cryptographic / Password Storage
    - Attack Vector: Attacker can crack MD5 hashes offline using rainbow tables or GPU acceleration to recover plaintext passwords.
    - Severity: Critical/High

    *Line B:* `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
    - Vulnerability: SQL Injection (String formatting in SQL query).
    - Category: Injection / SQLi
    - Attack Vector: Attacker provides malicious input in `username`, `password`, or `email` fields to inject arbitrary SQL commands, potentially extracting, modifying, or deleting data.
    - Severity: Critical

    *Line C:* `return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})`
    - Vulnerability: Sensitive Data Exposure / Information Disclosure (Returning password hash in API response).
    - Category: Information Disclosure / Sensitive Data Exposure
    - Attack Vector: Attacker calls `/api/profile` and receives the user's password hash, which can be cracked offline or used in credential stuffing if reused elsewhere.
    - Severity: High

    *Line D & E:* `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)` and `if role: sql += f" AND role = '{role}'"`
    - Vulnerability: SQL Injection (String formatting in SQL query).
    - Category: Injection / SQLi
    -

## Metadata
- **Top-k**: 8
- **Elapsed**: 6584ms
- **Prompt tokens**: 1078
- **Completion tokens**: 512
- **Total tokens**: 1590
