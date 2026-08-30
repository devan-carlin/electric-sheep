# Output: davidau-qwen3.6-40b × 04-security-audit

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
Let me analyze this Flask application for security vulnerabilities. I need to find 5 security issues, identify them by category, quote the exact code, describe attack vectors, rate severity, provide a patched version, and include a security checklist.

Let me go through the code systematically:

1. **Line A - Weak Password Hashing**: `password_hash = hashlib.md5(password.encode()).hexdigest()`
   - MD5 is cryptographically broken and too fast for password hashing
   - Category: CWE-328 (Use of Weak Cryptographic Algorithm)
   - Attack: Rainbow table or brute-force attack to recover passwords

2. **Line B - SQL Injection**: `db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")`
   - String formatting in SQL query
   - Category: CWE-89 (SQL Injection)
   - Attack: Inject SQL via username/email fields

3. **Line C - Sensitive Data Exposure**: Returns `password_hash` in profile response
   - Category: CWE-200 (Exposure of Sensitive Information)
   - Attack: Attacker retrieves password hash and cracks it offline

4. **Line D - SQL Injection**: `sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)`
   - String formatting in SQL query
   - Category: CWE-89 (SQL Injection)
   - Attack: Inject SQL via 'q' parameter

5. **Line E - SQL Injection**: `if role: sql += f" AND role = '{role}'"`
   - String formatting in SQL query
   - Category: CWE-89 (SQL Injection)
   - Attack: Inject SQL via 'role' parameter

6. **Line F - Path Traversal**: `return send_file(os.path.join('/tmp/uploads', filename))`
   - No validation of filename, allows directory traversal
   - Category: CWE-22 (Path Traversal)
   - Attack: Use `../../../etc/passwd` to read arbitrary files

7. **Hardcoded Secrets**: `app.secret_key = 'dev-secret-key-12345'` and `JWT_SECRET = 'jwt-secret-production-key'`
   - Category: CWE-798 (Use of Hard-coded Credentials)
   - Attack: Use known secrets to forge tokens/sign cookies

8. **Debug

## Metadata
- **Model**: davidau-qwen3.6-40b
- **Elapsed**: 24900ms
- **Prompt tokens**: 1078
- **Completion tokens**: 512
- **Total tokens**: 1590
- **Gen tok/s**: 20.5
