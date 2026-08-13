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

1.  **Analyze User Input:**
   - **Input Code:** A Flask app with 5 marked lines (A-F, though only A, B, C, D, E, F are mentioned in the prompt, the prompt says "5 security vulnerabilities" and lists A-F. I need to find exactly 5. Let's map them:
     - LINE A: `hashlib.md5(password.encode()).hexdigest()` -> Weak hashing
     - LINE B: `f"INSERT INTO users ... VALUES ('{username}', '{password_hash}', '{email}')" ` -> SQL Injection
     - LINE C: Returns `password_hash` in profile response -> Information Disclosure
     - LINE D: `sql = "SELECT ... LIKE '%{}%'".format(q)` -> SQL Injection
     - LINE E: `sql += f" AND role = '{role}'"` -> SQL Injection
     - LINE F: `send_file(os.path.join('/tmp/uploads', filename))` -> Path Traversal
     - Wait, the prompt says "5 security vulnerabilities". Lines D and E are both SQLi, but they might count as one category or two instances. I'll treat them as separate instances or combine them. The prompt asks to list each one (type + line + attack vector). I'll list A, B, C, D/E, F. Actually, D and E are two SQLi lines. I'll list them separately as requested, or note they're the same category. The prompt says "5 security vulnerabilities", so I'll pick the 5 most distinct ones or follow the lines. Let's stick to A, B, C, D, F. Or A, B, C, D/E, F. I'll list exactly 5 as requested.
     - Let's count: A (Weak Crypto), B (SQLi), C (Info Disc), D (SQLi), F (Path Traversal). That's 5. E is also SQLi, but I'll focus on the first 5 or combine D&E. I'll list A, B, C, D, F as the 5.
   - **Requirements:**
     1. Identify all five (label by category)
     2. For each: quote code, describe attack vector, rate severity
     3. Provide fully patched version
     4. Include security checklist
   - **Constraints:**
     - Use `werkzeug

## Metadata
- **Top-k**: 8
- **Elapsed**: 6500ms
- **Prompt tokens**: 1078
- **Completion tokens**: 512
- **Total tokens**: 1590
