# Output: qwen3.8-27b-int4-16k × 04-security-audit

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
We need answer user's request. Need analyze code. User says Flask app has 5 security vulnerabilities. They provide first snippet with lines A-F, then additional code appended? Let's parse.

User:
"# Vulnerability Audit — Find the Bugs

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

Produce the analysis and patched code. No placeholders, no TODOs."

There is inconsistency: first says 5 vulnerabilities, then code includes more potential issues: hardcoded secrets, MD5, SQL injection, sensitive data exposure, path traversal, debug endpoints, error details, debug=True, JWT no validation? Need choose five? The prompt likely expects five from original snippet: A weak hash, B SQLi, C sensitive info exposure, D/E SQLi, F path traversal. But then additional code includes debug endpoints and error handler. Requirements mention remove/protect debug endpoints, env vars. It says five vulnerabilities but constraints include debug endpoints. Maybe the five are: 1 weak password hashing (MD5), 2 SQL injection in register, 3 sensitive data exposure (password_hash), 4 SQL injection in search, 5 path traversal. But then debug endpoints are additional? The requirements ask identify all five vulnerabilities, but also include security checklist beyond 5. The appended code includes debug endpoints and error handler, likely they want those as vulnerabilities? Let's inspect.

The first code block has lines A-F. It says 5 vulnerabilities. There are 6 marked lines but B and D/E maybe two SQLi? Actually line B SQLi, line D SQLi, line E SQLi. That's 3 SQLi. But maybe they count: A weak hashing, B SQLi, C sensitive info, D SQLi, F path traversal = 5, ignoring E? Or B and D/E as one? The prompt: "This Flask app has 5 security vulnerabilities. List each one (type + line + attack vector):" then code with LINE A-F. Then after "For each vulnerability..." there is extra code not in code fence? It seems user pasted additional code after the initial question, maybe from a larger app. It includes admin endpoint, debug endpoints, error handler, app.run debug. The requirements ask remove/protect debug endpoints, env vars. So maybe the five vulnerabilities in full app are: hardcoded secrets, SQL injection, path traversal, debug endpoints, weak password hashing? But line C sensitive data? Need decide.

Let's read full user text carefully. It starts with code block containing import... download_file. Then after code block: "For each vulnerability, give: (1) CVE category, (2) which line, (3) one-sentence attack vector.
    token = auth.replace('Bearer ', '')

    try:
...
```
**Requirements:**..."
It appears there is a second code fragment not fenced, maybe continuation of app. It includes admin list users, debug stack, error-test, errorhandler, main. The first code block ended at download_file. The second fragment starts mid-function? "token = auth.replace..." likely part of an admin endpoint not fully shown. It references get_db, init_db not defined. The user likely combined two prompts? They ask produce analysis and patched code. Need satisfy: identify all five vulnerabilities, quote exact code, attack vector, severity; provide fully patched version; checklist. Constraints.

Need likely produce final with five vulnerabilities. Which five? We can state the five primary vulnerabilities in the provided code, and note additional hardening issues in checklist. But requirements mention debug endpoints; if not among five, checklist can cover. However "Remove or protect debug endpoints" is a constraint, so patched code must remove/protect. The five could include debug endpoints as one. Let's consider likely expected five from full code:
1. Hardcoded secrets (app.secret_key, JWT_SECRET) - CWE-798, line app.secret_key = ... / JWT_SECRET = ...
2. Weak password hashing MD5 - CWE-326/328, LINE A
3. SQL injection - LINE B/D/E (maybe one category)
4. Sensitive information exposure - LINE C (password_hash) or error handler details
5. Path traversal - LINE F
6. Debug endpoints / debug=True - CWE-489/209
That's more than 5.

The prompt says "This Flask app has 5 security vulnerabilities" likely from initial snippet with 5 (A, B, C, D, F) and E maybe part of D? But then constraints include debug endpoints. Could be they expect 5: weak hashing, SQLi, sensitive data, path traversal, debug endpoints. But line B and D/E are both SQLi; maybe count as one. Hardcoded secrets maybe not counted? But constraints env vars.

Need craft answer robust: "The five highest-priority vulnerabilities in the code as provided are..." and include categories. We can include line references. To be safe, list five that cover all constraints: (1) Hardcoded secrets, (2) Weak password hashing, (3) SQL injection, (4) Sensitive data exposure / debug info disclosure, (5) Path traversal. But then debug endpoints? Could fold debug endpoints into sensitive info disclosure? Or list debug endpoints as one and sensitive data as another? Need exactly five.

Let's map to requirements:
- Use werkzeug.security for password hashing -> vulnerability weak MD5.
- Use parameterized queries everywhere -> SQL injection.
- Validate and sanitize file uploads -> path traversal.
- Remove or protect debug endpoints -> debug endpoints / debug=True.
- Use environment variables for secrets -> hardcoded secrets.
That's exactly five constraints. The line C password_hash exposure is also a vulnerability but maybe not in constraints? It is in code. The prompt says 5 vulnerabilities, likely these five correspond to constraints. But initial code has LINE C. If we omit line C, maybe fail. Could include line C as part of sensitive information exposure and debug endpoints as additional? But then hardcoded secrets? Hmm.

The user explicitly: "This Flask app has 5 security vulnerabilities. List each one (type + line + attack vector):" then code with LINE A-F. Then later requirements include debug endpoints. Maybe the five are from the first code: A, B, C, D, F (E is part of D? or B?). But constraints mention debug endpoints because appended code has them. The final requirements likely override: identify all five vulnerabilities (label each by category) and provide patched code. The five could be:
1. Weak password hashing (LINE A)
2. SQL injection (LINE B, D, E)
3. Sensitive data exposure (LINE C)
4. Path traversal (LINE F)
5. Insecure debug/error disclosure (debug endpoints, errorhandler, debug=True)
Hardcoded secrets? Could be in checklist or patched code. But constraint says use env vars for secrets, so we must patch. If not listed as one of five, maybe okay as additional improvement? But "Use environment variables for secrets" is a constraint, not necessarily one of five. However hardcoded secrets is a vulnerability. If we list debug as fifth, we can still patch secrets.

Need decide final structure. We can say: "I’m treating the five required findings as the five distinct vulnerability classes that the constraints target. Where a class appears on multiple lines, I list all lines." Then list:
1. Hardcoded credentials/secrets (CWE-798) - lines app.secret_key, JWT_SECRET. Attack: attacker reads source or extracts keys to forge JWTs/session cookies. Severity Critical.
2. Weak password hashing (CWE-326/328) - LINE A. Attack: rainbow tables/offline hash cracking. Severity High.
3. SQL injection (CWE-89) - LINE B, LINE D, LINE E. Attack: inject SQL to dump users or bypass auth. Severity Critical.
4. Sensitive information exposure (CWE-209/532) - LINE C and debug/error endpoints? But if separate? Maybe line C: returns password_hash. Attack: authenticated user obtains MD5 hash and cracks. Severity High.
5. Path traversal (CWE-22) - LINE F. Attack: ../../etc/passwd. Severity High/Critical.
Then debug endpoints? Could be in checklist. But requirement "Remove or protect debug endpoints" must be in patched code. We can include in checklist as additional. But if the five expected include debug endpoints, we might miss. Could instead make #4 "Debug/error information disclosure" and mention line C as part of it? But line C is explicit. Need quote exact code for each. If #4 debug, quote debug_stack/errorhandler/app.run. Then line C not listed. But line C is a clear vulnerability. The prompt's initial "LINE C" likely expects it. So include line C.

Maybe list five as:
1. Weak password hashing (LINE A)
2. SQL injection (LINE B, D, E)
3. Sensitive data exposure (LINE C)
4. Path traversal (LINE F)
5. Insecure debug/error disclosure (debug endpoints, errorhandler, debug=True)
Then hardcoded secrets in checklist? But constraint says env vars; we can patch and mention as additional hardening, but it's a vulnerability. The user says five vulnerabilities; if we add hardcoded secrets as checklist, okay? But "Use environment variables for secrets" is a constraint, not necessarily finding. Yet hardcoded secrets is severe. Could include in checklist as "also fixed". But if they expect five, maybe not.

Alternative: list five as:
1. Hardcoded secrets (app.secret_key/JWT_SECRET)
2. Weak password hashing (LINE A)
3. SQL injection (LINE B/D/E)
4. Sensitive information exposure (LINE C + debug/error)
5. Path traversal (LINE F)
This covers line C and debug endpoints under sensitive info. But then debug endpoints not separate. Constraint remove/protect debug endpoints satisfied in patched code. Severity? Sensitive info High. Hardcoded secrets Critical. This seems plausible. But the prompt asks "which line" for each. For hardcoded secrets line: app.secret_key = ... and JWT_SECRET = ... (not labeled). For sensitive info: LINE C and debug endpoints? Could quote multiple.

Need produce "fully patched version of the application". Need include all endpoints? The provided code is incomplete: missing init_db, get_db, admin endpoint definition. We need produce complete Flask app with no placeholders/TODOs. Need implement register, profile, search, download, maybe admin list users? The appended fragment includes admin endpoint code but no route decorator? It starts with token = auth.replace... likely inside a function. We can include a complete admin endpoint /api/admin/users with proper auth and role check from DB not JWT? Need be secure. Also debug endpoints removed or protected. Error handler generic. Use env vars. Use werkzeug.security generate_password_hash/check_password_hash. Use parameterized queries. File upload validation: extension whitelist, safe filename. Need maybe upload endpoint? The code only has download. Constraint says validate and sanitize file uploads (extension whitelist, safe filename). We should include an upload endpoint to demonstrate, or at least download validation. "file uploads" maybe need upload route. We can add /api/upload with secure filename and extension whitelist. But original app only download. To satisfy, include upload and download. Need no placeholders. Use secure_filename from werkzeug.utils? Constraint says safe filename; can use secure_filename plus extension whitelist. But secure_filename may allow no extension? We'll validate.

Need use sqlite3. Need init_db with users table. Need maybe role column. The search uses role. Register doesn't set role; default user. Admin endpoint selects role. Need create table with id, username, password_hash, email, role, created_at, last_login. Use unique username/email. Use parameterized.

JWT: Use PyJWT. Need validate token, maybe require exp, iat, iss? Use jwt.decode with algorithms HS256, options require exp. Use secret from env. For profile, don't return password_hash. For admin, check role from database, not JWT claim (to prevent forged role if secret compromised? But if JWT secret secure, role claim okay; better DB). Need handle missing user. Use Authorization header. Need maybe login endpoint? Not provided. But JWT needs be issued somewhere. To be complete, include /api/login that validates password and returns JWT. The original doesn't have login, but profile expects token. A fully patched application should include login. Need no placeholders. Use env JWT_SECRET. Generate token with user_id, role, exp. Use jwt.encode. Need maybe refresh? Not necessary.

Need password hashing: werkzeug.security generate_password_hash uses scrypt by default? In recent Werkzeug, method='scrypt' or 'pbkdf2:sha256'. Constraint says bcrypt or scrypt. Use generate_password_hash(password, method='scrypt') and check_password_hash. Need note if bcrypt not installed? Werkzeug supports scrypt built-in. Use scrypt.

File upload: Need extension whitelist e.g. .png, .jpg, .jpeg, .pdf, .txt. Use secure_filename, ensure not empty, check suffix lower in whitelist, maybe max size. Store in /tmp/uploads? Better use env UPLOAD_DIR, create. For download, use secure_filename and ensure resolved path is within upload dir. Use os.path.realpath and commonpath. send_file with as_attachment maybe. Need sanitize filename: secure_filename, reject if empty, check extension. For path traversal, use os.path.basename? But if filename includes subdirs, secure_filename may keep subdirs? It replaces path separators? secure_filename('..%2F..%2Fetc%2Fpasswd')? It may return 'etc-passwd'? Need robust: take basename, secure_filename, then check extension. For download, use secure_filename(filename) and ensure it is within UPLOAD_DIR. Use os.path.realpath.

Secrets: app.secret_key = os.environ.get('SECRET_KEY', ...)? Constraint use environment variables for secrets. No placeholders? We can require env vars and fail if missing? "No placeholders" means don't put TODO. We can use os.environ['SECRET_KEY'] and os.environ['JWT_SECRET'] but if not set app fails. That's okay? Maybe provide default? Constraint says use environment variables for secrets, not hardcode. Could use os.environ.get with no default? If missing, RuntimeError. In code, at top: SECRET_KEY = os.environ.get('SECRET_KEY'); JWT_SECRET = os.environ.get('JWT_SECRET'); if not SECRET_KEY or not JWT_SECRET: raise RuntimeError(...). That's not placeholder. But for runnable, maybe user must set. That's acceptable. Or use os.environ['SECRET_KEY']. Need no hardcoded secrets.

Debug endpoints: Remove /api/debug/stack, /api/error-test. If need protect, can remove. Error handler returns generic JSON, no details. app.run(debug=False). If __main__, maybe use app.run(host='127.0.0.1', port=5000, debug=False). But production should use WSGI. We can include if __name__ == '__main__': init_db(); app.run(host='127.0.0.1', port=5000, debug=False). Constraint remove/protect debug endpoints. Good.

SQL injection: parameterized. For LIKE, use ? with f"%{q}%"? Need escape LIKE wildcards? To prevent user using % _ as wildcards, can escape with ESCAPE. In SQLite, use LIKE ? ESCAPE '\' and escape backslash, %, _. Or use parameterized with q.replace('\\','\\\\').replace('%','\\%').replace('_','\\_') and SQL LIKE ? ESCAPE '\'. That's secure. Or use GLOB? But LIKE okay. For role, parameterized.

Register: validate username, password, email. Need prevent duplicate. Use parameterized. Password min length? Could enforce. Email basic validation? Use simple regex or just non-empty. Need no external libs? Can use email-validator? Not required. Use simple check. Username alphanumeric? Could enforce. Need safe.

Profile: decode JWT, fetch user by id, return only id, username, email, role maybe. No password_hash. Handle missing user 404.

Search: q and role parameterized. Return limited fields. Maybe limit results.

Admin: require valid JWT and DB role admin. Return users without password_hash. Use parameterized? No injection. Maybe pagination.

Upload: accept file, validate extension, size, secure filename, save. Need maybe auth? Could require login. To be secure, require authenticated user. But original not. We can require Bearer token. Use same auth helper. Store with unique filename to avoid overwrite: uuid4 hex + secure_filename. Or use secure_filename only. Constraint safe filename. Use uuid prefix.

Download: require auth? Original public. Could be public but path traversal fixed. To be safer, require auth? Not specified. We can keep public but validate. Maybe file uploads should be protected. We can require auth for download too. But if original public, maybe not. Security checklist can mention. For patched, require auth for upload/download? It says validate and sanitize file uploads, not necessarily auth. But to be secure, protect. However if no login in original, adding auth may break. But we include login. Good.

Need include get_db with row_factory. init_db. Use context manager? We can define get_db: conn = sqlite3.connect(DATABASE); conn.row_factory = sqlite3.Row; return conn. Need close. Use try/finally.

JWT helper: get_current_user? Decode token. Need handle exceptions. Use jwt.decode(token, JWT_SECRET, algorithms=['HS256'], options={'require': ['exp']}). Maybe also verify audience? Not necessary. Use exp = datetime.utcnow() + timedelta(hours=1). Need import datetime, timedelta, uuid, re, os, sqlite3, jwt, flask, werkzeug.security, werkzeug.utils.

Potential issue: PyJWT jwt.decode with HS256 and secret from env. Need if token missing. Use Authorization header.

Need severity ratings. For five vulnerabilities:
Option final list:
1. Hardcoded secrets (CWE-798) - lines: app.secret_key = 'dev-secret-key-12345'; JWT_SECRET = 'jwt-secret-production-key'. Attack: source disclosure or repo leak lets attacker forge JWTs and hijack sessions. Severity Critical.
2. Weak password hashing (CWE-326/328) - LINE A. Attack: offline MD5 cracking/rainbow tables recover passwords. Severity High.
3. SQL injection (CWE-89) - LINE B, LINE D, LINE E. Attack: inject ' OR 1=1-- or UNION to dump users/bypass filters. Severity Critical.
4. Sensitive information exposure (CWE-209/532) - LINE C and debug/error endpoints? If we include debug here, quote line C and debug_stack/errorhandler/app.run. Attack: authenticated user gets password hash; unauthenticated gets stack/error details aiding exploitation. Severity High.
5. Path traversal (CWE-22) - LINE F. Attack: /api/download/..%2F..%2Fetc%2Fpasswd reads files. Severity High (Critical if sensitive files).
But then debug endpoints not separate. The requirement "Remove or protect debug endpoints" is in patched code and checklist. The five include sensitive info exposure covering debug. Hardcoded secrets included. This seems comprehensive.

But initial prompt asked "type + line + attack vector" for each. We can provide table with category, line(s), exact code, attack vector, severity. Need quote exact code. For hardcoded secrets, exact code lines. For SQLi, quote all three lines. For sensitive info, quote LINE C and maybe debug code. But if five, maybe don't overcomplicate. Could say "Finding 4: Sensitive information exposure (LINE C; also debug/error endpoints in appended code)". That quotes exact code for line C and debug. Good.

Need "CVE category" maybe they mean CWE? They say CVE category but likely CWE. Use CWE IDs. Could mention OWASP. Use "CWE-798" etc.

Patched code: Need complete. Let's design.

Imports:
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Flask, request, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)

SECRET_KEY = os.environ.get('SECRET_KEY')
JWT_SECRET = os.environ.get('JWT_SECRET')
DATABASE = os.environ.get('DATABASE', 'users.db')
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join('/tmp', 'uploads'))
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.txt'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

if not SECRET_KEY or not JWT_SECRET:
    raise RuntimeError('SECRET_KEY and JWT_SECRET environment variables must be set')

app.config.update(
    SECRET_KEY=SECRET_KEY,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    JSON_SORT_KEYS=False,
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL,
        last_login TEXT
    )''')
    conn.commit()
    conn.close()

def utcnow(): return datetime.now(timezone.utc).isoformat()

def is_valid_email(email): return bool(re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$', email))

def is_valid_username(username): return bool(re.match(r'^[A-Za-z0-9_]{3,32}$', username))

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def safe_upload_name(filename):
    name = secure_filename(filename)
    if not name or not allowed_file(name):
        return None
    return f"{uuid.uuid4().hex}_{name}"

def get_token():
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    return auth[7:].strip()

def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'], options={'require': ['exp']})
    except jwt.PyJWTError:
        return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = get_token()
        if not token:
            return jsonify({'error': 'Unauthorized'}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        db = get_db()
        try:
            user = db.execute('SELECT id, username, email, role FROM users WHERE id = ?', (payload.get('sub'),)).fetchone()
        finally:
            db.close()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        request.user = user
        return f(*args, **kwargs)
    return wrapper

def require_admin(f):
    @wraps(f)
    @require_auth
    def wrapper(*args, **kwargs):
        if request.user['role'] != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return wrapper

Routes:
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    email = str(data.get('email', '')).strip().lower()
    if not is_valid_username(username) or not is_valid_email(email) or len(password) < 8:
        return jsonify({'error': 'Invalid username, email, or password'}), 400
    password_hash = generate_password_hash(password, method='scrypt')
    db = get_db()
    try:
        db.execute('INSERT INTO users (username, password_hash, email, role, created_at) VALUES (?, ?, ?, ?, ?)',
                   (username, password_hash, email, 'user', utcnow()))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    finally:
        db.close()
    return jsonify({'message': 'User created'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    db = get_db()
    try:
        user = db.execute('SELECT id, username, email, role, password_hash FROM users WHERE username = ? OR email = ?', (username, username.lower())).fetchone()
    finally: db.close()
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    # update last_login
    db = get_db()
    try:
        db.execute('UPDATE users SET last_login = ? WHERE id = ?', (utcnow(), user['id']))
        db.commit()
    finally: db.close()
    payload = {
        'sub': user['id'],
        'username': user['username'],
        'role': user['role'],
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return jsonify({'token': token, 'token_type': 'Bearer'}), 200

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile():
    return jsonify({'id': request.user['id'], 'username': request.user['username'], 'email': request.user['email'], 'role': request.user['role']}), 200

@app.route('/api/users/search', methods=['GET'])
@require_auth
def search_users():
    q = request.args.get('q', '').strip()
    role = request.args.get('role', '').strip()
    # escape LIKE
    escaped_q = q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    sql = "SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\\'"
    params = [f"%{escaped_q}%"]
    if role:
        sql += " AND role = ?"
        params.append(role)
    sql += " LIMIT 100"
    db = get_db()
    try:
        rows = db.execute(sql, params).fetchall()
    finally: db.close()
    return jsonify([dict(r) for r in rows]), 200

@app.route('/api/admin/users', methods=['GET'])
@require_admin
def admin_users():
    db = get_db()
    try:
        rows = db.execute('SELECT id, username, email, role, created_at, last_login FROM users ORDER BY id LIMIT 1000').fetchall()
    finally: db.close()
    return jsonify([dict(r) for r in rows]), 200

Upload:
@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    original = secure_filename(file.filename)
    if not original or not allowed_file(original):
        return jsonify({'error': 'File type not allowed'}), 400
    stored_name = safe_upload_name(original)
    if not stored_name:
        return jsonify({'error': 'Invalid filename'}), 400
    file_path = os.path.join(UPLOAD_DIR, stored_name)
    file.save(file_path)
    return jsonify({'filename': stored_name}), 201

Download:
@app.route('/api/download/<path:filename>', methods=['GET'])
@require_auth
def download_file(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({'error': 'Invalid filename'}), 400
    # ensure no extension? maybe allow any uploaded file; but if extension whitelist? For download, check allowed? Could check allowed_file to prevent other files. But if stored names have uuid + original, extension allowed. Use allowed_file.
    if not allowed_file(safe_name):
        return jsonify({'error': 'File type not allowed'}), 400
    base = os.path.realpath(UPLOAD_DIR)
    file_path = os.path.realpath(os.path.join(base, safe_name))
    if not file_path.startswith(base + os.sep) and file_path != base:
        return jsonify({'error': 'Invalid filename'}), 400
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True, download_name=safe_name)

Error handlers:
@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Not found'}), 404
@app.errorhandler(405)
def method_not_allowed(e): return jsonify({'error': 'Method not allowed'}), 405
@app.errorhandler(413)
def too_large(e): return jsonify({'error': 'Payload too large'}), 413
@app.errorhandler(500)
def handle_500(e): return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='127.0.0.1', port=5000, debug=False)

Need ensure no debug endpoints. Good.

Potential issue: secure_filename may strip extension? If file.filename = 'file.png', secure_filename returns 'file.png'. allowed_file checks. If filename = 'file.png/../../etc/passwd', secure_filename? It may return 'file.png-etc-passwd'? Actually secure_filename replaces path separators with '-'? Let's check: werkzeug.utils.secure_filename('foo/bar') returns 'foo-bar'. So path traversal neutralized. But we also realpath check.

Need maybe use os.path.commonpath instead of startswith to avoid /tmp/uploads-evil. Use:
if os.path.commonpath([base, file_path]) != base: ...
But commonpath raises ValueError if different drives. Use try. Simpler:
if not file_path.startswith(base + os.sep): ...
Since base is /tmp/uploads, an attacker path /tmp/uploads-evil won't pass because startswith('/tmp/uploads/') false. Good.

Need ensure UPLOAD_DIR exists before realpath. makedirs.

Need maybe JWT decode: PyJWT accepts datetime for exp? It accepts int or datetime? PyJWT 2.x accepts datetime? I think it accepts int, float, or datetime? It may convert. To be safe, use int timestamps:
now = datetime.now(timezone.utc)
payload = {'sub': user['id'], 'role': user['role'], 'iat': int(now.timestamp()), 'exp': int((now + timedelta(hours=1)).timestamp())}
Then decode require exp. Good.

Need check_password_hash with scrypt. generate_password_hash(password, method='scrypt') works? Werkzeug's generate_password_hash supports method='scrypt' with salt. Yes.

Need maybe use sqlite3.Row dict conversion. In Python 3.9+ dict(row) works? sqlite3.Row supports keys? dict(row) works? It has keys method? Yes. Or use {k: row[k] for k in row.keys()}. Use dict(r) okay? In Python, sqlite3.Row is mapping? It supports keys. dict(row) works. Good.

Need maybe register email unique lower. If username or email exists, IntegrityError. Good.

Need login by username or email. If username contains @? It will query both. Fine.

Need search auth? Original public. But to protect PII, require auth. Good.

Need admin endpoint route. The appended fragment had no route, but we can include.

Need "No placeholders, no TODOs". Our code has no TODO. RuntimeError message is okay.

Need final answer: analysis and patched code. Need likely include table. Need mention "CVE category" but use CWE. Could say "CWE (common weakness enumeration) category; CVE is for specific vulnerabilities, but these map to CWE/OWASP." But user asks CVE category; maybe use "CWE/OWASP". We can say "I’m using CWE identifiers because the code patterns are weakness classes, not specific CVEs." That is accurate.

Need ensure five vulnerabilities. Let's choose final five:
1. Hardcoded secrets (CWE-798) - lines: app.secret_key = 'dev-secret-key-12345'; JWT_SECRET = 'jwt-secret-production-key'. Severity Critical.
2. Weak password hashing (CWE-326/328) - LINE A. Severity High.
3. SQL injection (CWE-89) - LINE B, LINE D, LINE E. Severity Critical.
4. Sensitive information exposure (CWE-209/532) - LINE C and debug/error endpoints (quote debug_stack, error_test, handle_500, app.run debug=True). Severity High.
5. Path traversal (CWE-22) - LINE F. Severity High.

This covers all constraints? Hardcoded secrets yes. Debug endpoints in #4. But requirement "Remove or protect debug endpoints" in patched code. Good.

But if they expected line C as separate and debug as separate, our #4 combines. It's still five. Need maybe in table for #4 quote exact code: LINE C and debug endpoints. Attack vector: authenticated user retrieves password_hash; unauthenticated requests debug/error endpoints to obtain stack traces/error details. Severity High. Good.

Need maybe mention "If you count debug endpoints separately, the app has more than five; I grouped information disclosure to keep the required five distinct classes." But user says five. Could include note: "The code contains additional hardening issues; I grouped related information-disclosure issues into one finding to match the five requested." That may be good.

Need severity: Hardcoded secrets Critical; SQLi Critical; weak hash High; sensitive info High; path traversal High (or Critical if can read sensitive files). Rate path traversal High. Could be Critical if /etc/passwd. But use High.

Need attack vector one-sentence each. In table maybe one sentence.

Patched code: Need include all. Need ensure no hardcoded secrets. But if we raise RuntimeError if env missing, code not runnable without env. That's okay. Could use os.environ.get with default? Constraint says use env vars for secrets; default would be hardcoded. So no default. But for local dev, user must set. No placeholders. Good.

Potential issue: app.secret_key required by Flask? We set. If missing, RuntimeError before app? We create app then check? Better check before app? Need app for config. We can:
SECRET_KEY = os.environ.get('SECRET_KEY')
JWT_SECRET = os.environ.get('JWT_SECRET')
if not SECRET_KEY or not JWT_SECRET: raise RuntimeError(...)
app = Flask(__name__)
app.secret_key = SECRET_KEY
Good.

Need maybe JWT_SECRET can be same as SECRET_KEY? Better separate. Require both.

Need maybe use environment variable for database and upload dir. Defaults are not secrets. Fine.

Need maybe use `werkzeug.security` for password hashing (bcrypt or scrypt). We use scrypt. Good.

Need maybe use parameterized queries everywhere. Yes.

Need validate and sanitize file uploads. Yes.

Need remove or protect debug endpoints. Removed.

Need use env vars for secrets. Yes.

Need maybe include security checklist additional improvements: HTTPS, rate limiting, CSRF, input validation, logging, dependency updates, least privilege, secrets management, JWT short expiry/refresh, audit logs, WAF, container hardening, file storage outside webroot, antivirus, CSP, security headers, DB backups, access control, MFA, etc.

Need final answer with code block. Ensure code syntactically correct. Let's write carefully.

Potential code issues:
- `from datetime import datetime, timedelta, timezone`
- `datetime.now(timezone.utc).timestamp()` works.
- `jwt.encode` returns str in PyJWT 2.x.
- `jwt.decode` with `options={'require': ['exp']}`. Need import jwt. If PyJWT 1.x, options require? It supports. Good.
- In require_auth, `request.user = user` works? Flask request object allows arbitrary attributes? Yes, request is a LocalProxy? Actually `request` is a proxy to the actual request; setting attributes on proxy? In Flask, `request` is a LocalProxy, but you can set attributes? I think `request.user = user` works? Let's verify: Flask's request is a LocalProxy to the actual request object; setting attribute on proxy sets on wrapped object? LocalProxy supports __setattr__? I think yes, it proxies setattr. But to be safe, use `g.user = user` from flask import g. Use `g.user`. Then in routes use `g.user`. Better. Import g. In require_auth, set `g.user = user`. In require_admin, check `g.user['role']`. In profile, use g.user. Good.

- In login, after first db close, second db. Could combine. Fine.
- In search, `ESCAPE '\\'` in Python string: SQL needs backslash. In Python, to get a single backslash in SQL string, use `ESCAPE '\\'`? Let's think. Python string literal `"ESCAPE '\\'"` results in SQL `ESCAPE '\'`? Actually Python: '\\' is one backslash. So `"ESCAPE '\\'"`? Need quotes. We want SQL: LIKE ? ESCAPE '\' . In Python string: "LIKE ? ESCAPE '\\'"? Let's count: Python literal `'\\'` is one backslash. To include SQL quote around backslash: `ESCAPE '\\'`? If Python string is `"ESCAPE '\\'"`, characters: E S C A P E space ' \ ' ? Wait Python: backslash escaped by backslash. `"\\'"`? Let's construct: We need SQL text: ESCAPE '\' (single quote, backslash, single quote). In Python double-quoted string: "ESCAPE '\\'"? The sequence `\\` produces one backslash, then `'`? Actually string: "ESCAPE '\\'" has characters: ESCAPE space, backslash? Let's write: `"ESCAPE '\\'"` length? It starts quote, E..., space, backslash? The Python source `\\` is two chars in source, produces one backslash. Then `'`? The source has `'` after `\\`? The string ends with `'"`? Let's do: `"ESCAPE '\\'"` source: double quote, ESCAPE space, backslash backslash, single quote, double quote. That produces: ESCAPE space, one backslash, single quote. Missing opening single quote before backslash. Need SQL: ESCAPE '\' . Python: "ESCAPE '\\'"? That's ESCAPE space, backslash, single quote. No opening quote. Need "ESCAPE '\\'"? Hmm.

SQL escape char can be specified as a string literal. To specify backslash as escape char, SQL literal is `'\'`? In SQL, single quote inside string must be escaped by doubling: `''`. So to represent a single backslash character in SQL string literal, you can write `'\'`? Wait SQL string literal: single quotes delimit. Backslash is not special in SQLite by default. To include a single quote, double it. To include a backslash, just backslash. So SQL literal for backslash is `'\'`? That would be: opening quote, backslash, closing quote? But the backslash is followed by quote; in SQL, backslash doesn't escape quote, so it's fine: `'\'` is a string containing backslash. In Python, to produce SQL text `'\'`, we need Python string: "'\\'"? Let's see: Python literal `"'\\'"` (single-quoted) contains: single quote, backslash, single quote. Source: ' (start), ' (escaped? Actually if using single quotes, to include single quote need escape or use double quotes). Easier use double quotes: `"'\\'"`? Python double-quoted string: `"'\\'"` characters: single quote, backslash, single quote. Source: double quote, single quote, backslash backslash, single quote, double quote. That yields `'\'`. So in SQL string: `ESCAPE '\\'`? Wait if we write in Python: `sql = "SELECT ... LIKE ? ESCAPE '\\'"`? Let's test mentally: Python double-quoted string `"ESCAPE '\\'"` source: double quote, ESCAPE space, backslash backslash, single quote, double quote. It produces: ESCAPE space, one backslash, single quote. That's not quoted. Need include opening single quote: `"ESCAPE '\\'"`? Source: double quote, ESCAPE space, single quote, backslash backslash, single quote, double quote. That produces: ESCAPE space, single quote, backslash, single quote. That's correct. So Python literal: `"ESCAPE '\\'"`? Wait that's what I just wrote: after space, single quote, then `\\`, then single quote. In text: `"ESCAPE '\\'"` has characters: double quote, E..., space, single quote, backslash, backslash, single quote, double quote. Yes produces ESCAPE space, single quote, one backslash, single quote. Good. So use `ESCAPE '\\'`? No, use `ESCAPE '\\'`? Let's be precise in final code: `sql = "SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\\'"`? In Markdown code, if I write `ESCAPE '\\'`, Python interprets as one backslash? Need include single quotes. I should write: `sql = "SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\\'"`? Let's count in code: `ESCAPE '\\'` in Python double quotes: source has backslash backslash then single quote? Actually `ESCAPE '\\'` is ESCAPE space, backslash, single quote? No, because `\\` is one backslash, then `'` is single quote. Missing opening quote. Need `ESCAPE '\\'`? Hmm.

Let's use a variable: `ESCAPE_CHAR = '\\'` (Python string one backslash). Then SQL: `sql = "SELECT ... LIKE ? ESCAPE ?"`? Can we parameterize ESCAPE? SQLite LIKE ESCAPE requires a string literal, not parameter? I think ESCAPE clause requires a string literal, not parameter. Could use `ESCAPE '\\'`? Need correct.

Alternative avoid ESCAPE by using `GLOB`? GLOB uses * and ? wildcards, not % _. Could escape * and ? with brackets? SQLite GLOB escape? GLOB has no ESCAPE clause? It does? SQLite GLOB uses * and ?, no escape. Could use `INSTR` or `LIKE` with parameter and not worry about wildcard injection? SQL injection is prevented; wildcard abuse is low severity. For simplicity, use parameterized `LIKE ?` with `f"%{q}%"` without escaping. That's secure from SQLi. The requirement only parameterized. We can avoid escape complexity. But if q contains % or _, it acts as wildcard; not a security vulnerability per se. Could mention in checklist. To keep code simple and correct, use `LIKE ?` with `f"%{q}%"`. That's parameterized. Good. No need ESCAPE. But if we want robust, can use `q.replace('%', '\\%')...` and ESCAPE. Need ensure syntax. Maybe use `ESCAPE '\\'`? Let's verify with Python: To get SQL `ESCAPE '\'`, Python string can be `"ESCAPE '\\'"`? Let's actually simulate: Python source: `"ESCAPE '\\'"` (double quote, ESCAPE space, backslash, backslash, single quote, double quote). It produces: ESCAPE space, one backslash, single quote. That's not correct. Need source: `"ESCAPE '\\'"`? Wait I keep same. Let's write desired SQL: ESCAPE '\' . As a Python double-quoted string, we need: opening double quote, ESCAPE space, single quote, backslash, single quote, closing double quote. To represent backslash in Python, use `\\`. So source: `"ESCAPE '\\'"`? That is: double quote, ESCAPE space, single quote, backslash, backslash, single quote, double quote. Yes! The source includes two backslashes between single quotes. In text, that appears as `ESCAPE '\\'`? Let's type: `"ESCAPE '\\'"` has after space: single quote, backslash, backslash, single quote. In Markdown, it will show as `ESCAPE '\\'`? Actually the code will show: `ESCAPE '\\'`? Let's count characters displayed: `ESCAPE '\\'` includes single quote? The string literal in code: `"ESCAPE '\\'"` displays as `"ESCAPE '\\'"` (with double quotes around). Inside: ESCAPE space, single quote, backslash, backslash, single quote. When Python parses, backslash backslash -> one backslash. So SQL gets ESCAPE space, single quote, one backslash, single quote. Correct. So in final code, write:
`sql = "SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\\'"`
Wait that displayed has only one backslash? In Markdown, to show two backslashes, type two. The string literal should be `"ESCAPE '\\'"`? That has two backslashes? Let's explicitly: characters between outer double quotes: E S C A P E space ' \ \ ' . In text: `ESCAPE '\\'`? The sequence `\\` is two backslashes in source. Then a single quote. So it is `ESCAPE '\\'`? Actually `\\` followed by `'` yields `\\'` (two backslashes then quote). So the literal is `"ESCAPE '\\'"`? That has single quote, two backslashes, single quote. Yes. In final code, I can write:
`sql = "SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\\'"`
But that shows one backslash? Let's not risk. Use a raw string? `r"ESCAPE '\'"`? In raw string, backslash before quote? Raw string cannot end with odd backslash, but `r"ESCAPE '\'"`? It contains single quote, backslash, single quote? In raw, backslash is literal, but the single quote after backslash is just char. The raw string: r"ESCAPE '\'" -> characters: ESCAPE space, single quote, backslash, single quote. That is correct SQL. But raw string with backslash before quote? It's okay because outer double quotes. Use `sql = r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"`? Wait raw string: r"ESCAPE '\'" has backslash then single quote; raw keeps backslash, and single quote is literal. That yields ESCAPE space, single quote, backslash, single quote. Correct. But in SQL, the escape char is backslash. Good. However in Python raw string, the backslash before the closing single quote? It's not closing outer double quote. Fine. Use raw string to avoid confusion:
`sql = r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"`
But does raw string end with `'"`? It has backslash then single quote then double quote. Raw string can contain backslash before quote. It will keep backslash. Good.
Then escape q: `escaped_q = q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')`. In Python, to replace backslash with two backslashes: `q.replace('\\', '\\\\')` (source: one backslash? Actually `'\\'` is one backslash, `'\\\\'` is two). Then replace % with backslash%: `'\\%'`? Python string `'\\%'` is backslash + %. Good. Use that.
But if we use raw SQL with ESCAPE '\', and escaped_q uses backslash, works.

However SQLite's LIKE ESCAPE with backslash: If escape char is backslash, then `\%` means literal %. Good.

Need ensure `escaped_q` parameter includes backslashes. Good.

But maybe simpler to avoid. I can include robust escape with raw string. Let's ensure code syntax:
```
escaped_q = q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
sql = r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"
params = [f"%{escaped_q}%"]
```
In raw string, `ESCAPE '\'`? The raw string is `r"SELECT ... ESCAPE '\'"`? Wait the raw string content: `SELECT ... ESCAPE '\'`? Let's write: `r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"` has after ESCAPE: space, single quote, backslash, single quote. Yes. Good.

But in Markdown, the backslash before the single quote might be interpreted? It's code block, fine.

Need maybe `sqlite3` LIKE ESCAPE expects a string literal, not parameter. Raw string okay.

Potential issue: If q contains backslash, we replace with double backslash. In SQL with escape char backslash, double backslash means literal backslash. Good.

Need maybe role validation: role in ('user','admin')? Could parameterize. If role arbitrary, no SQLi. Could restrict to known roles to avoid info leak. Use if role and role not in {'user','admin'}: return 400. Good.

Need maybe search limit. Good.

Need maybe admin users limit. Good.

Need maybe file upload extension whitelist includes .txt. Good.

Need maybe `secure_filename` can return name without extension if file has no dot. allowed_file fails. Good.

Need maybe `send_file` with `download_name` requires Werkzeug 2.0? It supports. If older, use `attachment_filename`. But modern. Could use `as_attachment=True, download_name=safe_name`. Good.

Need maybe `send_file` can send file from /tmp/uploads. Good.

Need maybe `os.path.realpath` and `os.path.commonpath`. Use startswith.

Need maybe `init_db()` before app.run. Good.

Need maybe `app.run(debug=False)`. Good.

Need maybe no debug endpoints. Good.

Need maybe error handler 500 generic. Good.

Need maybe `handle_500` should not log? Could log. But no external logging. Could use `app.logger.exception`? That logs to stderr. Fine. But not expose. Add:
```
@app.errorhandler(500)
def handle_500(error):
    app.logger.exception('Unhandled error')
    return jsonify({'error': 'Internal server error'}), 500
```
No details. Good.

Need maybe `init_db` called in main. If using WSGI, user should call. Could also call at import? Not necessary. But if app imported by gunicorn, init_db not called. Could call `init_db()` at module level after app? But if DB not writable? Better in main. For fully patched application, maybe call `init_db()` at import to ensure table. But if multiple workers, okay. Could do:
```
init_db()
if __name__ == '__main__': app.run(...)
```
But if env missing, raises before. If imported, init_db runs. That's okay. But if database path not set? default. Could call in main only. The original had init_db in main. Keep.

Need maybe `get_db` used before init? If request before init, table missing. In main, init before run. If WSGI, user should init. Could add `app.before_request`? Not. Maybe call `init_db()` at module load after function definition. That ensures. But if no DB permission, error. Fine. Let's call `init_db()` at bottom before main? But if env missing, already raised. If imported by test, it initializes. Good. However if multiple processes, CREATE IF NOT EXISTS okay. Do:
```
init_db()

if __name__ == '__main__':
    app.run(...)
```
No placeholders.

Need maybe `DATABASE` default 'users.db'. Not secret. Good.

Need maybe `UPLOAD_DIR` default '/tmp/uploads'. Not secret. Good.

Need maybe `ALLOWED_EXTENSIONS` set. Good.

Need maybe `MAX_CONTENT_LENGTH` 5MB. Good.

Need maybe `app.config['JSON_SORT_KEYS'] = False`? In Flask 2.3, JSON_SORT_KEYS deprecated? It may still work. Could omit. Not necessary.

Need maybe use `request.get_json(silent=True)` to avoid 415. Good.

Need maybe password min length 8. Good.

Need maybe username/email validation. Good.

Need maybe login rate limiting? Checklist.

Need maybe JWT `sub` is int. In decode, payload.get('sub'). Good.

Need maybe in require_auth, if payload sub missing, user None. Good.

Need maybe in login, if user found by email, username parameter may be email. Query `WHERE username = ? OR email = ?` with same value. If username is email, email match. If email is username, username match. Good. But if username contains lower? We lower only for email? We use `username.lower()` for email param. If username is 'User', email param 'user', okay. If actual email 'User@x', lower. Good.

Need maybe update last_login after check. Good.

Need maybe token role from DB at login. Good.

Need maybe admin check from DB each request. Good.

Need maybe profile returns role. Is that sensitive? It's user's own role, okay.

Need maybe search returns email of other users. Could be PII. If authenticated, maybe okay. Could restrict to username only. But original returns email. We can keep. Checklist mention data minimization.

Need maybe admin returns last_login. Original omitted last_login? It selected but returned created_at only. We can return last_login? Maybe not. To minimize, return id, username, email, role, created_at. But admin may need last_login. Could include. Not a vulnerability. Keep maybe.

Need maybe no debug endpoints. Good.

Now final answer structure:
- Brief note: using CWE/OWASP categories because CVEs are specific; five distinct vulnerability classes.
- Table with #, category, exact code, line(s), attack vector, severity.
Need quote exact code. For hardcoded secrets, quote:
```
app.secret_key = 'dev-secret-key-12345'
JWT_SECRET = 'jwt-secret-production-key'
```
Line: top of file (not labeled). Attack vector: source/repo leak or binary extraction reveals keys, attacker forges JWTs/session cookies. Severity Critical.

For weak hash:
```
password_hash = hashlib.md5(password.encode()).hexdigest()  # LINE A
```
Attack: obtain users.db, crack MD5 hashes with rainbow tables/GPU. Severity High.

SQLi:
```
db.execute(f"INSERT INTO users (username, password_hash, email) VALUES ('{username}', '{password_hash}', '{email}')")  # LINE B
...
sql = "SELECT id, username, email FROM users WHERE username LIKE '%{}%'".format(q)  # LINE D
if role: sql += f" AND role = '{role}'"  # LINE E
```
Attack: send username `x', 'hash', 'email', 'admin'--` or q `%' UNION SELECT ...--` to inject SQL. Severity Critical.

Sensitive info:
```
return jsonify({'id': user['id'], 'username': user['username'], 'password_hash': user['password_hash']})  # LINE C
```
and debug endpoints:
```
@app.route('/api/debug/stack', methods=['GET'])
def debug_stack(): ...
@app.route('/api/error-test', methods=['GET']) ...
@app.errorhandler(500) ... details ...
app.run(debug=True, host='0.0.0.0', port=5000)
```
Attack: authenticated user gets password hash; unauthenticated gets stack traces/error details and Werkzeug debugger if debug=True. Severity High.

Path traversal:
```
return send_file(os.path.join('/tmp/uploads', filename))  # LINE F
```
Attack: request `/api/download/..%2F..%2F..%2Fetc%2Fpasswd` to read arbitrary files. Severity High.

Need maybe mention "line" for debug endpoints from appended code not labeled. Use "appended code".

Patched code: Provide full code. Need ensure no placeholders. Let's write final code carefully.

Potential code:
```
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import Flask, request, jsonify, send_file, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

SECRET_KEY = os.environ.get('SECRET_KEY')
JWT_SECRET = os.environ.get('JWT_SECRET')
if not SECRET_KEY or not JWT_SECRET:
    raise RuntimeError('SECRET_KEY and JWT_SECRET environment variables must be set')

DATABASE = os.environ.get('DATABASE', 'users.db')
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join('/tmp', 'uploads'))
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.txt'}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL,
            last_login TEXT
        )
        '''
    )
    conn.commit()
    conn.close()

def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

def valid_username(value):
    return bool(re.fullmatch(r'[A-Za-z0-9_]{3,32}', value))

def valid_email(value):
    return bool(re.fullmatch(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', value))

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def safe_stored_name(filename):
    clean = secure_filename(filename)
    if not clean or not allowed_file(clean):
        return None
    return f'{uuid.uuid4().hex}_{clean}'

def bearer_token():
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    return auth[7:].strip()

def decode_jwt(token):
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=['HS256'],
            options={'require': ['exp']}
        )
    except jwt.PyJWTError:
        return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = bearer_token()
        if not token:
            return jsonify({'error': 'Unauthorized'}), 401
        payload = decode_jwt(token)
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        db = get_db()
        try:
            user = db.execute(
                'SELECT id, username, email, role FROM users WHERE id = ?',
                (payload.get('sub'),)
            ).fetchone()
        finally:
            db.close()
        if not user:
            return jsonify({'error': 'Unauthorized'}), 401
        g.user = user
        return f(*args, **kwargs)
    return wrapper

def require_admin(f):
    @wraps(f)
    @require_auth
    def wrapper(*args, **kwargs):
        if g.user['role'] != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return wrapper

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    email = str(data.get('email', '')).strip().lower()

    if not valid_username(username) or not valid_email(email) or len(password) < 8:
        return jsonify({'error': 'Invalid username, email, or password'}), 400

    password_hash = generate_password_hash(password, method='scrypt')
    db = get_db()
    try:
        db.execute(
            'INSERT INTO users (username, password_hash, email, role, created_at) VALUES (?, ?, ?, ?, ?)',
            (username, password_hash, email, 'user', utcnow_iso())
        )
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    finally:
        db.close()

    return jsonify({'message': 'User created'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))

    db = get_db()
    try:
        user = db.execute(
            'SELECT id, username, email, role, password_hash FROM users WHERE username = ? OR email = ?',
            (username, username.lower())
        ).fetchone()
    finally:
        db.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    db = get_db()
    try:
        db.execute('UPDATE users SET last_login = ? WHERE id = ?', (utcnow_iso(), user['id']))
        db.commit()
    finally:
        db.close()

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            'sub': user['id'],
            'username': user['username'],
            'role': user['role'],
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(hours=1)).timestamp())
        },
        JWT_SECRET,
        algorithm='HS256'
    )

    return jsonify({'token': token, 'token_type': 'Bearer'}), 200

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile():
    return jsonify({
        'id': g.user['id'],
        'username': g.user['username'],
        'email': g.user['email'],
        'role': g.user['role']
    }), 200

@app.route('/api/users/search', methods=['GET'])
@require_auth
def search_users():
    q = request.args.get('q', '').strip()
    role = request.args.get('role', '').strip()

    if role and role not in {'user', 'admin'}:
        return jsonify({'error': 'Invalid role'}), 400

    escaped_q = q.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    sql = r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"
    params = [f'%{escaped_q}%']

    if role:
        sql += ' AND role = ?'
        params.append(role)

    sql += ' LIMIT 100'

    db = get_db()
    try:
        rows = db.execute(sql, params).fetchall()
    finally:
        db.close()

    return jsonify([dict(row) for row in rows]), 200

@app.route('/api/admin/users', methods=['GET'])
@require_admin
def admin_users():
    db = get_db()
    try:
        rows = db.execute(
            'SELECT id, username, email, role, created_at, last_login FROM users ORDER BY id LIMIT 1000'
        ).fetchall()
    finally:
        db.close()

    return jsonify([dict(row) for row in rows]), 200

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    original = secure_filename(file.filename)
    if not original or not allowed_file(original):
        return jsonify({'error': 'File type not allowed'}), 400

    stored_name = safe_stored_name(original)
    if not stored_name:
        return jsonify({'error': 'Invalid filename'}), 400

    file.save(os.path.join(UPLOAD_DIR, stored_name))
    return jsonify({'filename': stored_name}), 201

@app.route('/api/download/<path:filename>', methods=['GET'])
@require_auth
def download_file(filename):
    safe_name = secure_filename(filename)
    if not safe_name or not allowed_file(safe_name):
        return jsonify({'error': 'Invalid filename'}), 400

    base_dir = os.path.realpath(UPLOAD_DIR)
    file_path = os.path.realpath(os.path.join(base_dir, safe_name))
    if file_path != base_dir and not file_path.startswith(base_dir + os.sep):
        return jsonify({'error': 'Invalid filename'}), 400

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True, download_name=safe_name)

@app.errorhandler(404)
def handle_404(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(405)
def handle_405(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def handle_413(error):
    return jsonify({'error': 'Payload too large'}), 413

@app.errorhandler(500)
def handle_500(error):
    app.logger.exception('Unhandled server error')
    return jsonify({'error': 'Internal server error'}), 500

init_db()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
```

Check raw string: `r"SELECT id, username, email FROM users WHERE username LIKE ? ESCAPE '\'"` In raw string, the backslash before the single quote is literal, and the single quote is literal. But does the raw string terminate at the single quote? No, outer double quotes. It contains a single quote. Good. However raw string cannot end with a backslash, but it doesn't; it ends with single quote then double quote. Good.

But in Python, raw string `r"ESCAPE '\'"`? Let's test mentally: r"ESCAPE '\'" characters: E..., space, ', \, ',? Wait the source has backslash then single quote then double quote. In raw, backslash is literal, single quote literal. It does not terminate because outer double quote. Good. It yields ESCAPE space, single quote, backslash, single quote. Correct.

But in the code above, I wrote `ESCAPE '\'"`? Actually `r"SELECT ... ESCAPE '\'"` has after ESCAPE: space, single quote, backslash, single quote. Yes.

Potential issue: In SQLite, `ESCAPE '\'` with backslash as escape char. But in SQL, a backslash inside a string literal is just a character. Good.

Potential issue: `escaped_q = q.replace('\\', '\\\\')` in Python: `'\\'` is one backslash, `'\\\\'` is two. Good. `.replace('%', '\\%')`: `'\\%'` is backslash + %. Good. `.replace('_', '\\_')`: backslash + _. Good.

Potential issue: If q is empty, escaped_q empty, param '%'. Good.

Potential issue: `dict(row)` for sqlite3.Row. In Python 3.8? `dict(row)` works? Let's confirm

## Response


## Metadata
- **Model**: qwen3.8-27b-int4-16k
- **Elapsed**: 342402ms
- **Prompt tokens**: 1120
- **Completion tokens**: 16384
- **Total tokens**: 17504
- **Gen tok/s**: 47.8
