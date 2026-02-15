### Link Management Handler

The link management handler provides a RESTful API for creating, reading, updating, and deleting shortened URLs. It supports optional password protection to secure management operations for individual short links.

Note: written by GenAI

#### Configuration

```
location /api/link {
  handler link_manage;
  data_path /path/to/link/storage;
}
```

The `data_path` should be the same as used for the link redirect and analytics handlers to ensure all components operate on the same underlying storage.

#### Endpoints

##### Create a Short Link

```text
POST /api/link
Content-Type: application/json

{
  "url": "https://example.com/some/long/path",
  "password": "optional-secret"
}
```

**Response (200 OK):**
```json
{"code":"abc123"}
```

**Error Responses:**
- `400 Bad Request` - Missing or invalid `url`, or empty `password` string
- `500 Internal Server Error` - Filesystem or storage error

**Notes:**
- The `password` field is optional. If provided, it must be a non-empty string.
- When a password is set, all subsequent management operations (GET, PUT, DELETE) on this short link require authentication via the `Link-Password` header.
- Passwords are never stored in plaintext; they are salted and hashed using SHA256.

##### Read Link Information

```text
GET /api/link/<code>
Link-Password: optional-password-if-protected
```

**Response (200 OK):**
```json
{
  "code": "abc123",
  "url": "https://example.com/some/long/path",
  "visits": 42
}
```

**Error Responses:**
- `400 Bad Request` - Invalid code format
- `403 Forbidden` - Missing or incorrect password for a protected link
- `404 Not Found` - Code does not exist
- `500 Internal Server Error` - Filesystem or storage error

**Notes:**
- If the link was created with a password, you must provide the correct password in the `Link-Password` header.
- The response includes the current visit count for this short code.
- The `password_hash` and `password_salt` fields are never returned in the response.

##### Update a Short Link

```text
PUT /api/link/<code>
Content-Type: application/json
Link-Password: optional-password-if-protected

{
  "url": "https://newexample.com/updated/path"
}
```

**Response (200 OK):**
```json
{"code":"abc123"}
```

**Error Responses:**
- `400 Bad Request` - Invalid code format or missing/invalid `url`
- `403 Forbidden` - Missing or incorrect password for a protected link
- `404 Not Found` - Code does not exist
- `500 Internal Server Error` - Filesystem or storage error

**Notes:**
- Only the destination URL can be updated; the password and code cannot be changed.
- Visit counts are preserved across updates.

##### Delete a Short Link

```text
DELETE /api/link/<code>
Link-Password: optional-password-if-protected
```

**Response (200 OK):**
```json
{"code":"abc123"}
```

**Error Responses:**
- `400 Bad Request` - Invalid code format
- `403 Forbidden` - Missing or incorrect password for a protected link
- `404 Not Found` - Code does not exist (deletion is considered successful)
- `500 Internal Server Error` - Filesystem or storage error

**Notes:**
- Deleting a link that doesn't exist returns success (idempotent operation).
- Once deleted, the short code becomes immediately unavailable for redirects.

#### Password Protection

Password protection is implemented using industry-standard cryptographic practices:

1. **Salt Generation**: When a password is provided during link creation, a 16-byte random salt is generated using `std::random_device` and `std::mt19937`.

2. **Hashing**: The password is hashed using SHA256 with the salt prepended:
   ```
   hash = SHA256(salt + password)
   ```
   Both the salt and hash are hex-encoded and stored with the link record.

3. **Verification**: For protected operations (GET, PUT, DELETE), the handler:
   - Retrieves the stored salt from the link record
   - Hashes the provided password with the stored salt
   - Compares the computed hash with the stored hash
   - Returns `403 Forbidden` if the hashes don't match or if no password is provided

4. **Storage**: Only the salt and hash are stored in the link record JSON file. The plaintext password is never persisted.

5. **Header Format**: The password must be provided in the `Link-Password` HTTP header for all management operations on protected links.

#### Code Validation

Short codes must match the pattern validated by `LinkManagerInterface::IsValidCode()`. Codes are generated automatically as base62-encoded sequential integers starting from a seed value, ensuring uniqueness and URL-safe characters.

#### Concurrency

All operations acquire a mutex lock in the `LinkManager` to ensure thread-safe access to the filesystem-based storage. Multiple concurrent requests are handled safely through in-process synchronization.

#### Integration with Other Handlers

The link management handler should be configured with the same `data_path` as:
- **Link Redirect Handler** (e.g., `/l`) - serves the redirects for short codes
- **Analytics Handler** (e.g., `/analytics`) - provides visit statistics

This ensures all handlers operate on the same underlying storage and see consistent state.

