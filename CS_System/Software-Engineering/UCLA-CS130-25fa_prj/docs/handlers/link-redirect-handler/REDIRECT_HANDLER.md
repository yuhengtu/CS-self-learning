## Redirect Handling

The redirect handler resolves short codes into their corresponding long URLs and issues HTTP redirects to route clients to the intended destination. This section defines the endpoint, expected behavior, and example responses.

---

### **Endpoint**
```GET /l/<short-code>```
Where `<short-code>` is the unique identifier previously generated during link creation.

---

### **Behavior**

1. The server receives a GET request containing a short code.
2. The redirect handler performs a lookup in persistent storage to find the long URL associated with the short code.
3. If a valid mapping is found:
   - The server issues an HTTP **302 Found** response.
   - The long URL is placed in the `Location` header.
   - The client (e.g., web browser) automatically follows the redirect.
4. If no mapping is found:
   - The server returns an HTTP **404 Not Found** or an equivalent error response.

This mechanism ensures short links remain concise while still routing seamlessly to full-length destination URLs.

---

### **Usage**

Clients or browsers can resolve a short link simply by performing:
```GET /l/<short-code>```


No request body or additional headers are required.

Example client accesses include:

- Directly entering the short URL into a browser
- Clicking a hyperlink containing the short code path
- Programmatic retrieval using HTTP clients (curl, fetch, etc.)

---

### **Successful Redirect Example**

After requesting a valid shortcode, our server will return an HTTP response like:

```
HTTP/1.1 302 Found
Location: https://www.example-destination.com/some/long/path
```
