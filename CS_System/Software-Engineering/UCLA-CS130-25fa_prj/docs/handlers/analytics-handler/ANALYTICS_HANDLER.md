### Analytics Handler

The analytics handler surfaces visit metrics gathered by the redirect handler. Each redirect bumps **two** counters:
- a per-short-code visit total stored with the `LinkRecord`
- an aggregate per-destination URL total in `url_stats.json`

#### Configuration

```
location /analytics {
  handler analytics;
  data_path /path/to/link/storage;
}
```

Uses the same `data_path` as the link manage/redirect handlers.

#### Endpoints

```text
GET /analytics/<code>
```
Returns JSON with:
- `code`: the requested short code
- `url`: the destination it maps to
- `visits`: how many times this specific short code has been used
- `url_visits`: the aggregate visit count for the destination URL (across every short code pointing to it)

```text
GET /analytics/top/<N>
```
Returns a JSON array of the top `N` most visited destination URLs sorted by visits (descending). Each entry includes the `url` and `visits`.

All other requests return `400 Bad Request`.
