### CRUD Handler

The CRUD handler provides an API for managing entities with persistent file-based storage.

#### Configuration

Add a CRUD handler to your server config:
```nginx
server {
  listen 8080;
  
  location /api {
    handler crud;
    data_path /path/to/data/storage;
  }
}
```

**Configuration directives:**
- `handler crud`: Specifies the CRUD handler
- `data_path`: Directory path where entity data will be stored (required)

#### API Endpoints

The CRUD handler supports standard RESTful operations:

##### Create Entity
```bash
POST /api/{EntityType}
Content-Type: application/json

{"field1": "value1", "field2": "value2"}
```
**Response:** `{"id": 1}`

**Example:**
```bash
curl -X POST http://localhost:8080/api/TestEntity \
  -H "Content-Type: application/json" \
  -d '{"name": "test_item", "value": "example_data"}'
```

##### Retrieve Entity
```bash
GET /api/{EntityType}/{id}
```
**Response:** JSON object with entity data

**Example:**
```bash
curl http://localhost:8080/api/TestEntity/1
# Returns: {"name": "test_item", "value": "example_data"}
```

##### List All Entities
```bash
GET /api/{EntityType}
```
**Response:** Array of entity IDs `[1, 2, 3]`

**Example:**
```bash
curl http://localhost:8080/api/TestEntity
# Returns: [1, 2, 3]
```

##### Update Entity
```bash
PUT /api/{EntityType}/{id}
Content-Type: application/json

{"field1": "updated_value"}
```
**Response:** `{"id": 1}`

**Example:**
```bash
curl -X PUT http://localhost:8080/api/TestEntity/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "updated_item", "value": "new_data"}'
```

##### Delete Entity
```bash
DELETE /api/{EntityType}/{id}
```
**Response:** `{"id": 1}`

**Example:**
```bash
curl -X DELETE http://localhost:8080/api/TestEntity/1
```

#### Storage Format

Entities are stored as files in the configured `data_path`:
```
data_path/
  └── EntityType/
      ├── 1
      ├── 2
      └── 3
```

- Each entity type gets its own subdirectory
- Entity files are named by their ID (no extension)
- Files contain raw data as provided in POST/PUT requests
- IDs are automatically assigned sequentially

#### Edge Case Behaviors

##### DELETE Behavior
- **DELETE on existing entity**: `200 OK`
- **DELETE on non-existent entity**: `200 OK` (idempotent)
- **Rationale**: Idempotent DELETE ensures desired end state is achieved, consistent with idempotent PUT

##### PUT Behavior
- **PUT supports idempotent creation**: Can create entities with predetermined IDs
- **PUT to non-existent ID**: Creates entity with that ID
- **PUT to existing entity**: Overwrites existing data
- **Rationale**: Follows REST idempotency principles

##### ID Generation
- IDs auto-generated sequentially starting from 1
- Each entity type has independent ID space
- IDs never reused after deletion
- Concurrent requests generate unique IDs

##### Entity Type Isolation
- Different entity types maintain separate ID spaces
- Example: `GET /api/Books/1` and `GET /api/Shoes/1` return different entities
- Entity types are case-sensitive

##### Data Storage
- No validation performed on stored data
- Data stored as-is in filesystem
- Clients responsible for data integrity

#### Testing

See CRUD tests in `tests/integration/test_cases.py` for complete examples of CRUD operations.

Test config example:
```python
@IntegrationTestRunner.register_test(TestConfig(
    config_file="../tests/integration/test_configs/crud_test_config",
    port=8080
))
def test_crud_operations(server: ServerFixture, helpers: TestHelpers):
    # Create
    response = helpers.post("/api/User", 
        data='{"name": "John"}',
        headers={"Content-Type": "application/json"})
    entity_id = response.json()["id"]
    
    # Retrieve
    response = helpers.get(f"/api/User/{entity_id}")
    
    # Update
    response = helpers.put(f"/api/User/{entity_id}",
        data='{"name": "Jane"}',
        headers={"Content-Type": "application/json"})
    
    # Delete
    response = helpers.delete(f"/api/User/{entity_id}")
```

#### Complete Usage Example

```bash
# 1. Create a new book
curl -X POST http://localhost:8080/api/Books \
  -H "Content-Type: application/json" \
  -d '{"title": "1984", "author": "George Orwell"}'
# Response: {"id": 1}

# 2. Retrieve the book
curl http://localhost:8080/api/Books/1
# Response: {"title": "1984", "author": "George Orwell"}

# 3. Update the book
curl -X PUT http://localhost:8080/api/Books/1 \
  -H "Content-Type: application/json" \
  -d '{"title": "1984", "author": "George Orwell", "year": 1949}'
# Response: {"id": 1}

# 4. Create book with predetermined ID
curl -X PUT http://localhost:8080/api/Books/42 \
  -H "Content-Type: application/json" \
  -d '{"title": "Brave New World", "author": "Aldous Huxley"}'
# Response: {"id": 42}

# 5. List all books
curl http://localhost:8080/api/Books
# Response: [1, 42]

# 6. Delete a book
curl -X DELETE http://localhost:8080/api/Books/1
# Response: {"id": 1}

# 7. Delete again (idempotent - still succeeds)
curl -X DELETE http://localhost:8080/api/Books/1
# Response: {"id": 1}

# 8. Verify deletion with GET
curl http://localhost:8080/api/Books/1
# Response: 404 Not Found
```

#### Troubleshooting

**Issue: "Entity not found" on GET**
- Verify entity was created successfully (check ID from POST response)
- Ensure correct entity type name (case-sensitive)
- Check `data_path` directory is accessible

**Issue: IDs not sequential**
- Normal after deletions - IDs never reused
- IDs increment from highest existing ID, not count of entities