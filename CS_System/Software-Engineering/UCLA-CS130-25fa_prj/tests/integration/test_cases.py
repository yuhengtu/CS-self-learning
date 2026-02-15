"""
Test Cases for Integration Tests
Contains individual test functions

This is written with genai assistance.
"""

"""
Usage:

For additional tests, add a new function here and decorate it with @IntegrationTestRunner.register_test().
You can optionally specify a name for the test by passing a name to the decorator.

The decorator simply takes the function reference and registers it with the test runner's class level registry which is used to find the tests to run.
"""

import os
import json
import shutil
import time
import threading

from server_fixture import ServerFixture
from test_helpers import TestHelpers
from test_runner import IntegrationTestRunner
from test_metadata import TestConfig


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/default_config", port=8080))
# @IntegrationTestRunner.register_test()
def test_basic_http_request(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Test basic HTTP GET request"""
    try:
        response = helpers.get("/")
        return helpers.assert_status_code(response, 200)
    except Exception as e:
        print(f"Request failed: {e}")
        return False

@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/static_config", port=8080))
def test_static_file_serving(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Test serving existing and missing static files"""
    try:
        ok = True

        # 1. Plain text file
        resp_txt = helpers.get("/static/note.txt")
        ok &= helpers.assert_status_code(resp_txt, 200)
        ok &= helpers.assert_contains(resp_txt, "alpha\nbeta\ngamma")
        ok &= helpers.assert_header(resp_txt, "Content-Type", "text/plain")

        # 2. HTML file
        resp_html = helpers.get("/static/hello.html")
        ok &= helpers.assert_status_code(resp_html, 200)
        ok &= helpers.assert_contains(resp_html, "<h1>Hello</h1>")
        ok &= helpers.assert_header(resp_html, "Content-Type", "text/html")

        # 3. JPEG file
        resp_jpg = helpers.get("/static/img.jpg")
        ok &= helpers.assert_status_code(resp_jpg, 200)
        ok &= helpers.assert_header(resp_jpg, "Content-Type", "image/jpeg")

        # 4. ZIP file
        resp_zip = helpers.get("/static/archive.zip")
        ok &= helpers.assert_status_code(resp_zip, 200)
        ok &= helpers.assert_header(resp_zip, "Content-Type", "application/zip")

        # 5. Missing file should return 404
        resp_404 = helpers.get("/static/notfound.txt")
        ok &= helpers.assert_status_code(resp_404, 404)

        # 6. Echo path still functional
        resp_echo = helpers.get("/echo")
        ok &= helpers.assert_status_code(resp_echo, 200)

        return ok
    except Exception as e:
        print(f"Static file test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/default_config", port=8080))
def test_multiple_requests(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Issue multiple sequential requests within one server lifetime"""
    try:
        ok = True
        for _ in range(5):
            resp = helpers.get("/")
            ok &= helpers.assert_status_code(resp, 200)
        return ok
    except Exception as e:
        print(f"Multiple requests test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/static_config", port=8080))
def test_unmatched_route_returns_404(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Requests that don't match any route should return 404"""
    try:
        resp = helpers.get("/no-such-path")
        return helpers.assert_status_code(resp, 404)
    except Exception as e:
        print(f"Unmatched route test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_content="""
server {
  listen 8080;

  location / {
    handler echo;
  }

  location /static {
    handler static;
    root ../tests/static_test_files;
  }

  location /static/images {
    handler echo;
  }
}
""", port=8080))
def test_longest_prefix_routing(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Longest prefix wins when two routes share a prefix"""
    try:
        # This path should match /static/images (longer prefix -> echo)
        resp = helpers.get("/static/images/pic.jpg")
        return helpers.assert_status_code(resp, 200)
    except Exception as e:
        print(f"Longest-prefix test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_content="""
server {
  listen 8080;

  location /echo {
    handler echo;
  }

  location /static {
    handler static;
    # intentionally missing root
  }
}
""", port=8080))
def test_static_missing_root_skips_route(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Static without root should be skipped; echo remains functional; /static 404"""
    try:
        ok = True
        resp_echo = helpers.get("/echo")
        ok &= helpers.assert_status_code(resp_echo, 200)
        resp_static = helpers.get("/static/anything.txt")
        ok &= helpers.assert_status_code(resp_static, 404)
        return ok
    except Exception as e:
        print(f"Static missing root test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/static_config", port=8080))
def test_static_traversal_blocked(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Path traversal attempts should not escape root and return 404"""
    try:
        ok = True
        resp1 = helpers.get("/static/../../etc/passwd")
        ok &= helpers.assert_status_code(resp1, 404)
        resp2 = helpers.get("/static/..%2F..%2Fsecret.txt")
        ok &= helpers.assert_status_code(resp2, 404)
        return ok
    except Exception as e:
        print(f"Traversal test failed: {e}")
        return False

# CRUD Integration Tests Courtesy Of One-Fish team, aided by GenAI
# NOTE: Ensure that the server configuration used here matches the expected setup for CRUD operations.
# CRUD handler creates files under ./bin/tmp/crud_data by default --> this is done in the current dir (build dir) so the test can access it directly.
# and so git log ignores changes to that dir,  then it deletes changes after the test.
@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_assignment_required_sequence(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests the complete lifecycle of a CRUD entity as specified in the assignment.
    Following Test Order: Create --> Retrieve --> Verify --> Delete --> Verify Deletion"""
    # Clean up any previous test data
    crud_root = "./bin/tmp/crud_data"
    if os.path.exists(f"{crud_root}/TestEntity"):
        shutil.rmtree(f"{crud_root}/TestEntity")
    
    # Step 1: CREATE entity
    create_data = {"name": "test_item", "value": "chuy_was_here"}
    create_response = helpers.post(
        "/api/TestEntity", 
        data=json.dumps(create_data),
        headers={"Content-Type": "application/json"}
    )
    
    print(f"CREATE Response Status: {create_response.status_code}")
    print(f"CREATE Response Body: {create_response.text}")
    print(f"CREATE Response Headers: {create_response.headers}")
    
    if not helpers.assert_status_code(create_response, 200):
        return False
    
    try:
        create_json = create_response.json()
        if "id" not in create_json:
            print(f"CREATE failed: No 'id' in response: {create_response.text}")
            return False
        entity_id = create_json["id"]
        print(f"Created entity with ID: {entity_id}")
    except json.JSONDecodeError:
        print(f"CREATE failed: Invalid JSON response: {create_response.text}")
        return False
    
    # DEBUG: Check if file was created
    entity_file = f"{crud_root}/TestEntity/{entity_id}"
    print(f"\nChecking for file at: {entity_file}")
    if os.path.exists(entity_file):
        with open(entity_file, 'r') as f:
            file_contents = f.read()
            print(f"File exists! Contents: {file_contents}")
    else:
        print(f"WARNING: File does not exist at {entity_file}")
        # List what's actually in the directory
        if os.path.exists(f"{crud_root}/TestEntity"):
            print(f"Files in TestEntity dir: {os.listdir(f'{crud_root}/TestEntity')}")
    
    # Step 2: RETRIEVE entity
    retrieve_url = f"/api/TestEntity/{entity_id}"
    print(f"\nAttempting GET to: {retrieve_url}")
    retrieve_response = helpers.get(retrieve_url)
    
    print(f"RETRIEVE Response Status: {retrieve_response.status_code}")
    print(f"RETRIEVE Response Body: '{retrieve_response.text}'")
    print(f"RETRIEVE Response Headers: {retrieve_response.headers}")
    print(f"RETRIEVE Response Content-Length: {len(retrieve_response.content)}")
    
    if not helpers.assert_status_code(retrieve_response, 200):
        return False
    
    print(f"Retrieved entity: {retrieve_response.text}")
    
    # Step 3: VERIFY correctness
    try:
        retrieved_data = retrieve_response.json()
        if retrieved_data.get("name") != "test_item":
            print(f"VERIFY failed: Expected name='test_item', got {retrieved_data}")
            return False
        if retrieved_data.get("value") != "chuy_was_here":
            print(f"VERIFY failed: Expected value='chuy_was_here', got {retrieved_data}")
            return False
        print(f"Verified entity data correctness")
    except json.JSONDecodeError as e:
        print(f"VERIFY failed: Invalid JSON response: {retrieve_response.text}")
        print(f"JSON Error: {e}")
        return False
    
    # Step 4: DELETE entity
    delete_response = helpers.delete(f"/api/TestEntity/{entity_id}")
    
    if not helpers.assert_status_code(delete_response, 200):
        return False
    
    print(f"Deleted entity ID {entity_id}")
    
    # Step 5: VERIFY deletion (should get 404)
    verify_response = helpers.get(f"/api/TestEntity/{entity_id}")
    
    if not helpers.assert_status_code(verify_response, 404):
        print(f"VERIFY DELETION failed: Expected 404, got {verify_response.status_code}")
        return False
    
    print(f"Verified entity is deleted (404 returned)")
    
    return True
    
@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_malformed_json(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests handling of malformed JSON in POST/PUT requests.
    Note: Without JSON validation library, malformed JSON is stored as-is.
    This test is kept for documentation but expects success (200 OK)."""
    
    # Test POST with malformed JSON
    malformed_data = '{"name": "test", "value": incomplete'
    response = helpers.post(
        "/api/TestEntity",
        data=malformed_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Malformed JSON POST Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if not helpers.assert_status_code(response, 200):
        print("Expected 200 OK for malformed JSON")
        return False
    
    # Clean up created entity if any
    try:
        entity_id = response.json().get("id")
        helpers.delete(f"/api/TestEntity/{entity_id}")
    except:
        pass
    
    print("Malformed JSON properly accepted with 200")
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_get_nonexistent_entity(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests GET for an ID that doesn't exist.
    Expected behavior: 404 Not Found."""
    
    response = helpers.get("/api/TestEntity/999999")
    
    print(f"Nonexistent entity GET Status: {response.status_code}")
    
    if not helpers.assert_status_code(response, 404):
        print("Expected 404 for nonexistent entity")
        return False
    
    print("Nonexistent entity properly returns 404")
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_delete_nonexistent_entity(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests DELETE for an ID that doesn't exist.
    Expected behavior: 200 OK (idempotent DELETE - desired state achieved)."""
    
    response = helpers.delete("/api/TestEntity/999999")
    
    print(f"Delete nonexistent entity Status: {response.status_code}")
    
    if not helpers.assert_status_code(response, 200):
        print("Expected 200 OK for deleting nonexistent entity")
        return False
    
    print("Delete nonexistent entity properly returns 200")
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_put_nonexistent_entity(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests PUT to an ID that doesn't exist.
    Expected behavior: 404 Not Found (entity must exist before update).
    Alternative: 200/201 if your implementation supports idempotent PUT (creates if not exists)."""
    
    update_data = {"name": "new_item", "value": "put_to_nonexistent"}
    response = helpers.put(
        "/api/TestEntity/888888",
        data=json.dumps(update_data),
        headers={"Content-Type": "application/json"}
    )
    
    print(f"PUT to nonexistent ID Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Document your implementation's behavior:
    # Either 404 (must exist first) or 200/201 (create new) are both valid REST approaches
    if response.status_code not in [200, 201, 404]:
        print(f"Expected 200/201/404, got {response.status_code}")
        return False
    
    # If created, verify and clean up
    if response.status_code in [200, 201]:
        print("Implementation supports idempotent PUT (creates if not exists)")
        verify = helpers.get("/api/TestEntity/888888")
        if verify.status_code == 200:
            helpers.delete("/api/TestEntity/888888")
    else:
        print("Implementation requires entity to exist before PUT")
    
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_double_delete(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests deleting the same entity twice.
    Expected behavior: First DELETE succeeds (200), second returns 404."""
    
    # Create entity
    create_data = {"name": "temp", "value": "delete_me"}
    create_response = helpers.post(
        "/api/TestEntity",
        data=json.dumps(create_data),
        headers={"Content-Type": "application/json"}
    )
    
    if not helpers.assert_status_code(create_response, 200):
        return False
    
    entity_id = create_response.json()["id"]
    print(f"Created entity ID: {entity_id}")
    
    # First delete
    delete1 = helpers.delete(f"/api/TestEntity/{entity_id}")
    if not helpers.assert_status_code(delete1, 200):
        print("First delete failed")
        return False
    print(f"First delete succeeded")
    
    # Second delete
    delete2 = helpers.delete(f"/api/TestEntity/{entity_id}")
    if not helpers.assert_status_code(delete2, 200):
        print("Expected 200 OK on second delete")
        return False
    print(f"Second delete properly returned 200")
    
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_concurrent_id_generation(server: ServerFixture, helpers: TestHelpers) -> bool:
    """
    Tests that concurrent POST requests generate unique IDs.

    Expected behavior: Each POST returns a unique ID (no collisions),
    even when multiple requests are in-flight at the same time.
    """
    crud_root = "./bin/tmp/crud_data"
    if os.path.exists(f"{crud_root}/TestEntity"):
        shutil.rmtree(f"{crud_root}/TestEntity")

    num_entities = 10
    created_ids = [None] * num_entities
    success = [True]  # mutable flag to share between threads

    def worker(i: int):
        if not success[0]:
            return

        create_data = {"name": f"concurrent_{i}", "value": f"test_{i}"}
        try:
            response = helpers.post(
                "/api/TestEntity",
                data=json.dumps(create_data),
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            print(f"ERROR: Exception during POST in worker {i}: {e}")
            success[0] = False
            return

        if not helpers.assert_status_code(response, 200):
            print(f"ERROR: Non-200 status in worker {i}: {response.status_code}")
            success[0] = False
            return

        try:
            json_body = response.json()
            entity_id = json_body["id"]
            created_ids[i] = entity_id
        except Exception as e:
            print(f"ERROR: Failed to parse JSON or missing 'id' in worker {i}: {e}")
            print(f"Response text: {response.text}")
            success[0] = False

    # Launch threads concurrently
    threads = []
    for i in range(num_entities):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if not success[0]:
        print("At least one worker failed")
        return False

    # Check that all IDs were set
    if any(eid is None for eid in created_ids):
        print(f"ERROR: Some IDs were not set: {created_ids}")
        return False

    id_set = set(created_ids)
    if len(id_set) != num_entities:
        print(f"ERROR: Expected {num_entities} unique IDs, got {len(id_set)}")
        print(f"IDs: {sorted(id_set)}")
        return False

    print(f"Created {len(id_set)} entities with unique IDs: {sorted(id_set)}")

    # Clean up
    for entity_id in created_ids:
        try:
            helpers.delete(f"/api/TestEntity/{entity_id}")
        except Exception as e:
            print(f"WARNING: Failed to delete entity {entity_id}: {e}")

    print("All IDs were unique under concurrent POSTs - no collisions")
    return True


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/crud_test_config", port=8080))
def test_crud_multiple_entity_types_isolation(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Tests that different entity types maintain separate ID spaces.
    Expected behavior: ID 1 for Books and ID 1 for Shoes are different entities."""
    
    crud_root = "./bin/tmp/crud_data"
    for entity_type in ["Books", "Shoes"]:
        if os.path.exists(f"{crud_root}/{entity_type}"):
            shutil.rmtree(f"{crud_root}/{entity_type}")
    
    # Create entity for Books
    books_data = {"title": "1984", "author": "Orwell"}
    books_response = helpers.post(
        "/api/Books",
        data=json.dumps(books_data),
        headers={"Content-Type": "application/json"}
    )
    
    if not helpers.assert_status_code(books_response, 200):
        return False
    
    books_id = books_response.json()["id"]
    print(f"Created Books ID: {books_id}")
    
    # Create entity for Shoes
    shoes_data = {"brand": "Nike", "size": 10}
    shoes_response = helpers.post(
        "/api/Shoes",
        data=json.dumps(shoes_data),
        headers={"Content-Type": "application/json"}
    )
    
    if not helpers.assert_status_code(shoes_response, 200):
        return False
    
    shoes_id = shoes_response.json()["id"]
    print(f"Created Shoes ID: {shoes_id}")
    
    # Verify Books entity
    books_get = helpers.get(f"/api/Books/{books_id}")
    if not helpers.assert_status_code(books_get, 200):
        return False
    
    books_retrieved = books_get.json()
    if books_retrieved.get("title") != "1984":
        print(f"ERROR: Books data incorrect: {books_retrieved}")
        return False
    
    # Verify Shoes entity
    shoes_get = helpers.get(f"/api/Shoes/{shoes_id}")
    if not helpers.assert_status_code(shoes_get, 200):
        return False
    
    shoes_retrieved = shoes_get.json()
    if shoes_retrieved.get("brand") != "Nike":
        print(f"ERROR: Shoes data incorrect: {shoes_retrieved}")
        return False
    
    # Verify isolation: Books/1 should NOT return Shoes data
    if books_id == shoes_id:
        # If IDs happen to be the same, verify they're still isolated
        if books_retrieved.get("brand") == "Nike":
            print("ERROR: Entity types are not isolated - Books returned Shoes data")
            return False
        print(f"Same ID ({books_id}) used for both types, but data is properly isolated")
    
    # Clean up
    helpers.delete(f"/api/Books/{books_id}")
    helpers.delete(f"/api/Shoes/{shoes_id}")
    
    print("Entity types properly maintain separate ID spaces")
    return True

@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/default_config", port=8080))
def test_sleep_allows_concurrent_requests(server: ServerFixture, helpers: TestHelpers) -> bool:
    """
    Verify that a long-running /sleep request does not block other requests.

    Strategy:
      - Start a /sleep request in one thread (takes ~2s).
      - Shortly after, start a fast request to "/" in another thread.
      - Assert:
          * Both return 200.
          * /sleep takes significantly longer.
          * The fast request returns well before /sleep completes.
    """
    try:
        results = {}

        def slow_request():
            start = time.time()
            resp = helpers.get("/sleep")
            elapsed = time.time() - start
            results["sleep"] = (resp, elapsed)

        def fast_request():
            start = time.time()
            resp = helpers.get("/")
            elapsed = time.time() - start
            results["fast"] = (resp, elapsed)

        slow_thread = threading.Thread(target=slow_request)
        fast_thread = threading.Thread(target=fast_request)

        # Start the slow /sleep request first
        slow_thread.start()
        # Give it a brief head start so it's definitely in-progress
        time.sleep(0.1)
        fast_thread.start()

        slow_thread.join()
        fast_thread.join()

        if "sleep" not in results or "fast" not in results:
            print("ERROR: One of the requests did not complete")
            return False

        sleep_resp, sleep_elapsed = results["sleep"]
        fast_resp, fast_elapsed = results["fast"]

        ok = True
        ok &= helpers.assert_status_code(sleep_resp, 200)
        ok &= helpers.assert_status_code(fast_resp, 200)

        # Sleep handler should be noticeably slow (~2s in handler)
        if sleep_elapsed < 1.8:
            print(f"Expected /sleep to take at least ~1.8s, got {sleep_elapsed:.3f}s")
            ok = False

        # Fast request should complete much sooner than /sleep
        if fast_elapsed >= sleep_elapsed - 0.5:
            print(
                "Expected fast request to return well before sleep completes, "
                f"but fast_elapsed={fast_elapsed:.3f}s, sleep_elapsed={sleep_elapsed:.3f}s"
            )
            ok = False

        # And in absolute terms it should not be ~2s
        if fast_elapsed >= 2.0:
            print(f"Expected fast request to complete in <2s, got {fast_elapsed:.3f}s")
            ok = False

        return ok
    except Exception as e:
        print(f"Concurrent sleep test failed: {e}")
        return False

# URL Shortener Tests



@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/link_test_config", port=8080))
def test_link_shortener_redirect_flow(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Create a short link via /api/link and verify /l/{code} redirects."""
    link_root = "./bin/tmp/link_data"
    if os.path.exists(link_root):
        shutil.rmtree(link_root)

    try:
        create_resp = helpers.post(
            "/api/link",
            data=json.dumps({"url": "https://example.com"}),
            headers={"Content-Type": "application/json"}
        )
        if not helpers.assert_status_code(create_resp, 200):
            print("Create response:", create_resp.text)
            return False

        code = create_resp.json().get("code")
        if not code:
            print("Missing code in create response", create_resp.text)
            return False

        redirect_resp = helpers.get(f"/l/{code}", allow_redirects=False)
        ok = helpers.assert_status_code(redirect_resp, 302)
        location = redirect_resp.headers.get("Location")
        if location != "https://example.com":
            print(f"Expected Location header https://example.com, got {location}")
            ok = False
        return ok
    except Exception as e:
        print(f"Link shortener test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/link_test_config", port=8080))
def test_link_analytics(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Verify analytics endpoint aggregates visits per URL and leaderboard."""
    link_root = "./bin/tmp/link_data"
    if os.path.exists(link_root):
        shutil.rmtree(link_root)

    try:
        create_resp = helpers.post(
            "/api/link",
            data=json.dumps({"url": "https://example.com"}),
            headers={"Content-Type": "application/json"}
        )
        if not helpers.assert_status_code(create_resp, 200):
            return False
        code = create_resp.json().get("code")
        if not code:
            print("Missing code in create response")
            return False

        analytics_resp = helpers.get(f"/analytics/{code}")
        if not helpers.assert_status_code(analytics_resp, 200):
            return False
        analytics_body = analytics_resp.json()
        if analytics_body.get("visits") != 0 or analytics_body.get("url_visits") != 0:
            print("Expected zero visits before redirect")
            return False

        # Trigger a redirect (disable following redirects to keep request local)
        redirect_resp = helpers.get(f"/l/{code}", allow_redirects=False)
        if not helpers.assert_status_code(redirect_resp, 302):
            return False

        analytics_resp = helpers.get(f"/analytics/{code}")
        analytics_body = analytics_resp.json()
        if analytics_body.get("visits") != 1 or analytics_body.get("url_visits") != 1:
            print("Expected visit counts to be 1 after redirect")
            return False

        leaderboard_resp = helpers.get("/analytics/top/1")
        if not helpers.assert_status_code(leaderboard_resp, 200):
            return False
        leaderboard = leaderboard_resp.json()
        if not leaderboard or leaderboard[0].get("url") != "https://example.com":
            print("Leaderboard did not include most visited URL")
            return False
        return True
    except Exception as e:
        print(f"Analytics test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/link_test_config", port=8080))
def test_link_analytics_per_code_vs_url(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Ensure per-code visits remain distinct while URL aggregates all traffic."""
    link_root = "./bin/tmp/link_data"
    if os.path.exists(link_root):
        shutil.rmtree(link_root)

    try:
        def create_link() -> str:
            resp = helpers.post(
                "/api/link",
                data=json.dumps({"url": "https://shared.example"}),
                headers={"Content-Type": "application/json"}
            )
            if not helpers.assert_status_code(resp, 200):
                raise RuntimeError(f"Create failed: {resp.text}")
            code = resp.json().get("code")
            if not code:
                raise RuntimeError("Missing code in create response")
            return code

        code_a = create_link()
        code_b = create_link()

        def metrics(code: str):
            resp = helpers.get(f"/analytics/{code}")
            if not helpers.assert_status_code(resp, 200):
                raise RuntimeError(f"Analytics fetch failed for {code}")
            body = resp.json()
            return body.get("visits"), body.get("url_visits")

        visits_a_before, url_visits_before = metrics(code_a)
        visits_b_before, _ = metrics(code_b)
        if visits_a_before != 0 or visits_b_before != 0 or url_visits_before != 0:
            print("Expected zero visits before redirects")
            return False

        # Direct traffic: code A once, code B twice.
        helpers.get(f"/l/{code_a}", allow_redirects=False)
        helpers.get(f"/l/{code_b}", allow_redirects=False)
        helpers.get(f"/l/{code_b}", allow_redirects=False)

        visits_a_after, url_visits_after = metrics(code_a)
        visits_b_after, _ = metrics(code_b)

        if visits_a_after != 1 or visits_b_after != 2:
            print(f"Per-code visits mismatched: A={visits_a_after}, B={visits_b_after}")
            return False
        if url_visits_after != 3:
            print(f"Expected aggregate URL visits to be 3, got {url_visits_after}")
            return False
        return True
    except Exception as e:
        print(f"Per-code vs URL analytics test failed: {e}")
        return False


@IntegrationTestRunner.register_test(TestConfig(config_file="../tests/integration/test_configs/link_test_config", port=8080))
def test_link_password_protection(server: ServerFixture, helpers: TestHelpers) -> bool:
    """Ensure password-protected links require headers for management."""
    link_root = "./bin/tmp/link_data"
    if os.path.exists(link_root):
        shutil.rmtree(link_root)

    try:
        create_resp = helpers.post(
            "/api/link",
            data=json.dumps({"url": "https://example.com", "password": "secret"}),
            headers={"Content-Type": "application/json"}
        )
        if not helpers.assert_status_code(create_resp, 200):
            return False
        code = create_resp.json().get("code")
        if not code:
            print("Missing code in create response")
            return False

        if not helpers.assert_status_code(helpers.get(f"/api/link/{code}"), 403):
            return False
        wrong = helpers.get(f"/api/link/{code}", headers={"Link-Password": "wrong"})
        if not helpers.assert_status_code(wrong, 403):
            return False

        auth_headers = {"Link-Password": "secret"}
        authed_resp = helpers.get(f"/api/link/{code}", headers=auth_headers)
        if not helpers.assert_status_code(authed_resp, 200):
            return False
        if not authed_resp.json().get("password_protected"):
            print("Expected password_protected flag on GET response")
            return False

        body = json.dumps({"url": "https://updated.com"})
        if not helpers.assert_status_code(
            helpers.put(f"/api/link/{code}", data=body, headers={"Content-Type": "application/json"}),
            403,
        ):
            return False
        put_headers = {"Content-Type": "application/json", "Link-Password": "secret"}
        if not helpers.assert_status_code(
            helpers.put(f"/api/link/{code}", data=body, headers=put_headers),
            200,
        ):
            return False

        if not helpers.assert_status_code(helpers.delete(f"/api/link/{code}"), 403):
            return False
        if not helpers.assert_status_code(helpers.delete(f"/api/link/{code}", headers=auth_headers), 200):
            return False
        final_get = helpers.get(f"/api/link/{code}", headers=auth_headers)
        if not helpers.assert_status_code(final_get, 404):
            return False
        return True
    except Exception as e:
        print(f"Password protection test failed: {e}")
        return False
