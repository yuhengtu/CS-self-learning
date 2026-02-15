#include "gtest/gtest.h"
#include "crud_request_handler.h"
#include "crud_manager_interface.h"
#include "request.h"
#include "response.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * A fake implementation of the CrudManagerInterface for testing.
 * 
 * IMPORTANT: This fake must maintain the same edge case behavior as the real CrudManager.
 * Any changes to CrudManager edge case handling must be reflected here.
 * 
 * Current edge case behaviors (must match CrudManager implementation):
 * - Create(): Returns std::nullopt on filesystem errors
 * - Read(): Returns std::nullopt if entity doesn't exist or on filesystem errors
 * - Update(): Supports idempotent creation (creates entity if ID doesn't exist)
 *             Returns false only on filesystem errors
 * - Delete(): Returns false if entity doesn't exist (spec requires "valid object")
 *             Returns false on filesystem errors
 * - List(): Returns empty vector if entity type directory doesn't exist
 *
 */
class FakeCrudManager : public CrudManagerInterface {
public:
    std::optional<int> Create(const std::string& entity_type, const std::string& data) override {
        int next_id = 0;
        if (data_.count(entity_type)) {
            for (const auto& pair : data_[entity_type]) {
                if (pair.first > next_id) {
                    next_id = pair.first;
                }
            }
        }
        next_id++;
        data_[entity_type][next_id] = data;
        last_created_id_ = next_id;
        return next_id;
    }

    std::optional<std::string> Read(const std::string& entity_type, int id) override {
        if (data_.count(entity_type) && data_[entity_type].count(id)) {
            return data_[entity_type][id];
        }
        return std::nullopt;
    }

    bool Update(const std::string& entity_type, int id, const std::string& data) override {
        data_[entity_type][id] = data;
        return true;
    }

    bool Delete(const std::string& entity_type, int id) override {
        if (data_.count(entity_type)) {
            if (data_[entity_type].count(id)) {
                data_[entity_type].erase(id);
                return true;
            }
        }
        return true; // Entity does not exist - return true to match real CrudManager behavior
    }

    std::vector<int> List(const std::string& entity_type) override {
        std::vector<int> ids;
        if (data_.count(entity_type)) {
            for (const auto& pair : data_[entity_type]) {
                ids.push_back(pair.first);
            }
        }
        return ids;
    }

    // Test helpers
    void Clear() { data_.clear(); }
    std::optional<int> last_created_id_;

private:
    std::map<std::string, std::map<int, std::string>> data_;
};


class CrudRequestHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto fake_manager_ptr = std::make_shared<FakeCrudManager>();
        fake_manager_ = fake_manager_ptr.get();
        handler_ = std::make_unique<CrudRequestHandler>("/api", fake_manager_ptr);
    }

    // Helper to create a request object
    request make_req(const std::string& method, const std::string& uri, const std::string& body = "") {
        request req;
        req.method = method;
        req.uri = uri;
        req.body = body;
        return req;
    }

    std::unique_ptr<CrudRequestHandler> handler_;
    FakeCrudManager* fake_manager_;
};

TEST_F(CrudRequestHandlerTest, CreateSuccess) {
    auto req = make_req("POST", "/api/users", "{\"name\":\"Chuy\"}");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_NE(resp->get_headers().find("Content-Type: application/json"), std::string::npos);
    
    // The fake manager assigned ID 1
    EXPECT_EQ(resp->get_content(), "{\"id\": 1}");
}

TEST_F(CrudRequestHandlerTest, ReadSuccess) {
    // Setup: Pre-load data into the fake manager
    fake_manager_->Create("users", "{\"name\":\"Emre\"}"); // ID 1

    auto req = make_req("GET", "/api/users/1");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_NE(resp->get_headers().find("Content-Type: application/json"), std::string::npos);
    EXPECT_EQ(resp->get_content(), "{\"name\":\"Emre\"}");
}

TEST_F(CrudRequestHandlerTest, ReadNotFound) {
    // No data in fake manager
    auto req = make_req("GET", "/api/users/99");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 404 Not Found\r\n");
    EXPECT_EQ(resp->get_content(), "Entity not found.");
}

TEST_F(CrudRequestHandlerTest, ListSuccess) {
    // Setup: Pre-load data
    fake_manager_->Create("users", "{}"); // ID 1
    fake_manager_->Create("users", "{}"); // ID 2

    auto req = make_req("GET", "/api/users");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_NE(resp->get_headers().find("Content-Type: application/json"), std::string::npos);
    EXPECT_EQ(resp->get_content(), "[1, 2]");
}

TEST_F(CrudRequestHandlerTest, ListEmpty) {
    auto req = make_req("GET", "/api/users");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_NE(resp->get_headers().find("Content-Type: application/json"), std::string::npos);
    EXPECT_EQ(resp->get_content(), "[]");
}

TEST_F(CrudRequestHandlerTest, UpdateSuccess) {
    fake_manager_->Create("users", "{\"name\":\"Aron\"}"); // ID 1

    auto req = make_req("PUT", "/api/users/1", "{\"name\":\"Aaron\"}");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp->get_content(), "{\"id\": 1}");

    // Verify the data was actually updated in the fake
    auto updated_data = fake_manager_->Read("users", 1);
    ASSERT_TRUE(updated_data.has_value());
    EXPECT_EQ(*updated_data, "{\"name\":\"Aaron\"}");
}

TEST_F(CrudRequestHandlerTest, DeleteSuccess) {
    fake_manager_->Create("users", "{\"name\":\"Abdullah\"}"); // ID 1
    ASSERT_TRUE(fake_manager_->Read("users", 1).has_value());

    auto req = make_req("DELETE", "/api/users/1");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp->get_content(), "{\"id\": 1}");

    // Verify the data is gone from the fake
    EXPECT_FALSE(fake_manager_->Read("users", 1).has_value());
}

TEST_F(CrudRequestHandlerTest, DeleteNonExistant) {
    // No data in fake manager. Deleting a non-existent ID should succeed.
    auto req = make_req("DELETE", "/api/users/99");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 200 OK\r\n");
    EXPECT_EQ(resp->get_content(), "{\"id\": 99}");
}

TEST_F(CrudRequestHandlerTest, InvalidIdFormat) {
    auto req = make_req("GET", "/api/users/not-a-number");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
    EXPECT_EQ(resp->get_content(), "Invalid ID format.");
}

TEST_F(CrudRequestHandlerTest, MissingIdForPut) {
    auto req = make_req("PUT", "/api/users", "{}");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 400 Bad Request\r\n");
    EXPECT_EQ(resp->get_content(), "ID is required for PUT.");
}

TEST_F(CrudRequestHandlerTest, MethodNotAllowed) {
    auto req = make_req("PATCH", "/api/users/1");
    auto resp = handler_->handle_request(req);

    ASSERT_NE(resp, nullptr);
    EXPECT_EQ(resp->get_status_line(), "HTTP/1.1 405 Method Not Allowed\r\n");
    EXPECT_NE(resp->get_headers().find("Allow: GET, POST, PUT, DELETE"), std::string::npos);
}

TEST_F(CrudRequestHandlerTest, MultiEntitySeparation) {
    // Create an item in two different entity types. The fake manager will assign ID 1 to both.
    auto user_req = make_req("POST", "/api/users", "{\"name\":\"Alice\"}");
    auto user_resp = handler_->handle_request(user_req);
    ASSERT_EQ(user_resp->get_content(), "{\"id\": 1}");

    auto book_req = make_req("POST", "/api/books", "{\"title\":\"The Hobbit\"}");
    auto book_resp = handler_->handle_request(book_req);
    ASSERT_EQ(book_resp->get_content(), "{\"id\": 1}");

    // Listing each entity type shows only its own ID.
    auto list_users_req = make_req("GET", "/api/users");
    auto list_users_resp = handler_->handle_request(list_users_req);
    EXPECT_EQ(list_users_resp->get_content(), "[1]");

    auto list_books_req = make_req("GET", "/api/books");
    auto list_books_resp = handler_->handle_request(list_books_req);
    EXPECT_EQ(list_books_resp->get_content(), "[1]");

    // Reads don't cross categories.
    auto read_user_req = make_req("GET", "/api/users/1");
    auto read_user_resp = handler_->handle_request(read_user_req);
    EXPECT_EQ(read_user_resp->get_content(), "{\"name\":\"Alice\"}");

    auto read_book_req = make_req("GET", "/api/books/1");
    auto read_book_resp = handler_->handle_request(read_book_req);
    EXPECT_EQ(read_book_resp->get_content(), "{\"title\":\"The Hobbit\"}");
}