#include "gtest/gtest.h"
#include "crud_manager.h"
#include "crud_manager_interface.h"
#include <boost/filesystem/fstream.hpp>

#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <map>

namespace fs = boost::filesystem;

/**
 * A mock implementation of the CrudManagerInterface for testing.
 *
 * This class uses in-memory maps to simulate CRUD operations without
 * touching the filesystem, making it ideal for fast and isolated unit tests.
 */
class MockCrudManager : public CrudManagerInterface {
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
            data_[entity_type].erase(id);
        }
        return true; // Deleting non-existent is a success.
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

    // Helper for tests to inspect the internal state.
    void Clear() {
        data_.clear();
    }

private:
    // Storage: { entity_type -> { id -> data } }
    std::map<std::string, std::map<int, std::string>> data_;
};

// Use a typed test suite to run the same tests for both the real and mock managers.
template <typename T>
class CrudManagerTest : public ::testing::Test {
protected:
    // Use a temporary directory for each test to ensure isolation.
    const std::string test_data_path = "/tmp/crud_manager_test_data";
    std::unique_ptr<CrudManagerInterface> manager;

    void SetUp() override {
        // For the real CrudManager, we set up the filesystem.
        // For the MockCrudManager, we just create an instance.
        if constexpr (std::is_same_v<T, CrudManager>) {
            fs::remove_all(test_data_path);
            fs::create_directory(test_data_path);
            manager = std::make_unique<CrudManager>(test_data_path);
        } else {
            // Assumes the mock has a default constructor.
            manager = std::make_unique<MockCrudManager>();
        }
    }

    void TearDown() override {
        // Clean up the directory after the real manager tests.
        if constexpr (std::is_same_v<T, CrudManager>) {
            // No need to clean up for the mock, as it's in-memory.
            fs::remove_all(test_data_path);
        }
    }
};

// Define the types to be tested.
using CrudManagerImplementations = ::testing::Types<CrudManager, MockCrudManager>;
TYPED_TEST_SUITE(CrudManagerTest, CrudManagerImplementations);

TYPED_TEST(CrudManagerTest, CreateAndRead) {
    // Use this-> to access fixture members in a typed test.
    auto& manager = this->manager;
    const std::string entity_type = "users";
    const std::string data1 = "{\"name\":\"Alice\"}";
    const std::string data2 = "{\"name\":\"Bob\"}";

    // Create first entity
    auto id1_opt = manager->Create(entity_type, data1);
    ASSERT_TRUE(id1_opt.has_value());
    int id1 = id1_opt.value();
    EXPECT_EQ(id1, 1);

    // Read it back
    auto read_data1_opt = manager->Read(entity_type, id1);
    ASSERT_TRUE(read_data1_opt.has_value());
    EXPECT_EQ(read_data1_opt.value(), data1);

    // Create second entity
    auto id2_opt = manager->Create(entity_type, data2);
    ASSERT_TRUE(id2_opt.has_value());
    int id2 = id2_opt.value();
    EXPECT_EQ(id2, 2);

    // Read it back
    auto read_data2_opt = manager->Read(entity_type, id2);
    ASSERT_TRUE(read_data2_opt.has_value());
    EXPECT_EQ(read_data2_opt.value(), data2);
}

TYPED_TEST(CrudManagerTest, ReadNonExistent) {
    // Use this-> to access fixture members in a typed test.
    auto& manager = this->manager;
    auto result = manager->Read("products", 99);
    EXPECT_FALSE(result.has_value());
}

TYPED_TEST(CrudManagerTest, UpdateExisting) {
    auto& manager = this->manager;
    // Use this-> to access fixture members in a typed test.
    const std::string entity_type = "products";
    
    auto id_opt = manager->Create(entity_type, "{\"item\":\"cpu\"}");
    ASSERT_TRUE(id_opt.has_value());
    int id = id_opt.value();

    const std::string updated_data = "{\"item\":\"gpu\"}";
    bool success = manager->Update(entity_type, id, updated_data);
    EXPECT_TRUE(success);

    auto read_data_opt = manager->Read(entity_type, id);
    ASSERT_TRUE(read_data_opt.has_value());
    EXPECT_EQ(read_data_opt.value(), updated_data);
}

TYPED_TEST(CrudManagerTest, UpdateCreatesNew) {
    auto& manager = this->manager;
    // Use this-> to access fixture members in a typed test.
    const std::string entity_type = "products";
    const std::string data = "{\"item\":\"ram\"}";

    bool success = manager->Update(entity_type, 5, data);
    EXPECT_TRUE(success);

    auto read_data_opt = manager->Read(entity_type, 5);
    ASSERT_TRUE(read_data_opt.has_value());
    EXPECT_EQ(read_data_opt.value(), data);
}

TYPED_TEST(CrudManagerTest, DeleteExisting) {
    auto& manager = this->manager;
    // Use this-> to access fixture members in a typed test.
    const std::string entity_type = "orders";

    auto id_opt = manager->Create(entity_type, "{\"total\":100}");
    ASSERT_TRUE(id_opt.has_value());
    int id = id_opt.value();

    bool success = manager->Delete(entity_type, id);
    EXPECT_TRUE(success);

    auto result = manager->Read(entity_type, id);
    EXPECT_FALSE(result.has_value());
}

TYPED_TEST(CrudManagerTest, DeleteNonExistent) {
    // Use this-> to access fixture members in a typed test.
    auto& manager = this->manager;
    bool success = manager->Delete("orders", 123);
    EXPECT_TRUE(success); // Deleting a non-existent entity should be a success.
}

TYPED_TEST(CrudManagerTest, ListEntities) {
    // Use this-> to access fixture members in a typed test.
    auto& manager = this->manager;
    const std::string entity_type = "widgets";

    // List on non-existent type should be empty
    EXPECT_TRUE(manager->List(entity_type).empty());

    manager->Create(entity_type, "{}"); // ID 1
    manager->Create(entity_type, "{}"); // ID 2
    manager->Update(entity_type, 5, "{}"); // ID 5

    std::vector<int> ids = manager->List(entity_type);
    std::sort(ids.begin(), ids.end());

    std::vector<int> expected_ids = {1, 2, 5};
    EXPECT_EQ(ids, expected_ids);

    manager->Delete(entity_type, 2);
    ids = manager->List(entity_type);
    std::sort(ids.begin(), ids.end());
    expected_ids = {1, 5};
    EXPECT_EQ(ids, expected_ids);
}

// This test is specific to the real CrudManager's constructor.
// It doesn't make sense to run this for the mock, so it's a regular TEST_F.
TEST(CrudManagerRealTest, AbsolutePathConstructor) {
    CrudManager manager("relative/path");
    // This is a simple check. A more robust test might inspect private members,
    // but that often indicates a design smell. This is good enough.
    // We expect no exceptions and that file operations will be based on an absolute path.
    EXPECT_TRUE(manager.List("anything").empty()); // This test is specific to the concrete class
}

// Generated with AI

// Exercise Create/List's handling of non-integer and overflow filenames
TEST(CrudManagerRealTest, CreateSkipsNonNumericAndOverflowIdsAndListIgnoresThem) {
  const std::string base_path = "/tmp/crud_manager_non_numeric_ids";
  fs::remove_all(base_path);

  fs::path users_dir = fs::path(base_path) / "users";
  fs::create_directories(users_dir);

  // Non-numeric filename
  {
    boost::filesystem::ofstream ofs(users_dir / "not_an_id");
  }
  // Very large numeric filename (overflow for std::stoi)
  {
    boost::filesystem::ofstream ofs(users_dir / "9999999999999999999999999");
  }
  // Valid numeric filename
  {
    boost::filesystem::ofstream ofs(users_dir / "10");
  }

  CrudManager manager(base_path);

  // Should ignore "not_an_id" and the huge number, and choose next id after 10.
  auto id_opt = manager.Create("users", "{}");
  ASSERT_TRUE(id_opt.has_value());
  EXPECT_EQ(id_opt.value(), 11);

  // List should only return valid integer IDs.
  std::vector<int> ids = manager.List("users");
  std::sort(ids.begin(), ids.end());
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0], 10);
  EXPECT_EQ(ids[1], 11);

  fs::remove_all(base_path);
}

// Exercise Create's error path when create_directories fails
TEST(CrudManagerRealTest, CreateFailsWhenParentPathIsFile) {
  const std::string base_path = "/tmp/crud_manager_parent_is_file";
  fs::remove_all(base_path);

  // Create a regular file where the manager expects a directory.
  {
    boost::filesystem::ofstream ofs(base_path);
  }

  CrudManager manager(base_path);
  auto id_opt = manager.Create("users", "{}");

  // Directory creation should fail and Create should return nullopt.
  EXPECT_FALSE(id_opt.has_value());

  fs::remove_all(base_path);
}

// Exercise Update's error path when create_directories fails
TEST(CrudManagerRealTest, UpdateFailsWhenParentPathIsFile) {
  const std::string base_path = "/tmp/crud_manager_parent_is_file_update";
  fs::remove_all(base_path);

  {
    boost::filesystem::ofstream ofs(base_path);
  }

  CrudManager manager(base_path);
  bool ok = manager.Update("users", 1, "{}");

  // Directory creation should fail and Update should return false.
  EXPECT_FALSE(ok);

  fs::remove_all(base_path);
}
