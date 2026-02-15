#include "gtest/gtest.h"

#include <memory>
#include <string>

#include "handler_registry.h"
#include "handler_types.h"
#include "request.h"
#include "response.h"

TEST(CrudHandlerFactoryTest, CreatesHandlerHappyPath) {
    HandlerSpec spec;
    spec.name = "api_handler";
    spec.path = "/api";
    spec.type = handler_types::CRUD_HANDLER;
    spec.options["data_path"] = "/tmp/crud_test_data";

    auto factory = HandlerRegistry::CreateFactory(spec);
    ASSERT_NE(factory, nullptr);

    auto handler = factory->create(spec.path, spec.path + std::string("/users"));
    ASSERT_NE(handler, nullptr);
    EXPECT_EQ(handler->name(), handler_types::CRUD_HANDLER);
}

TEST(CrudHandlerFactoryTest, MissingDataPathReturnsNullFactory) {
    HandlerSpec spec;
    spec.name = "bad_crud";
    spec.path = "/api";
    spec.type = handler_types::CRUD_HANDLER;
    // Missing spec.options["data_path"]

    auto factory = HandlerRegistry::CreateFactory(spec);
    EXPECT_EQ(factory, nullptr);
}