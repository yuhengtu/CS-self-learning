#include "gtest/gtest.h"

#include "handler_registry.h"

TEST(HandlerRegistryTest, UnknownTypeReturnsNullFactory) {
    HandlerSpec spec;
    spec.name = "mystery";
    spec.path = "/mystery";
    spec.type = "does_not_exist";

    auto factory = HandlerRegistry::CreateFactory(spec);
    EXPECT_EQ(factory, nullptr);
}

