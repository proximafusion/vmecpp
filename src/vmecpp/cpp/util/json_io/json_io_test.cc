// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/json_io/json_io.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace json_io {

using nlohmann::json;

using ::testing::ElementsAre;

TEST(TestJsonIO, CheckJsonReadBool) {
  json j = R"({"boolean_variable":true,"integer_variable":42})"_json;

  // test check for correct type
  auto read_int_as_bool = JsonReadBool(j, "integer_variable");
  ASSERT_FALSE(read_int_as_bool.ok());

  // test check for presence
  auto read_non_existent = JsonReadBool(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // test reading a boolean with true value
  auto boolean_true = JsonReadBool(j, "boolean_variable");
  ASSERT_TRUE(boolean_true.ok());
  ASSERT_TRUE(boolean_true->has_value());
  EXPECT_EQ(boolean_true.value(), true);

  // test reading a boolean with false value
  json j2 = R"({"boolean_variable":false})"_json;
  auto boolean_false = JsonReadBool(j2, "boolean_variable");
  ASSERT_TRUE(boolean_false.ok());
  ASSERT_TRUE(boolean_false->has_value());
  EXPECT_EQ(boolean_false.value(), false);
}  // CheckJsonReadBool

TEST(TestJsonIO, CheckJsonReadInt) {
  json j = R"({"integer_variable":123,"boolean_variable":false})"_json;

  // test check for correct type
  auto read_bool_as_int = JsonReadInt(j, "boolean_variable");
  ASSERT_FALSE(read_bool_as_int.ok());

  // test check for presence
  auto read_non_existent = JsonReadInt(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // test reading an int
  auto integer_variable = JsonReadInt(j, "integer_variable");
  ASSERT_TRUE(integer_variable.ok());
  ASSERT_TRUE(integer_variable->has_value());
  EXPECT_EQ(integer_variable.value(), 123);
}  // CheckJsonReadInt

TEST(TestJsonIO, CheckJsonReadDouble) {
  json j =
      R"({"double_variable":123.456,"integer_variable":1,"boolean_variable":false})"_json;

  // test check for correct type
  auto read_bool_as_double = JsonReadDouble(j, "boolean_variable");
  ASSERT_FALSE(read_bool_as_double.ok());

  // test check for presence
  auto read_non_existent = JsonReadDouble(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // test reading a double
  auto double_variable = JsonReadDouble(j, "double_variable");
  ASSERT_TRUE(double_variable.ok());
  ASSERT_TRUE(double_variable->has_value());
  EXPECT_EQ(double_variable.value(), 123.456);

  // test reading an int as a double
  auto integer_variable = JsonReadDouble(j, "integer_variable");
  ASSERT_TRUE(integer_variable.ok());
  ASSERT_TRUE(integer_variable->has_value());
  EXPECT_EQ(integer_variable.value(), 1.0);
}  // CheckJsonReadDouble

TEST(TestJsonIO, CheckJsonReadString) {
  json j =
      R"({"string_variable":"test string","double_variable":123.456})"_json;

  // test check for correct type
  auto read_double_as_string = JsonReadString(j, "double_variable");
  ASSERT_FALSE(read_double_as_string.ok());

  // test check for presence
  auto read_non_existent = JsonReadString(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // check reading a string
  auto string_variable = JsonReadString(j, "string_variable");
  ASSERT_TRUE(string_variable.ok());
  ASSERT_TRUE(string_variable->has_value());
  EXPECT_EQ(string_variable.value(), "test string");
}  // CheckJsonReadString

TEST(TestJsonIO, CheckJsonReadVectorInt) {
  json j =
      R"({"vector_int_variable":[5,11,55],"string_variable":"test string"})"_json;

  // test check for correct type
  auto read_string_as_vector_int = JsonReadVectorInt(j, "string_variable");
  ASSERT_FALSE(read_string_as_vector_int.ok());

  // test check for presence
  auto read_non_existent = JsonReadVectorInt(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // check reading a vector of int
  auto vector_int_variable = JsonReadVectorInt(j, "vector_int_variable");
  ASSERT_TRUE(vector_int_variable.ok());
  ASSERT_TRUE(vector_int_variable->has_value());
  EXPECT_THAT(vector_int_variable->value(), ElementsAre(5, 11, 55));
}  // CheckJsonReadVectorInt

TEST(TestJsonIO, CheckJsonReadVectorDouble) {
  json j =
      R"({"vector_double_variable":[1.0e-12,1.0e-16,123.456],"string_variable":"test string"})"_json;

  // test check for correct type
  auto read_string_as_vector_double =
      JsonReadVectorDouble(j, "string_variable");
  ASSERT_FALSE(read_string_as_vector_double.ok());

  // test check for presence
  auto read_non_existent = JsonReadVectorDouble(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // check reading a vector of int
  auto vector_double_variable =
      JsonReadVectorDouble(j, "vector_double_variable");
  ASSERT_TRUE(vector_double_variable.ok());
  ASSERT_TRUE(vector_double_variable->has_value());
  EXPECT_THAT(vector_double_variable->value(),
              ElementsAre(1.0e-12, 1.0e-16, 123.456));
}  // CheckJsonReadVectorDouble

}  // namespace json_io
