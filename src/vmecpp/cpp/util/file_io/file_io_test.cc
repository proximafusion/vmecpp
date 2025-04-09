// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/file_io/file_io.h"

#include <fstream>
#include <string>

#include "gtest/gtest.h"

namespace file_io {

TEST(TestFileIO, CheckReadFile) {
  // test that reading non-existent file fails
  std::string nonexistent_filename = "this/file/does/likely/not/exist";

  // make sure that the assumed-nonexisting file actually does not exist
  std::ifstream ifs(nonexistent_filename);
  ASSERT_FALSE(ifs.is_open());

  // now check that reading it actually returns an empty string
  absl::StatusOr<std::string> no_contents = ReadFile(nonexistent_filename);
  EXPECT_FALSE(no_contents.ok());

  // read empty file
  absl::StatusOr<std::string> empty = ReadFile("util/test_data/empty.txt");
  ASSERT_TRUE(empty.ok()) << empty.status().message();
  EXPECT_EQ(*empty, "");

  // read simple text file
  absl::StatusOr<std::string> lorem = ReadFile("util/test_data/lorem.txt");
  ASSERT_TRUE(lorem.ok()) << lorem.status().message();
  EXPECT_EQ(*lorem, "lorem ipsum\n");
}  // CheckReadFile

// rely on file_io::ReadFile for checking if test was successful
TEST(TestFileIO, CheckWriteFile) {
  // create a temporary file name
  namespace fs = std::filesystem;
  const fs::path temp_file =
      fs::temp_directory_path() /
      ("file_io_test_checkwritefile_" + std::to_string(getpid()));

  std::string contents = "Testing file_io::WriteFile...\n";

  absl::Status status = WriteFile(temp_file, contents);
  ASSERT_TRUE(status.ok()) << status.message();

  // now check that the file actually contains the desired contents
  absl::StatusOr<std::string> contents_reread = ReadFile(temp_file);
  ASSERT_TRUE(contents_reread.ok()) << contents_reread.status().message();
  EXPECT_EQ(*contents_reread, contents);

  // clean up the temporary file after use
  std::remove(temp_file.c_str());
}  // CheckWriteFile

}  // namespace file_io
