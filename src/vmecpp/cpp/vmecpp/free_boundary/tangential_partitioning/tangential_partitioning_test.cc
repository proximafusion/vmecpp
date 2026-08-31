// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace vmecpp {

namespace {

// The [ztMin, ztMax) range every thread of a team of the given size gets.
std::vector<std::pair<int, int> > PartitionRanges(int nZnT, int num_threads) {
  std::vector<std::pair<int, int> > ranges;
  ranges.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    const TangentialPartitioning tp(nZnT, num_threads, thread_id);
    EXPECT_EQ(tp.get_thread_id(), thread_id);
    ranges.emplace_back(tp.ztMin, tp.ztMax);
  }  // thread_id
  return ranges;
}

// The ranges of a team have to tile [0, nZnT) in thread order. Every routine in
// the vacuum solve writes its own [ztMin, ztMax) and then reads the whole
// surface, so a gap leaves stale values in place and an overlap has two threads
// writing the same element.
void ExpectTilesTheRange(int nZnT, int num_threads) {
  const std::vector<std::pair<int, int> > ranges =
      PartitionRanges(nZnT, num_threads);

  EXPECT_EQ(ranges.front().first, 0)
      << "nZnT=" << nZnT << ", num_threads=" << num_threads;
  EXPECT_EQ(ranges.back().second, nZnT)
      << "nZnT=" << nZnT << ", num_threads=" << num_threads;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    EXPECT_LE(ranges[thread_id].first, ranges[thread_id].second)
        << "nZnT=" << nZnT << ", num_threads=" << num_threads << ", thread "
        << thread_id;
    if (thread_id + 1 < num_threads) {
      EXPECT_EQ(ranges[thread_id].second, ranges[thread_id + 1].first)
          << "nZnT=" << nZnT << ", num_threads=" << num_threads
          << ", between threads " << thread_id << " and " << thread_id + 1;
    }
  }  // thread_id
}

// Block sizes take only the two values nZnT / num_threads and one more than
// that, and the larger block goes to exactly the first nZnT % num_threads
// threads.
void ExpectBalanced(int nZnT, int num_threads) {
  const std::vector<std::pair<int, int> > ranges =
      PartitionRanges(nZnT, num_threads);

  const int small_block = nZnT / num_threads;
  const int number_of_large_blocks = nZnT % num_threads;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    const int expected_size =
        thread_id < number_of_large_blocks ? small_block + 1 : small_block;
    EXPECT_EQ(ranges[thread_id].second - ranges[thread_id].first, expected_size)
        << "nZnT=" << nZnT << ", num_threads=" << num_threads << ", thread "
        << thread_id;
  }  // thread_id
}

}  // namespace

TEST(TestTangentialPartitioning, CheckSingleThreadTakesTheWholeRange) {
  static constexpr int kNumTangentialPoints = 360;

  // num_threads and thread_id default to a single-threaded team.
  const TangentialPartitioning tp(kNumTangentialPoints);

  EXPECT_EQ(tp.get_thread_id(), 0);
  EXPECT_EQ(tp.ztMin, 0);
  EXPECT_EQ(tp.ztMax, kNumTangentialPoints);
}

TEST(TestTangentialPartitioning, CheckPartitionsTileTheRange) {
  // Sizes that divide evenly, sizes that do not, and a size smaller than the
  // team. 360 is nZeta * nThetaEff for a 36 x 10 tangential grid.
  const std::vector<int> tangential_point_counts = {0,  1,  2,  3,   5,
                                                    12, 17, 64, 100, 360};
  const std::vector<int> team_sizes = {1, 2, 3, 4, 5, 7, 8, 16, 17, 64};

  for (const int nZnT : tangential_point_counts) {
    for (const int num_threads : team_sizes) {
      ExpectTilesTheRange(nZnT, num_threads);
      ExpectBalanced(nZnT, num_threads);
    }  // num_threads
  }  // nZnT
}

TEST(TestTangentialPartitioning, CheckEvenSplitGivesEqualBlocks) {
  static constexpr int kNumTangentialPoints = 360;
  static constexpr int kNumThreads = 8;
  static constexpr int kBlockSize = kNumTangentialPoints / kNumThreads;

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    const TangentialPartitioning tp(kNumTangentialPoints, kNumThreads,
                                    thread_id);
    EXPECT_EQ(tp.ztMin, thread_id * kBlockSize);
    EXPECT_EQ(tp.ztMax, (thread_id + 1) * kBlockSize);
  }  // thread_id
}

TEST(TestTangentialPartitioning, CheckRemainderGoesToTheLowestThreads) {
  // 10 points over 4 threads: the first two threads get three points, the last
  // two get two.
  static constexpr int kNumTangentialPoints = 10;
  static constexpr int kNumThreads = 4;

  const std::vector<std::pair<int, int> > expected = {
      {0, 3}, {3, 6}, {6, 8}, {8, 10}};

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    const TangentialPartitioning tp(kNumTangentialPoints, kNumThreads,
                                    thread_id);
    EXPECT_EQ(tp.ztMin, expected[thread_id].first) << "thread " << thread_id;
    EXPECT_EQ(tp.ztMax, expected[thread_id].second) << "thread " << thread_id;
  }  // thread_id
}

TEST(TestTangentialPartitioning, CheckMoreThreadsThanPointsGivesEmptyRanges) {
  // A team larger than the surface leaves the surplus threads with an empty
  // range at the end, rather than an inverted or out-of-range one.
  static constexpr int kNumTangentialPoints = 3;
  static constexpr int kNumThreads = 8;

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    const TangentialPartitioning tp(kNumTangentialPoints, kNumThreads,
                                    thread_id);
    if (thread_id < kNumTangentialPoints) {
      EXPECT_EQ(tp.ztMin, thread_id) << "thread " << thread_id;
      EXPECT_EQ(tp.ztMax, thread_id + 1) << "thread " << thread_id;
    } else {
      EXPECT_EQ(tp.ztMin, kNumTangentialPoints) << "thread " << thread_id;
      EXPECT_EQ(tp.ztMax, kNumTangentialPoints) << "thread " << thread_id;
    }
  }  // thread_id
}

TEST(TestTangentialPartitioning, CheckAdjustPartitioningRepartitionsInPlace) {
  // The tangential grid changes with the toroidal resolution, so an existing
  // partitioning is re-adjusted rather than rebuilt.
  static constexpr int kNumThreads = 3;
  static constexpr int kThreadId = 2;

  TangentialPartitioning tp(30, kNumThreads, kThreadId);
  EXPECT_EQ(tp.ztMin, 20);
  EXPECT_EQ(tp.ztMax, 30);

  tp.adjustPartitioning(11);
  EXPECT_EQ(tp.ztMin, 8);
  EXPECT_EQ(tp.ztMax, 11);
  EXPECT_EQ(tp.get_thread_id(), kThreadId);

  tp.adjustPartitioning(30);
  EXPECT_EQ(tp.ztMin, 20);
  EXPECT_EQ(tp.ztMax, 30);
}

}  // namespace vmecpp
