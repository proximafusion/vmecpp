// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "gtest/gtest.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

namespace vmecpp {

namespace {

// Cost = makespan = max over threads of (slice_size + boundary overhead).
// Boundary overhead is `k_axis` for tid 0 and `k_lcfs` for tid N-1, modelling
// the extra constant per-call work those threads carry. For num_threads == 1
// the single thread carries both.
int Makespan(const std::vector<int>& sizes, int k_axis, int k_lcfs) {
  const int n = static_cast<int>(sizes.size());
  if (n == 0) {
    return 0;
  }
  if (n == 1) {
    return sizes[0] + k_axis + k_lcfs;
  }
  int worst = 0;
  for (int t = 0; t < n; ++t) {
    int eff = sizes[t];
    if (t == 0) {
      eff += k_axis;
    }
    if (t == n - 1) {
      eff += k_lcfs;
    }
    if (eff > worst) {
      worst = eff;
    }
  }
  return worst;
}

}  // namespace

TEST(TestRadialPartitioning, CheckSingleThreadedFixedBoundary) {
  int ncpu = 1;
  int myid = 0;

  int ns = 15;
  bool lfreeb = false;

  RadialPartitioning r;
  r.adjustRadialPartitioning(ncpu, myid, ns, lfreeb);

  // --------------

  EXPECT_EQ(ncpu, r.get_num_threads());
  EXPECT_EQ(myid, r.get_thread_id());
  EXPECT_TRUE(r.has_boundary());

  // --------------

  EXPECT_EQ(0, r.nsMinF1);
  EXPECT_EQ(ns, r.nsMaxF1);

  EXPECT_EQ(0, r.nsMinH);
  EXPECT_EQ(ns - 1, r.nsMaxH);

  EXPECT_EQ(1, r.nsMinFi);
  EXPECT_EQ(ns - 1, r.nsMaxFi);

  EXPECT_EQ(0, r.nsMinF);
  EXPECT_EQ(ns - 1, r.nsMaxF);

  // --------------

  // test geometry arrays
  {
    // prepare accumulation target
    std::vector<std::vector<int> > visited_geometry(ns);

    // fill with contents to test
    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j - r.nsMinF1].push_back(myid);
    }

    // check contents
    for (int j = 0; j < ns; ++j) {
      ASSERT_EQ(1, visited_geometry[j].size());
      EXPECT_EQ(myid, visited_geometry[j][0]);
    }
  }

  // test fields arrays
  {
    std::vector<std::vector<int> > visited_field(ns - 1);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_field[j - r.nsMinH].push_back(myid);
    }

    for (int j = 0; j < ns - 1; ++j) {
      ASSERT_EQ(1, visited_field[j].size());
      EXPECT_EQ(myid, visited_field[j][0]);
    }
  }

  // test internal arrays
  {
    std::vector<std::vector<int> > visited_internal(ns - 2);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - r.nsMinFi].push_back(myid);
    }

    for (int j = 0; j < ns - 2; ++j) {
      ASSERT_EQ(1, visited_internal[j].size());
      EXPECT_EQ(myid, visited_internal[j][0]);
    }
  }

  // test forces arrays
  {
    int num_active_surfaces = ns - 1;
    if (lfreeb) {
      num_active_surfaces = ns;
    }

    std::vector<std::vector<int> > visited_forces(num_active_surfaces);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j - r.nsMinF].push_back(myid);
    }

    for (int j = 0; j < num_active_surfaces; ++j) {
      ASSERT_EQ(1, visited_forces[j].size());
      EXPECT_EQ(myid, visited_forces[j][0]);
    }
  }
}  // CheckSingleThreadedFixedBoundary

TEST(TestRadialPartitioning, CheckSingleThreadedFreeBoundary) {
  int ncpu = 1;
  int myid = 0;

  int ns = 15;
  bool lfreeb = true;

  RadialPartitioning r;
  r.adjustRadialPartitioning(ncpu, myid, ns, lfreeb);

  // --------------

  EXPECT_EQ(ncpu, r.get_num_threads());
  EXPECT_EQ(myid, r.get_thread_id());
  EXPECT_TRUE(r.has_boundary());

  // --------------

  EXPECT_EQ(0, r.nsMinF1);
  EXPECT_EQ(ns, r.nsMaxF1);

  EXPECT_EQ(0, r.nsMinH);
  EXPECT_EQ(ns - 1, r.nsMaxH);

  EXPECT_EQ(1, r.nsMinFi);
  EXPECT_EQ(ns - 1, r.nsMaxFi);

  EXPECT_EQ(0, r.nsMinF);
  EXPECT_EQ(ns, r.nsMaxF);

  // --------------

  // test forces arrays
  {
    int num_surfaces_to_distribute = ns - 1;
    if (lfreeb) {
      num_surfaces_to_distribute = ns;
    }

    std::vector<std::vector<int> > visited_forces(num_surfaces_to_distribute);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j - r.nsMinF].push_back(myid);
    }

    for (int j = 0; j < num_surfaces_to_distribute; ++j) {
      ASSERT_EQ(1, visited_forces[j].size());
      EXPECT_EQ(myid, visited_forces[j][0]);
    }
  }

  // test geometry arrays
  {
    std::vector<std::vector<int> > visited_geometry(ns);

    // fill with contents to test
    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j - r.nsMinF1].push_back(myid);
    }

    // check contents
    for (int j = 0; j < ns; ++j) {
      ASSERT_EQ(1, visited_geometry[j].size());
      EXPECT_EQ(myid, visited_geometry[j][0]);
    }
  }

  // test fields arrays
  {
    std::vector<std::vector<int> > visited_field(ns - 1);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_field[j - r.nsMinH].push_back(myid);
    }

    for (int j = 0; j < ns - 1; ++j) {
      ASSERT_EQ(1, visited_field[j].size());
      EXPECT_EQ(myid, visited_field[j][0]);
    }
  }

  // test internal arrays
  {
    std::vector<std::vector<int> > visited_internal(ns - 2);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - r.nsMinFi].push_back(myid);
    }

    for (int j = 0; j < ns - 2; ++j) {
      ASSERT_EQ(1, visited_internal[j].size());
      EXPECT_EQ(myid, visited_internal[j][0]);
    }
  }
}  // CheckSingleThreadedFreeBoundary

TEST(TestRadialPartitioning, CheckMultiThreadedFixedBoundaryAllActive) {
  int ncpu = 4;

  int ns = 15;
  bool lfreeb = false;

  RadialPartitioning r;

  int num_surfaces_to_distribute = ns - 1;
  if (lfreeb) {
    num_surfaces_to_distribute = ns;
  }

  int numPlasma = std::min(ncpu, num_surfaces_to_distribute / 2);

  std::vector<int> nsMinF(numPlasma);
  std::vector<int> nsMaxF(numPlasma);
  std::vector<int> nsMinF1(numPlasma);
  std::vector<int> nsMaxF1(numPlasma);
  std::vector<int> nsMinH(numPlasma);
  std::vector<int> nsMaxH(numPlasma);
  std::vector<int> nsMinFi(numPlasma);
  std::vector<int> nsMaxFi(numPlasma);

  std::vector<std::vector<int> > visited_forces(num_surfaces_to_distribute);
  std::vector<std::vector<int> > visited_geometry(ns);
  std::vector<std::vector<int> > visited_fields(ns - 1);
  std::vector<std::vector<int> > visited_internal(ns - 2);

  // Hard-coded expected slice sizes for ns=15, num_threads=4 (fixed-bdy).
  // Derived by hand from the imbalance-aware algorithm with
  // kBoundarySurfacesOffAxis = 2, kBoundarySurfacesOffLcfs = 1:
  //   total_work = 14 + 2 + 1 = 17, base = 4, rem = 1.
  //   assigned_work = [5, 4, 4, 4]; slice = assigned - {2, 0, 0, 1}.
  ASSERT_EQ(15, ns);
  ASSERT_EQ(4, numPlasma);
  ASSERT_EQ(2, vmec_algorithm_constants::kBoundarySurfacesOffAxis);
  ASSERT_EQ(1, vmec_algorithm_constants::kBoundarySurfacesOffLcfs);
  const std::vector<int> expected_sizes = {3, 4, 4, 3};
  int cumulative_min = 0;

  for (int myid = 0; myid < ncpu; ++myid) {
    r.adjustRadialPartitioning(ncpu, myid, ns, lfreeb);

    // --------------

    EXPECT_EQ(ncpu, r.get_num_threads());
    EXPECT_EQ(myid, r.get_thread_id());

    // --------------

    nsMinF[myid] = cumulative_min;
    nsMaxF[myid] = cumulative_min + expected_sizes[myid];
    cumulative_min = nsMaxF[myid];

    EXPECT_EQ(nsMinF[myid], r.nsMinF);
    EXPECT_EQ(nsMaxF[myid], r.nsMaxF);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j].push_back(myid);
    }

    // --------------------------------

    nsMinF1[myid] = std::max(0, nsMinF[myid] - 1);
    nsMaxF1[myid] = std::min(ns, nsMaxF[myid] + 1);

    EXPECT_EQ(nsMinF1[myid], r.nsMinF1);
    EXPECT_EQ(nsMaxF1[myid], r.nsMaxF1);

    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j].push_back(myid);
    }

    EXPECT_EQ(r.has_boundary(), nsMaxF1[myid] == ns);

    // --------------------------------

    nsMinH[myid] = nsMinF1[myid];
    nsMaxH[myid] = nsMaxF1[myid] - 1;

    EXPECT_EQ(nsMinH[myid], r.nsMinH);
    EXPECT_EQ(nsMaxH[myid], r.nsMaxH);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_fields[j].push_back(myid);
    }

    // --------------------------------

    nsMinFi[myid] = std::max(1, nsMinF[myid]);
    nsMaxFi[myid] = std::min(ns - 1, nsMaxF[myid]);

    EXPECT_EQ(nsMinFi[myid], r.nsMinFi);
    EXPECT_EQ(nsMaxFi[myid], r.nsMaxFi);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - 1].push_back(myid);
    }
  }

  // ...F
  for (int j = 0; j < num_surfaces_to_distribute; ++j) {
    ASSERT_EQ(1, visited_forces[j].size());
    int visitor_id = visited_forces[j][0];
    EXPECT_TRUE(nsMinF[visitor_id] <= j && j < nsMaxF[visitor_id]);
  }

  // ...F1
  for (int j = 0; j < ns; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (nsMinF1[myid] <= j && j < nsMaxF1[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_geometry[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_geometry[j][visitor];
      EXPECT_TRUE(nsMinF1[visitor_id] <= j && j < nsMaxF1[visitor_id]);
    }
  }

  // ...H
  for (int j = 0; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (nsMinH[myid] <= j && j < nsMaxH[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_fields[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_fields[j][visitor];
      EXPECT_TRUE(nsMinH[visitor_id] <= j && j < nsMaxH[visitor_id]);
    }
  }

  // ...Fi
  for (int j = 1; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (nsMinFi[myid] <= j && j < nsMaxFi[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_internal[j - 1].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_internal[j - 1][visitor];
      EXPECT_TRUE(nsMinFi[visitor_id] <= j && j < nsMaxFi[visitor_id]);
    }
  }
}  // CheckMultiThreadedFixedBoundaryAllActive

TEST(TestRadialPartitioning, CheckMultiThreadedFixedBoundarySomeActive) {
  const int ns = 15;
  const bool lfreeb = false;

#ifdef _OPENMP
  // This is initially given by the OMP_NUM_THREADS environment variable.
  // In combination with below limit, we never use more threads than given by
  // OMP_NUM_THREADS, and also never more than needed for VMEC (ns/2).
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif

  int num_surfaces_to_distribute = ns - 1;
  if (lfreeb) {
    num_surfaces_to_distribute = ns;
  }

  // Objective: Distribute num_surfaces_to_distribute among max_threads threads.
  // A minimum of 2 flux surfaces per thread is allowed
  // to have at least a single shared half-grid point in between them.
  // --> maximum number of usable threads for plasma == floor(ns / 2), as done
  // by integer divide
  const int num_threads = std::min(max_threads, num_surfaces_to_distribute / 2);

#ifdef _OPENMP
  // This must be done _before_ the '#pragma omp parallel' is entered.
  omp_set_num_threads(num_threads);
#endif

  RadialPartitioning r;

  std::vector<int> nsMinF(num_threads);
  std::vector<int> nsMaxF(num_threads);
  std::vector<int> nsMinF1(num_threads);
  std::vector<int> nsMaxF1(num_threads);
  std::vector<int> nsMinH(num_threads);
  std::vector<int> nsMaxH(num_threads);
  std::vector<int> nsMinFi(num_threads);
  std::vector<int> nsMaxFi(num_threads);

  std::vector<std::vector<int> > visited_forces(num_surfaces_to_distribute);
  std::vector<std::vector<int> > visited_geometry(ns);
  std::vector<std::vector<int> > visited_fields(ns - 1);
  std::vector<std::vector<int> > visited_internal(ns - 2);

  // Hard-coded expected slice sizes for ns=15 (fixed-bdy, num_surfaces=14)
  // across the legal num_threads range, derived by hand from the
  // imbalance-aware algorithm with k_axis=2 / k_lcfs=1. The actual
  // num_threads used in this test is determined by OMP_NUM_THREADS,
  // capped at ns/2=7 by the partitioner's upstream invariant.
  ASSERT_EQ(15, ns);
  ASSERT_EQ(2, vmec_algorithm_constants::kBoundarySurfacesOffAxis);
  ASSERT_EQ(1, vmec_algorithm_constants::kBoundarySurfacesOffLcfs);
  const std::vector<std::vector<int> > kExpectedSizesByThreads = {
      {},                          // unused (idx 0)
      {14},                        // T=1: single thread takes everything
      {7, 7},                      // T=2: total=17, base=8, rem=1; [9-2, 8-1]
      {4, 6, 4},                   // T=3: total=17, base=5, rem=2; [6-2, 6, 5-1]
      {3, 4, 4, 3},                // T=4: total=17, base=4, rem=1
      {2, 4, 3, 3, 2},             // T=5: total=17, base=3, rem=2
      {1, 3, 3, 3, 3, 1},          // T=6: total=17, base=2, rem=5
      {1, 3, 3, 2, 2, 2, 1},       // T=7: total=17, base=2, rem=3
  };
  ASSERT_LT(num_threads, static_cast<int>(kExpectedSizesByThreads.size()));
  const std::vector<int>& expected_sizes =
      kExpectedSizesByThreads[num_threads];
  ASSERT_EQ(num_threads, static_cast<int>(expected_sizes.size()));
  int cumulative_min = 0;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    r.adjustRadialPartitioning(num_threads, thread_id, ns, lfreeb);

    // --------------

    EXPECT_EQ(num_threads, r.get_num_threads());
    EXPECT_EQ(thread_id, r.get_thread_id());

    // --------------

    nsMinF[thread_id] = cumulative_min;
    nsMaxF[thread_id] = cumulative_min + expected_sizes[thread_id];
    cumulative_min = nsMaxF[thread_id];

    EXPECT_EQ(nsMinF[thread_id], r.nsMinF);
    EXPECT_EQ(nsMaxF[thread_id], r.nsMaxF);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j].push_back(thread_id);
    }

    // --------------------------------

    nsMinF1[thread_id] = std::max(0, nsMinF[thread_id] - 1);
    nsMaxF1[thread_id] = std::min(ns, nsMaxF[thread_id] + 1);

    EXPECT_EQ(nsMinF1[thread_id], r.nsMinF1);
    EXPECT_EQ(nsMaxF1[thread_id], r.nsMaxF1);

    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j].push_back(thread_id);
    }

    EXPECT_EQ(r.has_boundary(), nsMaxF1[thread_id] == ns);

    // --------------------------------

    nsMinH[thread_id] = nsMinF1[thread_id];
    nsMaxH[thread_id] = nsMaxF1[thread_id] - 1;

    EXPECT_EQ(nsMinH[thread_id], r.nsMinH);
    EXPECT_EQ(nsMaxH[thread_id], r.nsMaxH);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_fields[j].push_back(thread_id);
    }

    // --------------------------------

    nsMinFi[thread_id] = std::max(1, nsMinF[thread_id]);
    nsMaxFi[thread_id] = std::min(ns - 1, nsMaxF[thread_id]);

    EXPECT_EQ(nsMinFi[thread_id], r.nsMinFi);
    EXPECT_EQ(nsMaxFi[thread_id], r.nsMaxFi);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - 1].push_back(thread_id);
    }
  }

  // ...F
  for (int j = 0; j < num_surfaces_to_distribute; ++j) {
    ASSERT_EQ(1, visited_forces[j].size());
    int visitor_id = visited_forces[j][0];
    EXPECT_TRUE(nsMinF[visitor_id] <= j && j < nsMaxF[visitor_id]);
  }

  // ...F1
  for (int j = 0; j < ns; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (nsMinF1[thread_id] <= j && j < nsMaxF1[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_geometry[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_geometry[j][visitor];
      EXPECT_TRUE(nsMinF1[visitor_id] <= j && j < nsMaxF1[visitor_id]);
    }
  }

  // ...H
  for (int j = 0; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (nsMinH[thread_id] <= j && j < nsMaxH[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_fields[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_fields[j][visitor];
      EXPECT_TRUE(nsMinH[visitor_id] <= j && j < nsMaxH[visitor_id]);
    }
  }

  // ...Fi
  for (int j = 1; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (nsMinFi[thread_id] <= j && j < nsMaxFi[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_internal[j - 1].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_internal[j - 1][visitor];
      EXPECT_TRUE(nsMinFi[visitor_id] <= j && j < nsMaxFi[visitor_id]);
    }
  }
}  // CheckMultiThreadedFixedBoundarySomeActive

// Sweep (ns, num_threads) and confirm that the partition produced by
// `RadialPartitioning::adjustRadialPartitioning` never has a worse
// makespan than the equal-share partition under the boundary-overhead
// model. Makespan is the bottleneck thread's effective work
// (slice + boundary overhead), which gates every team barrier in the
// hot iteration loop, so this is the property that translates directly
// to wall-clock scaling.
TEST(TestRadialPartitioning, BoundaryAwarePartitioningLowersMakespan) {
  const int k_axis = vmec_algorithm_constants::kBoundarySurfacesOffAxis;
  const int k_lcfs = vmec_algorithm_constants::kBoundarySurfacesOffLcfs;

  // Skip the no-op case (constants both zero) -- the imbalance-aware
  // partition is then identical to equal-share by construction.
  if (k_axis <= 0 && k_lcfs <= 0) {
    GTEST_SKIP() << "Boundary surface offsets are zero; partitioning is "
                 << "identical to the equal-share scheme.";
  }

  const std::vector<int> ns_values = {15, 31, 50, 99, 150, 199, 299, 499};
  const std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32, 64};
  const bool lfreeb = false;

  int num_cases_checked = 0;
  for (int ns : ns_values) {
    const int num_surfaces = lfreeb ? ns : (ns - 1);

    for (int num_threads : thread_counts) {
      // Same upstream invariant as the implementation: at least 2 surfaces
      // per thread (one half-grid point shared with each neighbour).
      if (num_threads > num_surfaces / 2) {
        continue;
      }
      ++num_cases_checked;

      // Drive the partitioner once per thread to populate this thread's
      // slice; collect the resulting (slice_size) for every thread.
      std::vector<int> shifted_sizes(num_threads);
      RadialPartitioning r;
      for (int tid = 0; tid < num_threads; ++tid) {
        r.adjustRadialPartitioning(num_threads, tid, ns, lfreeb,
                                   /*printout=*/false);
        shifted_sizes[tid] = r.nsMaxF - r.nsMinF;
        EXPECT_GE(shifted_sizes[tid], 1)
            << "thread " << tid << " starved at ns=" << ns
            << " num_threads=" << num_threads;
      }
      int sum = 0;
      for (int s : shifted_sizes) {
        sum += s;
      }
      EXPECT_EQ(sum, num_surfaces)
          << "ns=" << ns << " num_threads=" << num_threads;

      // Equal-share (uniform) partition for comparison.
      std::vector<int> uniform_sizes(num_threads);
      const int base = num_surfaces / num_threads;
      const int rem = num_surfaces % num_threads;
      for (int t = 0; t < num_threads; ++t) {
        uniform_sizes[t] = base + (t < rem ? 1 : 0);
      }

      const int shifted_cost = Makespan(shifted_sizes, k_axis, k_lcfs);
      const int uniform_cost = Makespan(uniform_sizes, k_axis, k_lcfs);

      EXPECT_LE(shifted_cost, uniform_cost)
          << "boundary-aware partition has higher makespan than the "
          << "equal-share partition: "
          << "ns=" << ns << " num_threads=" << num_threads
          << " shifted_cost=" << shifted_cost
          << " uniform_cost=" << uniform_cost;
    }
  }
  EXPECT_GT(num_cases_checked, 0) << "no (ns, num_threads) cases were tested";
}  // BoundaryAwarePartitioningLowersMakespan

}  // namespace vmecpp
