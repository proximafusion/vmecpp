// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "gtest/gtest.h"

namespace vmecpp {

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

  int num_plasma = std::min(ncpu, num_surfaces_to_distribute / 2);

  std::vector<int> ns_min_f(num_plasma);
  std::vector<int> ns_max_f(num_plasma);
  std::vector<int> ns_min_f1(num_plasma);
  std::vector<int> ns_max_f1(num_plasma);
  std::vector<int> ns_min_h(num_plasma);
  std::vector<int> ns_max_h(num_plasma);
  std::vector<int> ns_min_fi(num_plasma);
  std::vector<int> ns_max_fi(num_plasma);

  std::vector<std::vector<int> > visited_forces(num_surfaces_to_distribute);
  std::vector<std::vector<int> > visited_geometry(ns);
  std::vector<std::vector<int> > visited_fields(ns - 1);
  std::vector<std::vector<int> > visited_internal(ns - 2);

  int work_per_cpu = num_surfaces_to_distribute / num_plasma;
  int work_remainder = num_surfaces_to_distribute % num_plasma;

  for (int myid = 0; myid < ncpu; ++myid) {
    r.adjustRadialPartitioning(ncpu, myid, ns, lfreeb);

    // --------------

    EXPECT_EQ(ncpu, r.get_num_threads());
    EXPECT_EQ(myid, r.get_thread_id());

    // --------------

    ns_min_f[myid] = myid * work_per_cpu;
    ns_max_f[myid] = (myid + 1) * work_per_cpu;
    if (myid < work_remainder) {
      ns_min_f[myid] += myid;
      ns_max_f[myid] += myid + 1;
    } else {
      ns_min_f[myid] += work_remainder;
      ns_max_f[myid] += work_remainder;
    }

    EXPECT_EQ(ns_min_f[myid], r.nsMinF);
    EXPECT_EQ(ns_max_f[myid], r.nsMaxF);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j].push_back(myid);
    }

    // --------------------------------

    ns_min_f1[myid] = std::max(0, ns_min_f[myid] - 1);
    ns_max_f1[myid] = std::min(ns, ns_max_f[myid] + 1);

    EXPECT_EQ(ns_min_f1[myid], r.nsMinF1);
    EXPECT_EQ(ns_max_f1[myid], r.nsMaxF1);

    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j].push_back(myid);
    }

    EXPECT_EQ(r.has_boundary(), ns_max_f1[myid] == ns);

    // --------------------------------

    ns_min_h[myid] = ns_min_f1[myid];
    ns_max_h[myid] = ns_max_f1[myid] - 1;

    EXPECT_EQ(ns_min_h[myid], r.nsMinH);
    EXPECT_EQ(ns_max_h[myid], r.nsMaxH);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_fields[j].push_back(myid);
    }

    // --------------------------------

    ns_min_fi[myid] = std::max(1, ns_min_f[myid]);
    ns_max_fi[myid] = std::min(ns - 1, ns_max_f[myid]);

    EXPECT_EQ(ns_min_fi[myid], r.nsMinFi);
    EXPECT_EQ(ns_max_fi[myid], r.nsMaxFi);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - 1].push_back(myid);
    }
  }

  // ...F
  for (int j = 0; j < num_surfaces_to_distribute; ++j) {
    ASSERT_EQ(1, visited_forces[j].size());
    int visitor_id = visited_forces[j][0];
    EXPECT_TRUE(ns_min_f[visitor_id] <= j && j < ns_max_f[visitor_id]);
  }

  // ...F1
  for (int j = 0; j < ns; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (ns_min_f1[myid] <= j && j < ns_max_f1[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_geometry[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_geometry[j][visitor];
      EXPECT_TRUE(ns_min_f1[visitor_id] <= j && j < ns_max_f1[visitor_id]);
    }
  }

  // ...H
  for (int j = 0; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (ns_min_h[myid] <= j && j < ns_max_h[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_fields[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_fields[j][visitor];
      EXPECT_TRUE(ns_min_h[visitor_id] <= j && j < ns_max_h[visitor_id]);
    }
  }

  // ...Fi
  for (int j = 1; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int myid = 0; myid < ncpu; ++myid) {
      if (ns_min_fi[myid] <= j && j < ns_max_fi[myid]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_internal[j - 1].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_internal[j - 1][visitor];
      EXPECT_TRUE(ns_min_fi[visitor_id] <= j && j < ns_max_fi[visitor_id]);
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

  std::vector<int> ns_min_f(num_threads);
  std::vector<int> ns_max_f(num_threads);
  std::vector<int> ns_min_f1(num_threads);
  std::vector<int> ns_max_f1(num_threads);
  std::vector<int> ns_min_h(num_threads);
  std::vector<int> ns_max_h(num_threads);
  std::vector<int> ns_min_fi(num_threads);
  std::vector<int> ns_max_fi(num_threads);

  std::vector<std::vector<int> > visited_forces(num_surfaces_to_distribute);
  std::vector<std::vector<int> > visited_geometry(ns);
  std::vector<std::vector<int> > visited_fields(ns - 1);
  std::vector<std::vector<int> > visited_internal(ns - 2);

  int work_per_cpu = num_surfaces_to_distribute / num_threads;
  int work_remainder = num_surfaces_to_distribute % num_threads;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    r.adjustRadialPartitioning(num_threads, thread_id, ns, lfreeb);

    // --------------

    EXPECT_EQ(num_threads, r.get_num_threads());
    EXPECT_EQ(thread_id, r.get_thread_id());

    // --------------

    ns_min_f[thread_id] = thread_id * work_per_cpu;
    ns_max_f[thread_id] = (thread_id + 1) * work_per_cpu;
    if (thread_id < work_remainder) {
      ns_min_f[thread_id] += thread_id;
      ns_max_f[thread_id] += thread_id + 1;
    } else {
      ns_min_f[thread_id] += work_remainder;
      ns_max_f[thread_id] += work_remainder;
    }

    EXPECT_EQ(ns_min_f[thread_id], r.nsMinF);
    EXPECT_EQ(ns_max_f[thread_id], r.nsMaxF);

    for (int j = r.nsMinF; j < r.nsMaxF; ++j) {
      visited_forces[j].push_back(thread_id);
    }

    // --------------------------------

    ns_min_f1[thread_id] = std::max(0, ns_min_f[thread_id] - 1);
    ns_max_f1[thread_id] = std::min(ns, ns_max_f[thread_id] + 1);

    EXPECT_EQ(ns_min_f1[thread_id], r.nsMinF1);
    EXPECT_EQ(ns_max_f1[thread_id], r.nsMaxF1);

    for (int j = r.nsMinF1; j < r.nsMaxF1; ++j) {
      visited_geometry[j].push_back(thread_id);
    }

    EXPECT_EQ(r.has_boundary(), ns_max_f1[thread_id] == ns);

    // --------------------------------

    ns_min_h[thread_id] = ns_min_f1[thread_id];
    ns_max_h[thread_id] = ns_max_f1[thread_id] - 1;

    EXPECT_EQ(ns_min_h[thread_id], r.nsMinH);
    EXPECT_EQ(ns_max_h[thread_id], r.nsMaxH);

    for (int j = r.nsMinH; j < r.nsMaxH; ++j) {
      visited_fields[j].push_back(thread_id);
    }

    // --------------------------------

    ns_min_fi[thread_id] = std::max(1, ns_min_f[thread_id]);
    ns_max_fi[thread_id] = std::min(ns - 1, ns_max_f[thread_id]);

    EXPECT_EQ(ns_min_fi[thread_id], r.nsMinFi);
    EXPECT_EQ(ns_max_fi[thread_id], r.nsMaxFi);

    for (int j = r.nsMinFi; j < r.nsMaxFi; ++j) {
      visited_internal[j - 1].push_back(thread_id);
    }
  }

  // ...F
  for (int j = 0; j < num_surfaces_to_distribute; ++j) {
    ASSERT_EQ(1, visited_forces[j].size());
    int visitor_id = visited_forces[j][0];
    EXPECT_TRUE(ns_min_f[visitor_id] <= j && j < ns_max_f[visitor_id]);
  }

  // ...F1
  for (int j = 0; j < ns; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (ns_min_f1[thread_id] <= j && j < ns_max_f1[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_geometry[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_geometry[j][visitor];
      EXPECT_TRUE(ns_min_f1[visitor_id] <= j && j < ns_max_f1[visitor_id]);
    }
  }

  // ...H
  for (int j = 0; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (ns_min_h[thread_id] <= j && j < ns_max_h[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_fields[j].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_fields[j][visitor];
      EXPECT_TRUE(ns_min_h[visitor_id] <= j && j < ns_max_h[visitor_id]);
    }
  }

  // ...Fi
  for (int j = 1; j < ns - 1; ++j) {
    int num_visitors = 0;
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      if (ns_min_fi[thread_id] <= j && j < ns_max_fi[thread_id]) {
        num_visitors++;
      }
    }
    ASSERT_EQ(num_visitors, visited_internal[j - 1].size());
    for (int visitor = 0; visitor < num_visitors; ++visitor) {
      int visitor_id = visited_internal[j - 1][visitor];
      EXPECT_TRUE(ns_min_fi[visitor_id] <= j && j < ns_max_fi[visitor_id]);
    }
  }
}  // CheckMultiThreadedFixedBoundarySomeActive

}  // namespace vmecpp
