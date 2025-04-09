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

  int work_per_CPU = num_surfaces_to_distribute / numPlasma;
  int work_remainder = num_surfaces_to_distribute % numPlasma;

  for (int myid = 0; myid < ncpu; ++myid) {
    r.adjustRadialPartitioning(ncpu, myid, ns, lfreeb);

    // --------------

    EXPECT_EQ(ncpu, r.get_num_threads());
    EXPECT_EQ(myid, r.get_thread_id());

    // --------------

    nsMinF[myid] = myid * work_per_CPU;
    nsMaxF[myid] = (myid + 1) * work_per_CPU;
    if (myid < work_remainder) {
      nsMinF[myid] += myid;
      nsMaxF[myid] += myid + 1;
    } else {
      nsMinF[myid] += work_remainder;
      nsMaxF[myid] += work_remainder;
    }

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

  int work_per_CPU = num_surfaces_to_distribute / num_threads;
  int work_remainder = num_surfaces_to_distribute % num_threads;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    r.adjustRadialPartitioning(num_threads, thread_id, ns, lfreeb);

    // --------------

    EXPECT_EQ(num_threads, r.get_num_threads());
    EXPECT_EQ(thread_id, r.get_thread_id());

    // --------------

    nsMinF[thread_id] = thread_id * work_per_CPU;
    nsMaxF[thread_id] = (thread_id + 1) * work_per_CPU;
    if (thread_id < work_remainder) {
      nsMinF[thread_id] += thread_id;
      nsMaxF[thread_id] += thread_id + 1;
    } else {
      nsMinF[thread_id] += work_remainder;
      nsMaxF[thread_id] += work_remainder;
    }

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

}  // namespace vmecpp
