// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Asynchronous NESTOR worker for the CUDA free-boundary path.
//
// The free-boundary iteration body keeps the NESTOR vacuum solve on the host;
// run inline it serializes the GPU iteration behind a millisecond-scale host
// solve every step. This worker runs that solve on a background thread so it
// overlaps the device force assembly, residual reduction, time step, and the
// next iteration's geometry. The device applies the edge-pressure profile from
// the previous iteration, a one-iteration-stale vacuum response; the staged
// K-window sync-elision experiment established that one-iteration staleness
// converges (K=2 converged, K>=3 did not), and at convergence the geometry is
// stationary so the stale and current responses coincide and the fixed point
// is unchanged.
//
// The worker owns a thread-private Nestor (its own response matrix, right-hand
// side, pivots, and vacuum_magnetic_pressure output), so it shares only the
// read-only mgrid and tangential partitioning with the main thread. It is
// engaged only once the vacuum contribution is fully active and for
// single-configuration runs; the soft-start window and the batched path run
// the inline solve.
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_NESTOR_ASYNC_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_NESTOR_ASYNC_H_

#ifdef VMECPP_USE_CUDA

#include <cmath>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"

namespace vmecpp {

class NestorAsyncWorker {
 public:
  // The worker is the sole user of fb after construction. bsqvac aliases fb's
  // vacuum_magnetic_pressure output span. The geometry constants size the
  // per-iteration buffers.
  NestorAsyncWorker(FreeBoundaryBase* fb, std::span<const double> bsqvac,
                    int nZnT, int nZeta, int nThetaEff)
      : fb_(fb),
        bsqvac_(bsqvac),
        nZnT_(nZnT),
        nZeta_(nZeta),
        nThetaEff_(nThetaEff) {
    in_.r1_lcfs.assign(nZnT_, 0.0);
    work_.r1_lcfs.assign(nZnT_, 0.0);
    out_.rBSq.assign(nZnT_, 0.0);
    const char* e = std::getenv("VMECPP_FB_ASYNC_INLINE");
    inline_mode_ = (e != nullptr && std::atoi(e) > 0);
    if (!inline_mode_) thread_ = std::thread([this] { Run(); });
  }

  ~NestorAsyncWorker() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      cv_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
  }

  NestorAsyncWorker(const NestorAsyncWorker&) = delete;
  NestorAsyncWorker& operator=(const NestorAsyncWorker&) = delete;

  // Snapshot the main thread fills before Submit(). The LCFS spectrum and axis
  // vectors are copied from HandoverStorage; r1_lcfs holds r1_e + r1_o on the
  // LCFS row.
  struct Inputs {
    std::vector<double> rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS;
    std::vector<double> rAxis, zAxis;
    int signOfJacobian = 1;
    double netToroidalCurrent = 0.0;
    int ivacskip = 0;
    double edgePressure = 0.0;
    double deltaS = 0.0;
    std::vector<double> r1_lcfs;
  };
  struct Outputs {
    std::vector<double> rBSq;
    double bSubUVac = 0.0;
    double bSubVVac = 0.0;
    bool ok = true;
  };

  // Main thread fills this, then calls Submit().
  Inputs& input_buffer() { return in_; }

  // Hand the filled input buffer to the worker. Requires no job in flight
  // (CollectPrevious() since the last Submit, or the first call).
  void Submit() {
    if (inline_mode_) {
      std::swap(in_, work_);
      Compute(work_, out_);
      return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    std::swap(in_, work_);  // worker reads work_; in_ is free to refill
    state_ = State::kRunning;
    cv_.notify_all();
  }

  // Wait for the in-flight job and return its result.
  const Outputs& CollectPrevious() {
    if (inline_mode_) return out_;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return state_ == State::kDone; });
    state_ = State::kIdle;
    return out_;
  }

 private:
  enum class State { kIdle, kRunning, kDone };

  // Runs the host vacuum solve and assembles the edge-pressure profile.
  // Reads `in`, writes `result`. Used by the worker thread and, under
  // VMECPP_FB_ASYNC_INLINE, directly from Submit() for debugging.
  void Compute(const Inputs& in, Outputs& result) {
    const auto sp = [](const std::vector<double>& v) {
      return std::span<const double>(v.data(), v.size());
    };
    double bsubu = 0.0;
    double bsubv = 0.0;
    fb_->update(sp(in.rCC), sp(in.rSS), sp(in.rSC), sp(in.rCS), sp(in.zSC),
                sp(in.zCS), sp(in.zCC), sp(in.zSS), in.signOfJacobian,
                sp(in.rAxis), sp(in.zAxis), &bsubu, &bsubv,
                in.netToroidalCurrent, in.ivacskip);
    result.rBSq.assign(nZnT_, 0.0);
    result.bSubUVac = bsubu;
    result.bSubVVac = bsubv;
    result.ok = true;
    for (int kl = 0; kl < nZnT_; ++kl) {
      const int k = kl / nThetaEff_;
      const int l = kl % nThetaEff_;
      const int idx_lk = l * nZeta_ + k;
      const double outside = bsqvac_[idx_lk] + in.edgePressure;
      result.rBSq[kl] = outside * in.r1_lcfs[kl] / in.deltaS;
    }
  }

  void Run() {
    while (true) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this] { return state_ == State::kRunning || stop_; });
      if (stop_) return;
      lk.unlock();  // work_ is private to the worker while kRunning

      Outputs result;
      Compute(work_, result);

      lk.lock();
      out_ = std::move(result);
      state_ = State::kDone;
      cv_.notify_all();
    }
  }

  FreeBoundaryBase* fb_;
  std::span<const double> bsqvac_;
  const int nZnT_;
  const int nZeta_;
  const int nThetaEff_;

  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  State state_ = State::kIdle;
  bool stop_ = false;
  bool inline_mode_ = false;

  Inputs in_;    // filled by the main thread
  Inputs work_;  // owned by the worker while kRunning
  Outputs out_;  // produced by the worker, read after CollectPrevious()
};

}  // namespace vmecpp

#endif  // VMECPP_USE_CUDA
#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_NESTOR_ASYNC_H_
