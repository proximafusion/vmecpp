// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec/anderson_acceleration.h"

#include <array>
#include <span>

namespace {

// The evolved state arrays of a FourierGeometry, in a fixed order. The spans
// cover the full stored radial range of the owning thread, satellite points
// included.
std::array<std::span<double>, 12> StateSpans(const vmecpp::FourierGeometry& x,
                                             const vmecpp::Sizes& s,
                                             int& m_num_spans) {
  std::array<std::span<double>, 12> spans;
  int count = 0;
  spans[count++] = x.rmncc;
  spans[count++] = x.zmnsc;
  spans[count++] = x.lmnsc;
  if (s.lthreed) {
    spans[count++] = x.rmnss;
    spans[count++] = x.zmncs;
    spans[count++] = x.lmncs;
  }
  if (s.lasym) {
    spans[count++] = x.rmnsc;
    spans[count++] = x.zmncc;
    spans[count++] = x.lmncc;
    if (s.lthreed) {
      spans[count++] = x.rmncs;
      spans[count++] = x.zmnss;
      spans[count++] = x.lmnss;
    }
  }
  m_num_spans = count;
  return spans;
}

}  // namespace

vmecpp::AndersonAcceleration::AndersonAcceleration(const Sizes* s, int window)
    : s_(s), window_(window) {}

void vmecpp::AndersonAcceleration::Reset() {
  map_outputs_.clear();
  residuals_.clear();
}

void vmecpp::AndersonAcceleration::Pack(const FourierGeometry& x,
                                        Eigen::VectorXd& m_out) const {
  int num_spans = 0;
  const auto spans = StateSpans(x, *s_, num_spans);
  Eigen::Index total = 0;
  for (int i = 0; i < num_spans; ++i) {
    total += static_cast<Eigen::Index>(spans[i].size());
  }
  m_out.resize(total);
  Eigen::Index offset = 0;
  for (int i = 0; i < num_spans; ++i) {
    for (const double value : spans[i]) {
      m_out[offset++] = value;
    }
  }
}

void vmecpp::AndersonAcceleration::CapturePreStep(const FourierGeometry& x) {
  Pack(x, pre_step_);
}

void vmecpp::AndersonAcceleration::PushPostStep(const FourierGeometry& x) {
  Eigen::VectorXd post;
  Pack(x, post);
  residuals_.push_back(post - pre_step_);
  map_outputs_.push_back(std::move(post));
  while (static_cast<int>(map_outputs_.size()) > window_ + 1) {
    map_outputs_.pop_front();
    residuals_.pop_front();
  }
}

int vmecpp::AndersonAcceleration::NumDifferences() const {
  return static_cast<int>(map_outputs_.size()) - 1;
}

void vmecpp::AndersonAcceleration::LocalNormalEquations(double* m_gram,
                                                        double* m_rhs) const {
  const int k = NumDifferences();
  const Eigen::VectorXd& current = residuals_.back();
  for (int i = 0; i < k; ++i) {
    const Eigen::VectorXd delta_i = residuals_[i + 1] - residuals_[i];
    for (int j = i; j < k; ++j) {
      const double dot = delta_i.dot(residuals_[j + 1] - residuals_[j]);
      m_gram[i * k + j] = dot;
      m_gram[j * k + i] = dot;
    }
    m_rhs[i] = delta_i.dot(current);
  }
}

void vmecpp::AndersonAcceleration::ApplyCombination(
    const double* gamma, FourierGeometry& m_x) const {
  const int k = NumDifferences();
  Eigen::VectorXd accelerated = map_outputs_.back();
  for (int i = 0; i < k; ++i) {
    accelerated -= gamma[i] * (map_outputs_[i + 1] - map_outputs_[i]);
  }

  int num_spans = 0;
  const auto spans = StateSpans(m_x, *s_, num_spans);
  Eigen::Index offset = 0;
  for (int i = 0; i < num_spans; ++i) {
    for (double& value : spans[i]) {
      value = accelerated[offset++];
    }
  }
}
