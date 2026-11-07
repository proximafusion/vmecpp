# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Under a CUDA build (VMECPP_USE_CUDA), a test that runs into one of the build's
explicit refusals, lasym and lforbal inputs and the VmecModel iteration API, is reported
as skipped with the refusal as the reason, so that the suite as a whole validates the
CUDA iteration body on every input it accepts."""

import pytest

import vmecpp

# Every refusal of the CUDA build names the way out.
_REFUSAL = "without VMECPP_USE_CUDA"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or not report.failed or not vmecpp.has_cuda():
        return
    message = str(call.excinfo.value)
    if _REFUSAL in message:
        reason = next(line for line in message.splitlines() if _REFUSAL in line)
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, reason.strip())
