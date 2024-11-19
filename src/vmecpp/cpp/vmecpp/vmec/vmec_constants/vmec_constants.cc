#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

vmecpp::VmecConstants::VmecConstants() { reset(); }

void vmecpp::VmecConstants::reset() {
  rmsPhiP = 0.;
  lamscale = 0.;
}
