
#include <iostream>
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "util/file_io/file_io.h"

int main() {
  auto indata_json = ReadFile("vmecpp/test_data/solovev.json");
  if (!indata_json.ok()) {
    std::cerr << "Failed to read file: " << indata_json.status() << std::endl;
    return 1;
  }

  auto indata = vmecpp::VmecINDATA::FromJson(*indata_json);
  if (!indata.ok()) {
    std::cerr << "Failed to parse JSON: " << indata.status() << std::endl;
    return 1;
  }

  std::cout << "Success! mpol=" << indata->mpol << ", ntor=" << indata->ntor << std::endl;
  std::cout << "rbc size=" << indata->rbc.size() << ", expected=" << (indata->mpol + 1) * (2 * indata->ntor + 1) << std::endl;
  return 0;
}

