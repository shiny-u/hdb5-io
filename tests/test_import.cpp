//
// Created by camille on 21/09/23.
//

#include "hdb5_io/hdb5_io.h"

using namespace hdb5_io;

int main(int argc, char* argv[]) {

  auto filename = argv[1];
  auto hdb = import_HDB(filename);

  return 0;
}