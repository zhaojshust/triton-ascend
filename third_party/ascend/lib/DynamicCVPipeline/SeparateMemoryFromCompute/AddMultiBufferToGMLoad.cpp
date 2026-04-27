/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferToGMLoad";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

namespace {

/// A single marked op with its dependency chain inside the enclosing region.
struct MarkedLoad {
  Operation *markedOp;
  SmallVector<Operation *> chain;
  memref::AllocOp allocOp = nullptr;
};

} // anonymous namespace

void AddMultiBufferToGMLoadPass::runOnOperation()
{
  auto module = getOperation();
  LDBG("Enter AddMultiBufferToGMLoad pass");

  // All marked ops collected from the module IR.
  SmallVector<MarkedLoad> markedLoads;

  // Step 1: Scan the module IR for ops carrying the `gm_load_bufferable`
  //   attribute, and compute the dependency chain for each marked op.

  if (markedLoads.empty()) {
    LDBG("No marked loads found, nothing to transform");
    return;
  }

  LDBG("Marked loads collected, start transformation");

  // Step 2: For each marked load, apply multi-buffer transformation.
  //   Allocate buffer slots, build producer/consumer logic, and rewrite
  //   the original op to consume data from the selected buffer slot.

  // Step 3: Clean up transformed IR.
  //   Erase replaced original ops and prune dead values introduced
  //   during the transformation.

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferToGMLoadPass()
{
  return std::make_unique<AddMultiBufferToGMLoadPass>();
}

} // namespace triton
} // namespace mlir
