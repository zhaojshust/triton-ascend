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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

static constexpr const char *DEBUG_TYPE = "UpdateLoopIterTimes";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace hivm;

void UpdateLoopIterTimesPass::runOnOperation() {
  ModuleOp module = getOperation();

  LDBG("\nEnter UpdateLoopIterTimes pass!");
  LLVM_DEBUG(DBGS() << "before updateloopitertimes:\n" << module << "\n");

  // // step1: Get mainloop id to loop operation mapping, separated into cube and vector
  // DenseMap<int, SmallVector<Operation *>> cmap;
  // DenseMap<int, SmallVector<Operation *>> vmap;
  // GetMainLoopIdToLoopOpMap(module, cmap, vmap);

  // // step2: Calculate info for each loop operation, store into the same iterationTimesinfoMap
  // DenseMap<Operation *, IterationTimesInfo> infoMap;
  // ComputeMainLoopTimes(cmap, infoMap);
  // ComputeMainLoopTimes(vmap, infoMap);

  // // step3: Update loop iteration count (process loop operations with same id from both cmap and vmap)
  // UpdateForLoopIteration(cmap, vmap, infoMap);
  // LDBG("Update ForLoop Iteration success!\n");
  // LLVM_DEBUG(DBGS() << "after UpdateForLoopIteration:\n" << module << "\n");

  // // step4: Replace loop counter by if blocks' counter
  // replaceForOpCounterInIfOps();
  // LLVM_DEBUG(DBGS() << "replaceForOpCounterInIfOps success!\n");

  LLVM_DEBUG(DBGS() << "after updateloopitertimes:\n" << module << "\n");
  LDBG("\nExit UpdateLoopIterTimes pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopIterTimesPass() {
  return std::make_unique<UpdateLoopIterTimesPass>();
}
} // namespace triton
} // namespace mlir