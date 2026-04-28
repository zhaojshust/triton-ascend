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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"

#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "TritonToUnstructure/OffsetAnalysis.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "ReorderOpsByBlockIdPass";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace triton;
using namespace CVPipeline;

namespace {

// A dependency DAG of both SSA and memory of the ops
struct BlockOpGraph {
    SmallVector<Operation *> ops;
    DenseMap<Operation *, unsigned> opIndex;               // op → position in ops
    DenseMap<Operation *, SmallVector<Operation *>> preds; // op → its defs
    DenseMap<Operation *, SmallVector<Operation *>> succs; // op → its uses
    BlockOpGraph(Block *block, const MemoryDependenceGraph &memGraph);
};

} // namespace

BlockOpGraph::BlockOpGraph(Block *block, const MemoryDependenceGraph &memGraph)
{
    llvm_unreachable("To be implemented by following commit");
}

static llvm::FailureOr<DenseMap<Operation *, int>> collectBlockIds(ArrayRef<Operation *> allOps)
{
    DenseMap<Operation *, int> opBlockId;
    for (Operation *op : allOps) {
        if (llvm::failed(verifyOpBlockId(op))) {
            return llvm::failure();
        }
        auto blockIdAttrRes = getOpBlockId(op);
        int64_t blockId =
            blockIdAttrRes.has_value() ? blockIdAttrRes.value() : ComputeBlockIdManager::getInstance().getNextId();
        opBlockId[op] = blockId;
    }
    return opBlockId;
}

/**
 * Builds a topological order of groups based on operation dependencies.
 *
 * This function collapses an operation-level dependency graph into a group-level
 * Directed Acyclic Graph (DAG) and returns the groups in a valid execution order.
 */
static SmallVector<int> buildGroupOrder(const BlockOpGraph & /*g*/, const DenseMap<Operation *, int> & /*opBlockId*/)
{
    llvm_unreachable("To be implemented by following commit");
}

// Stable sort ops based on their group orders
static SmallVector<Operation *> buildReorderedOps(const BlockOpGraph &graph,
                                                  const DenseMap<Operation *, int> &opBlockId)
{
    SmallVector<Operation *> reordered;
    SmallVector<int> const groupOrder = buildGroupOrder(graph, opBlockId);

    for (int const blockId : groupOrder) {
        for (Operation *op : graph.ops) {
            if (opBlockId.at(op) == blockId) {
                reordered.push_back(op);
            }
        }
    }
    return reordered;
}

// Reorder the ops in the mlir representation
static void applyReorder(Block &block, ArrayRef<Operation *> reordered)
{
    Operation *terminator = block.mightHaveTerminator() ? block.getTerminator() : nullptr;
    for (Operation *op : reordered) {
        op->moveBefore(&block, block.end());
    }

    if (terminator) {
        terminator->moveBefore(&block, block.end());
    }
}

static llvm::LogicalResult reorderOpsInBlock(Block &block, const MemoryDependenceGraph &memGraph)
{
    auto allOps = llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

    const BlockOpGraph graph {&block, memGraph};
    llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt = collectBlockIds(allOps);
    if (failed(opBlockIdOpt)) {
        return failure();
    }
    auto &opBlockId = *opBlockIdOpt;
    LOG_DEBUG("Initial opBlockIds:\n");
    for (Operation *op : allOps) {
        LOG_DEBUG("  Op: " << *op << ", opBlockId = " << opBlockId[op] << "\n");
    }
    SmallVector<Operation *> const reordered = buildReorderedOps(graph, opBlockId);
    applyReorder(block, reordered);

    return llvm::success();
}

void ReorderOpsByBlockIdPass::runOnOperation()
{
    LOG_DEBUG("\n=== Pass: TuningOpSeq ===\n");
    OpBuilder const builder(&getContext());

    auto moduleOp = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    auto memGraph = MemoryDependenceGraph(moduleOp, aa);
    moduleOp.walk([&](Block *block) {
        auto *parentOp = block->getParentOp();
        if (!parentOp ||
            // whitelist ops to reorder
            !(isa<func::FuncOp>(parentOp) || isa<scf::SCFDialect>(parentOp->getDialect()))) {
            return WalkResult::skip();
        }
        if (llvm::failed(reorderOpsInBlock(*block, memGraph))) {
            signalPassFailure();
        }
        return WalkResult::advance();
    });

    LOG_DEBUG("=== Pass TuningOpSeq complete ===\n");
    ComputeBlockIdManager::getInstance().reset();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createReorderOpsByBlockIdPass()
{
    return std::make_unique<ReorderOpsByBlockIdPass>();
}
