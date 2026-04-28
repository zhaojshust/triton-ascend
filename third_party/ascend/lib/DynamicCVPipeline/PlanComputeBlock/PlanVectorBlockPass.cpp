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

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

#define DEBUG_TYPE "plan-vector-block"
#define LOG_DEBUG(msg) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

using namespace mlir;
using namespace triton;

bool isFusableOp(Operation *op)
{
    if (CVPipeline::lookupOpCoreType(op) == CVPipeline::CoreType::VECTOR_ONLY) {
        if (isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::WhileOp>(op)) {
            return false;
        }
        if (op == op->getBlock()->getTerminator()) {
            return false;
        }
        return true;
    }
    return false;
}


void passAndCollectCandidates(Operation *nowOp, DenseMap<Operation *, int> &indegree, SmallVector<Operation *> &candidates,
                         DenseMap<Operation *, bool> &visited, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    LOG_DEBUG("Bypassing non-fusable op " << *nowOp << "\nnow candidates size: " << candidates.size() << "\n");
    auto block = nowOp->getBlock();
    SmallVector<Operation *> allusers;
    allusers.append(nowOp->getUsers().begin(), nowOp->getUsers().end());
    for (auto memUser : memGraph.getExecAfter(nowOp)) {
        allusers.push_back(memUser);
    }
    for (auto user : allusers) {
        auto userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (!userInBlock)
            continue;
        if (!bm.isSameBlock(userInBlock, nowOp)) {
            indegree[userInBlock]--;
        }

        if (bm.isWholeCubeReady(userInBlock, indegree)) {
            if (!isFusableOp(userInBlock) && !visited[userInBlock]) {
                if (bm.getBlockIdByOp(userInBlock) == -1) {
                    visited[userInBlock] = true; // mark as fused to avoid duplicate bypass
                    passAndCollectCandidates(userInBlock, indegree, candidates, visited, memGraph);
                } else {
                    for (auto cubeop : bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock))) {
                        if (!visited[cubeop]) {
                            visited[cubeop] = true;
                            passAndCollectCandidates(cubeop, indegree, candidates, visited, memGraph);
                        }
                    }
                }
            } else if (isFusableOp(userInBlock) && !visited[userInBlock]) {
                visited[userInBlock] = true;
                candidates.push_back(userInBlock);
            }
        }
    }

    LOG_DEBUG("After bypassing, candidates size: " << candidates.size() << "\n");
}

void byPassNonFusable(DenseMap<Operation *, int> &indegree, SmallVector<Operation *> &candidates,
                      DenseMap<Operation *, bool> &visited, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // for every non-fusable candidates, bypass it.
    for (auto &[op, degree] : indegree) {
        auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
        if (bm.isWholeCubeReady(op, indegree) && !isFusableOp(op) && !visited[op]) {
            if (bm.getBlockIdByOp(op) == -1) {
                visited[op] = true; // mark as fused to avoid duplicate bypass
                passAndCollectCandidates(op, indegree, candidates, visited, memGraph);
            } else {
                for (auto cubeop : bm.getOpsByBlockId(bm.getBlockIdByOp(op))) {
                    if (!visited[cubeop]) {
                        visited[cubeop] = true;
                        passAndCollectCandidates(cubeop, indegree, candidates, visited, memGraph);
                    }
                }
            }
        }
    }
    
}

void updateCandidates(Operation *nextFused, SmallVector<Operation *> &candidates, DenseMap<Operation *, int> &indegree,
                      DenseMap<Operation *, bool> &visited, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // 1. Already fuse with nextFused, so remove it from candidates
    auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
        if (*it == nextFused) {
            it = candidates.erase(it);
            break;
        }
    }
    
    // 2. Add new candidates whose indegree becomes 0 after fusing nextFused.
    auto block = nextFused->getBlock();
    SmallVector<Operation *> allusers;
    allusers.append(nextFused->getUsers().begin(), nextFused->getUsers().end());
    for (auto memUser : memGraph.getExecAfter(nextFused)) {
        allusers.push_back(memUser);
    }
    for (auto user : allusers) {
        auto userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (!userInBlock)
            continue;
        if (!visited[userInBlock]) {
            indegree[userInBlock]--;
            if (indegree[userInBlock] == 0 && isFusableOp(userInBlock)) {
                visited[userInBlock] = true;
                candidates.push_back(userInBlock);
            }
        }
    }
}

void findCandidates(DenseMap<Operation *, int> &indegree, SmallVector<Operation *> &candidates,
                    DenseMap<Operation *, bool> &visited, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // 1. if no candidate, try to bypass non-fusable
    LOG_DEBUG("Finding source ops............\n");
    if (candidates.empty()) {
        LOG_DEBUG("No candidates available, try bypass" << "\n");
        byPassNonFusable(indegree, candidates, visited, memGraph);
    }
    // 2. find candidates whose indegree is 0 and not visited, add them to candidates and mark visited
    for (auto &[op, degree] : indegree) {
        if (degree == 0 && isFusableOp(op) && !visited[op]) {
            visited[op] = true;
            candidates.push_back(op);
        }
    }
    LOG_DEBUG("end finding source ops............\n");
}

void planVectorBlockId(Block *block, const CVPipeline::MemoryDependenceGraph &memGraph)
{
    // 1. topo initialize
    llvm::DenseMap<Operation *, int> indegree;
    llvm::SmallVector<Operation *> queue;
    llvm::DenseMap<Operation *, bool> visited; // has been visited in search
    CVPipeline::initializeIndegreeForBlock(block, indegree, memGraph);

    // 2. initialize visited and find initial candidates
    block->walk([&](Operation *op) {
        if (op->getBlock() == block) {
            visited[op] = false;
            if (isFusableOp(op) && indegree[op] == 0) {
                visited[op] = true;
                queue.push_back(op);
            }
        }
    });
    findCandidates(indegree, queue, visited, memGraph);  
    
    // 3. find fuse group follow topo order
    llvm::SmallVector<Operation *> nowFuseGroup;
    while (!queue.empty()) {
        auto nextFused = queue.front();
        if (nextFused) {
            // fused one && update candidates
            nowFuseGroup.push_back(nextFused);
            updateCandidates(nextFused, queue, indegree, visited, memGraph);
        }
        if (queue.empty() || nextFused == nullptr) {
            // finish one group, assign block id and start next BFS
            // Cut error operations before assigning block id

            auto &bm = CVPipeline::ComputeBlockIdManager::getInstance();
            bm.markOpsWithNewId(nowFuseGroup);
            nowFuseGroup.clear();
            // reset queue
            findCandidates(indegree, queue, visited, memGraph);
        }
    }
}

// namespace

namespace mlir {
namespace triton {

void PlanVectorBlockPass::runOnOperation()
{
    ModuleOp module = getOperation();
    // 1. Build memory dependence graph
    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

    // 2. search blocks in topo order and assign block id for each block
    llvm::SmallVector<Block *> blocks;
    module.walk([&](Block *block) { planVectorBlockId(block, memDepGraph); });
}

std::unique_ptr<OperationPass<ModuleOp>> createPlanVectorBlockPass()
{
    return std::make_unique<PlanVectorBlockPass>();
}

} // namespace triton
} // namespace mlir
