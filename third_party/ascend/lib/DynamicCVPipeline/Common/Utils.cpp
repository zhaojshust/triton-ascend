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

#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir {
namespace CVSplit {

bool isWholeCubeReady(Operation *seedOp, DenseMap<Operation *, int> &indegree)
{
    auto id = CVSplit::lookupOpBlockId(seedOp);
    if (id == -1) {
        return (indegree[seedOp] == 0);
    }
    auto &bm = CVSplit::ComputeBlockIdManager::getInstance();
    auto cubeBlock = bm.getOpsByBlockId(id);
    for (auto op : cubeBlock) {
        if (indegree[op] != 0) {
            return false;
        }
    }
    return true;
}

bool isSameBlock(Operation *a, Operation *b)
{
    auto &bm = CVSplit::ComputeBlockIdManager::getInstance();
    if (bm.getBlockIdByOp(a) == bm.getBlockIdByOp(b) && bm.getBlockIdByOp(a) != -1) {
        return true;
    }
    return false;
}

void initializeIndegreeForBlock(Block *block, DenseMap<Operation *, int> &indegree,
                                const MemoryDependenceGraph &memGraph)
{
    block->walk([&](Operation *op) {
        if (op->getBlock() == block) {
            indegree[op] = 0;
            op->walk([&](Operation *nestedOp) {
                for (auto inValue : nestedOp->getOperands()) {
                    if (auto defOp = inValue.getDefiningOp()) {
                        if (defOp->getBlock() == block && !CVSplit::isSameBlock(defOp, op))
                            indegree[op]++;
                    }
                }

                for (auto memDepUser : memGraph.getExecBefore(nestedOp)) {
                    if (memDepUser->getBlock() == block && !CVSplit::isSameBlock(memDepUser, op))
                        indegree[op]++;
                }
            });
        }
    });
}

Operation *getAncestorInBlock(Operation *inner, Block *block)
{
    Operation *cur = inner;
    while (cur) {
        if (cur->getBlock() == block) {
            return cur;
        }
        cur = cur->getParentOp();
    }
    return nullptr;
}

const char *literalCoreType(CoreType ct)
{
    switch (ct) {
        case VECTOR_ONLY:
            return "VECTOR_ONLY";
        case CUBE_ONLY:
            return "CUBE_ONLY";
        case CUBE_AND_VECTOR:
            return "CUBE_AND_VECTOR";
        case UNDETERMINED:
            return "UNDETERMINED";
    }
    return "Unknown";
}

CoreType lookupOpCoreType(Operation *op)
{
    if (!op) {
        return UNDETERMINED;
    }
    if (auto a = op->getAttrOfType<StringAttr>(attr::kCoreType)) {
        return fromStrCoreType(a.getValue());
    }
    return UNDETERMINED;
}

void markOpCoreType(Operation *op, CoreType ct)
{
    op->setAttr(attr::kCoreType, StringAttr::get(op->getContext(), toStrCoreType(ct)));
}

int lookupOpBlockId(Operation *op)
{
    if (!op) {
        return -1;
    }
    if (auto a = op->getAttrOfType<IntegerAttr>(attr::kBlockId)) {
        return a.getInt();
    }
    return -1;
}

void markOpBlockId(Operation *op, int blockId)
{
    if (!op) {
        return;
    }
    MLIRContext *ctx = op->getContext();
    op->setAttr(attr::kBlockId, IntegerAttr::get(IntegerType::get(ctx, 32), blockId));
    ComputeBlockIdManager::getInstance().record(op, blockId);
}

llvm::SmallVector<Operation *> ComputeBlockIdManager::getOpsByBlockId(int blockId)
{
    if (blockId == -1) {
        return {};
    }
    auto it = blockIdToOps.find(blockId);
    if (it == blockIdToOps.end()) {
        return {};
    }
    return llvm::SmallVector<Operation *>(it->second.begin(), it->second.end());
}

int ComputeBlockIdManager::getBlockIdByOp(Operation *op)
{
    if (!op) {
        return -1;
    }
    auto it = opToBlockId.find(op);
    if (it != opToBlockId.end()) {
        return it->second;
    }
    int id = lookupOpBlockId(op);
    if (id >= 0) {
        record(op, id);
    }
    return id;
}

void ComputeBlockIdManager::markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops)
{
    int id = static_cast<int>(getNextId());
    for (Operation *op : ops) {
        if (!op) {
            continue;
        }
        markOpBlockId(op, id);
    }
}

void ComputeBlockIdManager::record(Operation *op, int blockId)
{
    if (!op || blockId < 0) {
        return;
    }
    auto itOld = opToBlockId.find(op);
    if (itOld != opToBlockId.end()) {
        int oldId = itOld->second;
        if (oldId != blockId) {
            auto itVec = blockIdToOps.find(oldId);
            if (itVec != blockIdToOps.end()) {
                auto &vec = itVec->second;
                for (auto i = vec.begin(); i != vec.end(); ++i) {
                    if (*i == op) {
                        vec.erase(i);
                        break;
                    }
                }
                if (vec.empty()) {
                    blockIdToOps.erase(itVec);
                }
            }
        }
    }

    opToBlockId[op] = blockId;
    auto &vec = blockIdToOps[blockId];
    for (Operation *existing : vec) {
        if (existing == op) {
            return;
        }
    }
    vec.push_back(op);
}

} // namespace CVSplit
} // namespace mlir
