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

#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferInnerScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace hivm;
using namespace annotation;

using BufferPair = std::pair<Value, Value>;
constexpr int kInlineBufferSize = 8;
using BufferMap = DenseMap<Value, SmallVector<BufferPair, kInlineBufferSize>>;

namespace mlir {
namespace triton {
namespace AddMultiBufferInnerScope {

    struct BlockInfo {
        Value blockId;
        SmallVector<Operation *, 32> ops;
    };

    int getSsbufferId(Operation *op) {
        if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
            return idAttr.getInt();
        if (auto idAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.id"))
            return idAttr.getInt();
        return -1;
    }

    void collectNestedOps(Block *block, SmallVector<Operation *, 32> &ops) {
        for (auto &op : *block) {
            ops.push_back(&op);
            for (auto &region : op.getRegions())
                for (auto &innerBlock : region)
                    collectNestedOps(&innerBlock, ops);
        }
    }

    // 获取 forOp 的优先级（数值越小优先级越高）
    static int getForOpPriority(scf::ForOp f) {
        bool hasMainloop = f->hasAttr("ssbuffer.main_loop");
        bool bodyHasMainloop = false;
        bool bodyHasBlockId = false;

        if (auto *term = f.getBody()->getTerminator()) {
            bodyHasMainloop = term->hasAttr("ssbuffer.main_loop");
            bodyHasBlockId = term->getAttrOfType<IntegerAttr>("ssbuffer.block_id") != nullptr;
        }

        bool opHasBlockId = f->getAttrOfType<IntegerAttr>("ssbuffer.block_id") != nullptr;
        bool hasIterArgs = f.getNumResults() > 0 || !f.getInitArgs().empty();

        if (hasMainloop || bodyHasMainloop)
            return 1;
        if (opHasBlockId || bodyHasBlockId)
            return 2;
        if (hasIterArgs)
            return 3;
        return 0;
    }

    scf::ForOp findMainloopInScope(scope::ScopeOp scope) {
        SmallVector<Operation *, 32> allOps;
        collectNestedOps(&scope.getBodyRegion().front(), allOps);

        scf::ForOp mainLoopForOp;
        int bestPriority = INT_MAX;

        for (Operation *op : allOps) {
            auto f = dyn_cast<scf::ForOp>(op);
            if (!f) continue;

            int priority = getForOpPriority(f);
            if (priority > 0 && priority < bestPriority) {
                mainLoopForOp = f;
                bestPriority = priority;
            }
        }
        return mainLoopForOp;
    }

    // 收集单个依赖值到 depValueMap
    static void collectDepValue(Value operand, Block *body, int currentBlockId,
                            DenseMap<Value, int> &outputToBlockId,
                            DenseMap<Value, SmallVector<Value, 8>> &depValueMap,
                            Value groupKey) {
        if (auto barg = dyn_cast<BlockArgument>(operand)) {
            if (barg.getOwner() == body &&
                !llvm::is_contained(depValueMap[groupKey], barg))
                depValueMap[groupKey].push_back(barg);
            return;
        }

        if (outputToBlockId.count(operand) &&
            outputToBlockId[operand] != currentBlockId &&
            !llvm::is_contained(depValueMap[groupKey], operand))
            depValueMap[groupKey].push_back(operand);
    }

    // 递归查找嵌套的 main_loop
    static scf::ForOp findNestedMainloopInForOp(scf::ForOp forOp) {
        SmallVector<Operation *, 32> allOps;
        collectNestedOps(forOp.getBody(), allOps);

        for (Operation *op : allOps) {
            auto nestedFor = dyn_cast<scf::ForOp>(op);
            if (!nestedFor) continue;
            if (nestedFor->hasAttr("ssbuffer.main_loop"))
                return nestedFor;
        }
        return {};
    }

    bool isInsideMainLoopForOp(Operation *op) {
        Operation *parent = op->getParentOp();
        if (!parent) return false;
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            return forOp->hasAttr("ssbuffer.main_loop");
        }
        return false;
    }

    bool isInsideMainLoopForOpTraverse(Operation *op) {
        Operation *parent = op->getParentOp();
        while (parent) {
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
                if (forOp->hasAttr("ssbuffer.main_loop")) {
                    return true;
                }
            }
            parent = parent->getParentOp();
        }
        return false;
    }

    DenseMap<Value, SmallVector<Value, 8>>
    collectBlockInfo(scf::ForOp forOp, DenseMap<Value, BlockInfo> &blocks) {
        DenseMap<Value, SmallVector<Value, 8>> depValueMap;
        Block *body = forOp.getBody();
        if (!body) return depValueMap;

        SmallVector<Operation *, 32> allOps;
        collectNestedOps(body, allOps);

        // 1. 收集所有有 ssbuffer.id 的 op 及其结果
        llvm::MapVector<Value, Operation *> opsByValue;
        for (Operation *op : allOps) {
            int id = getSsbufferId(op);
            if (id < 0) continue;
            for (auto res : op->getResults())
                opsByValue[res] = op;
        }
        if (opsByValue.empty()) return depValueMap;

        // 2. 按 ssbuffer id 分组
        llvm::MapVector<int, SmallVector<Operation *, 32>> opsById;
        for (auto &p : opsByValue) {
            int id = getSsbufferId(p.second);
            if (id >= 0) opsById[id].push_back(p.second);
        }

        // 3. 建立输出到 block id 的映射
        DenseMap<Value, int> outputToBlockId;
        for (auto &p : opsById)
            for (Operation *op : p.second)
                for (auto res : op->getResults())
                    outputToBlockId[res] = p.first;

        // 4. 为每个 block 收集依赖值
        for (auto &p : opsById) {
            Value groupKey = p.second.front()->getResult(0);
            BlockInfo bi;
            bi.blockId = groupKey;
            bi.ops = p.second;
            blocks[groupKey] = bi;

            for (Operation *op : bi.ops)
                for (Value operand : op->getOperands())
                    collectDepValue(operand, body, p.first, outputToBlockId, depValueMap, groupKey);
        }

        return depValueMap;
    }

    DenseMap<Value, SmallVector<Operation *, 8>>
    buildDepUserMap(DenseMap<Value, BlockInfo> &blocks) {
        DenseMap<Value, SmallVector<Operation *, 8>> depUserMap;
        for (auto &p : blocks)
            for (Operation *op : p.second.ops)
                for (Value operand : op->getOperands())
                    depUserMap[operand].push_back(op);
        return depUserMap;
    }

    SmallVector<Value, 8>
    collectBufferValues(DenseMap<Value, SmallVector<Value, 8>> &depValueMap) {
        SmallVector<Value, 8> valueList;
        SmallVector<void *, 8> seenPtrs;

        for (auto &p : depValueMap) {
            for (Value depVal : p.second) {
                if (depVal.getDefiningOp() &&
                    !llvm::is_contained(seenPtrs, depVal.getAsOpaquePointer())) {
                    seenPtrs.push_back(depVal.getAsOpaquePointer());

                    auto shapedType = dyn_cast<ShapedType>(depVal.getType());
                    if (!shapedType) continue;

                    valueList.push_back(depVal);
                }
            }
        }

        return valueList;
    }

    SmallVector<Value, 8>
    collectScalarDeps(DenseMap<Value, SmallVector<Value, 8>> &depValueMap,
                    DenseMap<Value, SmallVector<Operation *, 8>> &depUserMap) {
        SmallVector<Value, 8> scalarValueList;

        for (auto &p : depValueMap) {
            for (Value depVal : p.second) {
                if (isa<BlockArgument>(depVal)) continue;

                Operation *depDefinedOp = depVal.getDefiningOp();
                if (!depDefinedOp) continue;

                if (isa<ShapedType>(depVal.getType())) continue;

                auto userIt = depUserMap.find(depVal);
                if (userIt == depUserMap.end()) continue;

                int producerId = getSsbufferId(depDefinedOp);
                if (producerId == -1) continue;

                SmallVector<Operation *, 8> depUsers = userIt->second;
                bool hasCrossBlockUser = false;
                for (Operation *depUser : depUsers) {
                    if (getSsbufferId(depUser) != producerId) {
                        hasCrossBlockUser = true;
                        break;
                    }
                }

                if (hasCrossBlockUser)
                    scalarValueList.push_back(depVal);
            }
        }

        return scalarValueList;
    }

} // namespace AddMultiBufferInnerScope

} // namespace triton
} // namespace mlir

void mlir::triton::AddMultiBufferInnerScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    OpPassManager pm(module.getOperationName());
    LDBG("Enter pass.");

    // Step 1: Walk scope operations

    // Step 2: Find main loop in each scope

    // Step 3: Apply inner multi-buffer optimization

    if (failed(runPipeline(pm, module))) {
        module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
        signalPassFailure();
    }

    LDBG("Process successfully");
}

void mlir::triton::AddMultiBufferInnerScopePass::getDependentDialects(
    DialectRegistry &registry) const
{
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect,
                    linalg::LinalgDialect, bufferization::BufferizationDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createAddMultiBufferInnerScopePass()
{
    return std::make_unique<AddMultiBufferInnerScopePass>();
}