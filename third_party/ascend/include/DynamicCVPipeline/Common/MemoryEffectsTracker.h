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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_MEMORY_EFFECTS_TRACKER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_MEMORY_EFFECTS_TRACKER_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace CVSplit {

const uint32_t USUALLY_READ_TIMES = 4;

class MemoryDependenceGraph {
public:
    MemoryDependenceGraph(Operation *root, AliasAnalysis &aa);

    ArrayRef<Operation *> getMemDefs(Operation *op) const;
    ArrayRef<Operation *> getMemUsers(Operation *op) const;

    ArrayRef<Operation *> getExecBefore(Operation *op) const;
    ArrayRef<Operation *> getExecAfter(Operation *op) const;

    /// Full tables -- useful for iterating over every op that has dep edges.
    const DenseMap<Operation *, SmallVector<Operation *>> &allDefs() const { return memDefs; }
    const DenseMap<Operation *, SmallVector<Operation *>> &allUsers() const { return memUsers; }

    const DenseMap<Operation *, SmallVector<Operation *>> &allBefore() const { return execBefore; }
    const DenseMap<Operation *, SmallVector<Operation *>> &allAfter() const { return execAfter; }

private:
    struct MemSlot {
        Value memref;
        Operation *lastWriter = nullptr;
        SmallPtrSet<Operation *, USUALLY_READ_TIMES> pendingReads;
        bool rejectMayAlias = false; // if true, it means MayAlias is considered as no alias.
        explicit MemSlot(Value v) : memref(v) {}
    };

    struct Snapshot {
        struct SlotState {
            Value memref;
            Operation *lastWriter;
            SmallPtrSet<Operation *, USUALLY_READ_TIMES> pendingReads;
        };
        SmallVector<SlotState> states;
    };

    void analyzeOp(Operation *op);
    void analyzeRegionsOf(Operation *op);

    SmallVector<MemoryEffects::EffectInstance> collectOuterEffects(Operation *op, bool &unknown);

    SmallVector<MemSlot *> findAliasSlots(Value v);
    MemSlot *getOrCreateSlot(Value v);

    void collectPreds(ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown,
                      SmallVectorImpl<Operation *> &defsOut, SmallVectorImpl<Operation *> &predsOut);

    void applyEffects(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown);

    Snapshot takeSnapshot() const;
    void restoreSnapshot(Snapshot &&snap);

    void recordEdges(Operation *op, ArrayRef<Operation *> defs, ArrayRef<Operation *> preds);

    Operation *root;
    AliasAnalysis &aa;
    DenseMap<Operation *, SmallVector<Operation *>> memDefs;
    DenseMap<Operation *, SmallVector<Operation *>> memUsers;
    DenseMap<Operation *, SmallVector<Operation *>> execBefore;
    DenseMap<Operation *, SmallVector<Operation *>> execAfter;
    SmallVector<std::unique_ptr<MemSlot>> slots;
    DenseMap<Value, MemSlot *> valueToSlot;
};

} // namespace CVSplit
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_MEMORY_EFFECTS_TRACKER_H
