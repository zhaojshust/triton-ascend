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

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::CVSplit;

namespace {

bool isDefinedInside(Value v, Operation *op)
{
    if (!v || !op) {
        return false;
    }

    if (auto arg = dyn_cast<BlockArgument>(v)) {
        Operation *parent = arg.getOwner()->getParentOp();
        return parent && op->isAncestor(parent);
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        return false;
    }

    return op->isProperAncestor(defOp);
}

} // namespace

MemoryDependenceGraph::MemoryDependenceGraph(Operation *root, AliasAnalysis &aa) : root(root), aa(aa)
{
    if (!root) {
        return;
    }
    analyzeOp(root);
    slots.clear();
    valueToSlot.clear();
}

ArrayRef<Operation *> MemoryDependenceGraph::getMemDefs(Operation *op) const
{
    auto it = memDefs.find(op);
    if (it == memDefs.end())
        return {};
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getMemUsers(Operation *op) const
{
    auto it = memUsers.find(op);
    if (it == memUsers.end())
        return {};
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getExecBefore(Operation *op) const
{
    auto it = execBefore.find(op);
    if (it == execBefore.end())
        return {};
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getExecAfter(Operation *op) const
{
    auto it = execAfter.find(op);
    if (it == execAfter.end())
        return {};
    return it->second;
}

void MemoryDependenceGraph::analyzeOp(Operation *op)
{
    if (!op) {
        return;
    }

    // The graph root is treated as a pure container: we don't query its own
    // effects against itself, only recurse into its regions.
    if (op == root) {
        analyzeRegionsOf(op);
        return;
    }

    bool unknown = false;
    SmallVector<MemoryEffects::EffectInstance> effects = collectOuterEffects(op, unknown);

    SmallVector<Operation *> defs;
    SmallVector<Operation *> preds;
    collectPreds(effects, unknown, defs, preds);

    recordEdges(op, defs, preds);

    if (op->getNumRegions() > 0) {
        analyzeRegionsOf(op);
    }

    applyEffects(op, effects, unknown);
}

void MemoryDependenceGraph::analyzeRegionsOf(Operation *op)
{
    const bool isolated = op != root && op->hasTrait<OpTrait::IsIsolatedFromAbove>();

    for (Region &region : op->getRegions()) {
        Snapshot snap = takeSnapshot();
        if (isolated) {
            slots.clear();
            valueToSlot.clear();
        }

        for (Block &block : region) {
            for (BlockArgument arg : block.getArguments()) {
                if (isa<BaseMemRefType>(arg.getType())) {
                    auto *slot = getOrCreateSlot(arg);
                    if (op == root) {
                        slot->rejectMayAlias = true;
                    }
                }
            }
            for (Operation &inner : block) {
                analyzeOp(&inner);
            }
        }

        restoreSnapshot(std::move(snap));
    }
}

SmallVector<MemoryEffects::EffectInstance> MemoryDependenceGraph::collectOuterEffects(Operation *op, bool &unknown)
{
    unknown = false;

    std::optional<SmallVector<MemoryEffects::EffectInstance>> raw = getEffectsRecursively(op);
    if (!raw) {
        unknown = true;
        return {};
    }

    SmallVector<MemoryEffects::EffectInstance> filtered;
    filtered.reserve(raw->size());
    for (auto &e : *raw) {
        if (isDefinedInside(e.getValue(), op)) {
            continue;
        }
        filtered.push_back(e);
    }
    return filtered;
}

SmallVector<MemoryDependenceGraph::MemSlot *> MemoryDependenceGraph::findAliasSlots(Value v)
{
    SmallVector<MemSlot *> result;

    if (!v) {
        result.reserve(slots.size());
        for (auto &slot : slots) {
            result.push_back(slot.get());
        }
        return result;
    }

    SmallPtrSet<MemSlot *, 4> seen;

    auto it = valueToSlot.find(v);
    if (it != valueToSlot.end()) {
        result.push_back(it->second);
        seen.insert(it->second);
    }

    for (auto &slot : slots) {
        MemSlot *raw = slot.get();
        if (seen.contains(raw) || !raw->memref) {
            continue;
        }
        auto aliasResult = aa.alias(raw->memref, v);
        if (aliasResult != AliasResult::NoAlias) {
            if (raw->rejectMayAlias && aliasResult == AliasResult::MayAlias) {
                continue;
            }
            result.push_back(raw);
            seen.insert(raw);
        }
    }

    return result;
}

MemoryDependenceGraph::MemSlot *MemoryDependenceGraph::getOrCreateSlot(Value v)
{
    if (!v) {
        return nullptr;
    }
    auto it = valueToSlot.find(v);
    if (it != valueToSlot.end()) {
        return it->second;
    }
    auto slot = std::make_unique<MemSlot>(v);
    MemSlot *raw = slot.get();
    slots.push_back(std::move(slot));
    valueToSlot[v] = raw;
    return raw;
}

void MemoryDependenceGraph::collectPreds(ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown,
                                         SmallVectorImpl<Operation *> &defsOut,
                                         SmallVectorImpl<Operation *> &predsOut)
{
    llvm::SmallSetVector<Operation *, 8> defs;
    llvm::SmallSetVector<Operation *, 8> preds;

    auto addFromSlot = [&](MemSlot *slot, bool isWriteLike) {
        if (slot->lastWriter) {
            preds.insert(slot->lastWriter);
            if (!isWriteLike) {
                defs.insert(slot->lastWriter);
            }
        }
        if (isWriteLike) {
            for (Operation *r : slot->pendingReads) {
                preds.insert(r);
            }
        }
    };

    if (unknown) {
        for (auto &slot : slots) {
            addFromSlot(slot.get(), true);
        }
        predsOut.assign(preds.begin(), preds.end());
        return;
    }

    DenseMap<Value, SmallVector<MemSlot *>> cache;
    auto slotsFor = [&](Value v) -> ArrayRef<MemSlot *> {
        auto it = cache.find(v);
        if (it != cache.end()) {
            return it->second;
        }
        auto [ins, _] = cache.try_emplace(v, findAliasSlots(v));
        return ins->second;
    };

    for (const auto &e : effects) {
        Value v = e.getValue();
        if (isa<MemoryEffects::Read>(e.getEffect())) {
            for (MemSlot *s : slotsFor(v)) {
                addFromSlot(s, false);
            }
        } else if (isa<MemoryEffects::Write>(e.getEffect()) || isa<MemoryEffects::Free>(e.getEffect())) {
            for (MemSlot *s : slotsFor(v)) {
                addFromSlot(s, true);
            }
        }
    }

    defsOut.assign(defs.begin(), defs.end());
    predsOut.assign(preds.begin(), preds.end());
}

void MemoryDependenceGraph::applyEffects(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown)
{
    if (unknown) {
        for (auto &slot : slots) {
            slot->lastWriter = op;
            slot->pendingReads.clear();
        }
        return;
    }

    DenseMap<Value, SmallVector<MemSlot *>> cache;
    auto slotsFor = [&](Value v) -> ArrayRef<MemSlot *> {
        auto it = cache.find(v);
        if (it != cache.end()) {
            return it->second;
        }
        auto [ins, _] = cache.try_emplace(v, findAliasSlots(v));
        return ins->second;
    };

    for (const auto &e : effects) {
        // Read leads. For the op reading and writting the same memory.
        if (isa<MemoryEffects::Read>(e.getEffect())) {
            for (MemSlot *s : slotsFor(e.getValue())) {
                s->pendingReads.insert(op);
            }
        }
    }

    for (const auto &e : effects) {
        Value v = e.getValue();
        if (isa<MemoryEffects::Allocate>(e.getEffect())) {
            if (MemSlot *s = getOrCreateSlot(v)) {
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Write>(e.getEffect())) {
            for (MemSlot *s : slotsFor(v)) {
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Free>(e.getEffect())) {
            SmallPtrSet<MemSlot *, 4> toRemove;
            if (!v) {
                for (auto &slot : slots) {
                    toRemove.insert(slot.get());
                }
            } else {
                auto aliasSlots = findAliasSlots(v);
                for (auto *slot : aliasSlots) {
                    toRemove.insert(slot);
                }
            }
            llvm::erase_if(slots, [&](std::unique_ptr<MemSlot> &sp) {
                if (toRemove.contains(sp.get())) {
                    if (sp->memref) {
                        valueToSlot.erase(sp->memref);
                    }
                    return true;
                }
                return false;
            });
            cache.clear();
        }
    }
}

MemoryDependenceGraph::Snapshot MemoryDependenceGraph::takeSnapshot() const
{
    Snapshot snap;
    snap.states.reserve(slots.size());
    for (const auto &slot : slots) {
        Snapshot::SlotState s;
        s.memref = slot->memref;
        s.lastWriter = slot->lastWriter;
        s.pendingReads = slot->pendingReads;
        snap.states.push_back(std::move(s));
    }
    return snap;
}

void MemoryDependenceGraph::restoreSnapshot(Snapshot &&snap)
{
    slots.clear();
    valueToSlot.clear();
    slots.reserve(snap.states.size());
    for (auto &s : snap.states) {
        auto slot = std::make_unique<MemSlot>(s.memref);
        slot->lastWriter = s.lastWriter;
        slot->pendingReads = std::move(s.pendingReads);
        if (s.memref) {
            valueToSlot[s.memref] = slot.get();
        }
        slots.push_back(std::move(slot));
    }
}

void MemoryDependenceGraph::recordEdges(Operation *op, ArrayRef<Operation *> defs, ArrayRef<Operation *> preds)
{
    if (!defs.empty()) {
        auto &defList = memDefs[op];
        defList.assign(defs.begin(), defs.end());
        for (Operation *p : defs) {
            memUsers[p].push_back(op);
        }
    }

    if (!preds.empty()) {
        auto &execBeforeList = execBefore[op];
        execBeforeList.assign(preds.begin(), preds.end());
        for (Operation *p : preds) {
            execAfter[p].push_back(op);
        }
    }
}
