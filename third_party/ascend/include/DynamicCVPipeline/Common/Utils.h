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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_UTILS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_UTILS_H

#include <cassert>
#include <string_view>
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "ascend/include/AddAutoScheduling/Common/MemoryEffectsTracker.h"

namespace mlir {
namespace CVSplit {

// ComputeBlockIdManager for managing compute block IDs
class ComputeBlockIdManager {
    // Singleton class to manage compute block IDs
public:
    static ComputeBlockIdManager &getInstance()
    {
        static ComputeBlockIdManager instance;
        return instance;
    }

    unsigned int getNextId() { return cntComputeBlockId++; }

    llvm::SmallVector<Operation *> getOpsByBlockId(int blockId);
    int getBlockIdByOp(Operation *op);
    void markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops);

private:
    friend void markOpBlockId(Operation *op, int blockId);
    ComputeBlockIdManager() : cntComputeBlockId(0) {}
    void record(Operation *op, int blockId);

    unsigned int cntComputeBlockId;
    llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToOps;
    llvm::DenseMap<Operation *, int> opToBlockId;
};

// Functions for managing block IDs
int lookupOpBlockId(Operation *op);
void markOpBlockId(Operation *op, int blockId);

// Attribute names for block ID management
namespace attr {
inline constexpr llvm::StringLiteral kBlockId = "ssbuffer.block_id";
inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";
} // namespace attr

// Core-type enums
enum class OpAbility {
    PREFER_VECTOR = 1 << 0,
    CUBE_ONLY = 1 << 1,
    CUBE_AND_VECTOR = (1 << 0) | (1 << 1),
};

enum CoreType {
    UNDETERMINED = 0,
    VECTOR_ONLY = 1 << 0,
    CUBE_ONLY = 1 << 1,
    CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr std::string_view toStrCoreType(CoreType a)
{
    switch (a) {
        case CoreType::VECTOR_ONLY:
            return "VECTOR";
        case CoreType::CUBE_ONLY:
            return "CUBE";
        case CoreType::CUBE_AND_VECTOR:
            return "CUBE_AND_VECTOR";
        case CoreType::UNDETERMINED:
        default:
            return "undetermined";
    }
}
inline constexpr CoreType fromStrCoreType(std::string_view s)
{
    if (s == "VECTOR")
        return CoreType::VECTOR_ONLY;
    if (s == "CUBE")
        return CoreType::CUBE_ONLY;

    return CoreType::UNDETERMINED;
}
inline constexpr CoreType toCoreType(OpAbility a)
{
    return static_cast<CoreType>(static_cast<int>(a));
}
inline constexpr CoreType operator|(CoreType l, CoreType r)
{
    return static_cast<CoreType>(static_cast<int>(l) | static_cast<int>(r));
}
inline constexpr CoreType operator&(CoreType l, CoreType r)
{
    return static_cast<CoreType>(static_cast<int>(l) & static_cast<int>(r));
}
inline bool intersects(CoreType l, CoreType r)
{
    return (l & r) != UNDETERMINED;
}
inline bool exactlyOneType(CoreType ct)
{
    return ct == CUBE_ONLY || ct == VECTOR_ONLY;
}
/// Return true when data must cross a Cube↔Vector boundary.
inline bool needsSync(CoreType src, CoreType dst)
{
    return (src == CUBE_ONLY && dst == VECTOR_ONLY) || (src == VECTOR_ONLY && dst == CUBE_ONLY);
}

// Functions for managing core types
CoreType lookupOpCoreType(Operation *op);
void markOpCoreType(Operation *op, CoreType ct);
const char *literalCoreType(CoreType ct);

Operation *getAncestorInBlock(Operation *inner, Block *block);
bool isSameBlock(Operation *a, Operation *b);
bool isWholeCubeReady(Operation *seedOp, DenseMap<Operation *, int> &indegree);
void initializeIndegreeForBlock(Block *block, DenseMap<Operation *, int> &indegree,
                                const MemoryDependenceGraph &memGraph);
} // namespace CVSplit
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_UTILS_H
