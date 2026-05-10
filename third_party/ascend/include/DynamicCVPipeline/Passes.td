/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_DYNAMIC_CV_PIPELINE_PASSES
#define TRITON_DYNAMIC_CV_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def AddDynamicCVPipeline : Pass<"add_dynamic_cv_pipeline", "mlir::ModuleOp"> {
    let summary = "Add dynamic cv pipeline based on ssbufer";
    let constructor = "triton::createAddDynamicCVPipelinePass()";
    let options = [
        Option<"compileOn91095", "compile-on-910-95",
        "bool", /*default*/"false",
        "compile on 910_95">
    ];
}

def AddBlockIdForControlOpsPass : Pass<"add-block-id-for-control-ops", "mlir::ModuleOp"> {
    let summary = "Add block_id attribute to control flow operations";
    let constructor = "triton::createAddBlockIdForControlOpsPass()";
}

def DataDependencyAnalysisPass : Pass<"data-dependency-analysis", "mlir::ModuleOp"> {
    let summary = "Analyze data dependencies between Vector and Cube blocks";
    let constructor = "triton::createDataDependencyAnalysisPass()";
}

def InterCoreTransferAndSyncPass : Pass<"inter-core-transfer-and-sync", "mlir::ModuleOp"> {
    let summary = "Insert inter-core transfer operations and PIPE_S synchronization between Vector and Cube cores";
    let constructor = "triton::createInterCoreTransferAndSyncPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def MarkMainLoopPass : Pass<"mark-main-loop", "mlir::ModuleOp"> {
    let summary = "Mark main computation loops containing FixpipeOp or CopyOp with main_loop attribute";
    let constructor = "triton::createMarkMainLoopPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}


#endif // TRITON_DYNAMIC_CV_PIPELINE_PASSES