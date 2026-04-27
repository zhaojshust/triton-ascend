add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  PlanComputeBlock.cpp
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
)