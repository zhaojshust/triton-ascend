// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True' | FileCheck %s

// CHECK-LABEL: tt.func @triton_indirect_load
// CHECK: ascend.indirect_load {{.*}} -> tensor<1024xi32>
tt.func @triton_indirect_load(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5, %3, %cst {is_discrete_mask} : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @triton_indirect_store
// CHECK: ascend.indirect_store {{.*}}
tt.func @triton_indirect_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5 : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8, %3 {is_discrete_mask} : tensor<1024x!tt.ptr<i32>>
  tt.return
}