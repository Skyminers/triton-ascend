// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s --implicit-check-not="triton_ascend.dynamic_cv_pipeline.rc"

// A nested scf.if must not become the MultiCache boundary when an outer loop
// is explicitly selected. The alloc and its nested-region user both belong to
// the unmarked inner loop's block at the selected main-loop boundary, so this
// is not an unsupported cross-block memref dependency.
// CHECK-LABEL: func.func @inner_explicit_outer
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     %[[ALLOC:.*]] = memref.alloc()
// CHECK:     scf.if
// CHECK:       linalg.fill
// CHECK-SAME:  outs(%[[ALLOC]]
// CHECK:   } {ssbuffer.block_id = 10 : i32}
// CHECK: } {ssbuffer.block_id = 11 : i32, ssbuffer.main_loop = 0 : i32}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @inner_explicit_outer() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %zero = arith.constant 0.0 : f32
    scope.scope : () -> () {
      scf.for %outer = %c0 to %c4 step %c1 {
        scf.for %inner = %c0 to %c4 step %c1 {
          %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<4xf32>
          scf.if %true {
            linalg.fill {ssbuffer.block_id = 9 : i32} ins(%zero : f32) outs(%alloc : memref<4xf32>)
          } {ssbuffer.block_id = 9 : i32}
          %tensor = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 9 : i32} : memref<4xf32> to tensor<4xf32>
          %result = arith.addf %tensor, %tensor {ssbuffer.block_id = 9 : i32} : tensor<4xf32>
        } {ssbuffer.block_id = 10 : i32}
      } {ssbuffer.block_id = 11 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
