// RUN: triton-opt --mark-main-loop --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="tt.compile_hint" --implicit-check-not="ssbuffer.transfer_id = 1"

// The frontend-selected outer loop is the sole transfer boundary. A nested
// loop participates in that pipeline, while the unselected sibling loop does
// not create a second transfer group or become another main loop.
// CHECK-LABEL: func.func @explicit_outer_transfer
// CHECK-COUNT-2: memref.alloc()
// CHECK: hivm.hir.sync_block_set
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     hivm.hir.sync_block_wait
// CHECK:     hivm.hir.copy
// CHECK:     hivm.hir.sync_block_set
// CHECK:     hivm.hir.sync_block_wait
// CHECK:     hivm.hir.convert_layout
// CHECK:     hivm.hir.sync_block_set
// CHECK:   } {ssbuffer.block_id = 4 : i32, test.loop = "nested"}
// CHECK: } {ssbuffer.block_id = 5 : i32, ssbuffer.main_loop = 0 : i32, test.loop = "selected"}
// CHECK: hivm.hir.sync_block_wait
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:   } {test.loop = "unselected_nested"}
// CHECK: } {test.loop = "unselected"}
// CHECK-NOT: ssbuffer.main_loop

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @explicit_outer_transfer(%lb: index, %ub: index, %step: index,
                                     %b: tensor<16x16xf16>) {
    scf.for %outer = %lb to %ub step %step {
      scf.for %inner = %lb to %ub step %step {
        %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f16
        %empty0 = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf16>
        %fill0 = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%c0 : f16) outs(%empty0 : tensor<16x16xf16>) -> tensor<16x16xf16>
        %vec0 = math.exp %fill0 {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf16>
        %out0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
        %mm0 = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%vec0, %b : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out0 : tensor<16x16xf32>) -> tensor<16x16xf32>
      } {test.loop = "nested"}
    } {test.loop = "selected", tt.compile_hint = "main_loop"}

    scf.for %unselected = %lb to %ub step %step {
      scf.for %unselected_inner = %lb to %ub step %step {
        %c1 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f16
        %empty1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf16>
        %fill1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} ins(%c1 : f16) outs(%empty1 : tensor<16x16xf16>) -> tensor<16x16xf16>
        %vec1 = math.exp %fill1 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf16>
        %out1 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
        %mm1 = linalg.matmul {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%vec1, %b : tensor<16x16xf16>, tensor<16x16xf16>) outs(%out1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      } {test.loop = "unselected_nested"}
    } {test.loop = "unselected"}
    return
  }
}
