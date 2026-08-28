// RUN: triton-opt --mark-main-loop --mark-main-loop %s | FileCheck %s --implicit-check-not="tt.compile_hint"
// RUN: triton-opt --mark-main-loop --add-block-id-for-control-ops %s | FileCheck %s --check-prefix=BOUNDARY --implicit-check-not="tt.compile_hint"

// An explicit frontend hint must select the outer loop even though the
// Fixpipe/Copy heuristic would otherwise select the inner loop.
// CHECK-LABEL: func.func @explicit_outer_main_loop
// CHECK: scf.for
// CHECK: } {test.loop = "outside"}
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:   } {test.loop = "candidate"}
// CHECK: } {ssbuffer.main_loop = 0 : i32, test.loop = "selected"}
// CHECK-NOT: ssbuffer.main_loop

// Only control flow contained by the selected loop belongs to the pipeline.
// BOUNDARY-LABEL: func.func @explicit_outer_main_loop
// BOUNDARY: scf.for
// BOUNDARY: } {test.loop = "outside"}
// BOUNDARY: scf.for
// BOUNDARY:   scf.for
// BOUNDARY:   } {ssbuffer.block_id = 0 : i32, test.loop = "candidate"}
// BOUNDARY: } {ssbuffer.block_id = 1 : i32, ssbuffer.main_loop = 0 : i32, test.loop = "selected"}
func.func @explicit_outer_main_loop(%src: tensor<4xf32>, %dst: memref<4xf32>,
                                    %lb: index, %ub: index, %step: index) {
  scf.for %outside = %lb to %ub step %step {
    hivm.hir.copy ins(%src : tensor<4xf32>) outs(%dst : memref<4xf32>)
  } {test.loop = "outside"}
  scf.for %outer = %lb to %ub step %step {
    scf.for %inner = %lb to %ub step %step {
      hivm.hir.copy ins(%src : tensor<4xf32>) outs(%dst : memref<4xf32>)
    } {test.loop = "candidate"}
  } {test.loop = "selected", tt.compile_hint = "main_loop"}
  return
}
