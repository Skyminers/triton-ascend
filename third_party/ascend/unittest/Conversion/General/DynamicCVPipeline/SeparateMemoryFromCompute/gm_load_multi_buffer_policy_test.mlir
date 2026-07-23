// RUN: triton-opt --gm-load-multi-buffer-policy --split-input-file %s | FileCheck %s

// GMLoadMultiBufferPolicyPass: per-load GM-load multi-buffer depth policy.
//
// Depth is a binary {single, double} choice. Whether double is worth it is a
// RELATIVE urgency question (throughput is unknown, so no reliable block time):
// urgency = size / same-core-ancestor element count of the consumer (purely
// topological). The UB/L1 budget grants double from the most urgent down; a user
// hivm.multi_buffer hint is a pure output override.

// Case 1: Urgency ranking under a tight budget. Two same-pool loads: one feeds a
// source block V0 (no same-core predecessor -> maskWork=0 -> very urgent), one
// feeds V3 which sits behind a heavy vector chain V1->V2 (large maskWork -> low
// urgency). ub_budget=20% only affords one double -> the exposed load doubles,
// the masked one stays single.
// CHECK-LABEL: func.func @urgency_ranking
// masked load (block 3) is the low-urgency one -> single, no mark
// CHECK: memref.alloc() {ssbuffer.block_id = 3 {{.*}} memref<64x64xf32>
// CHECK-NOT: annotation.mark
// exposed load (block 5) is the high-urgency one -> double
// CHECK: %[[E:.*]] = memref.alloc() {ssbuffer.block_id = 5 {{.*}} memref<64x64xf32>
// CHECK-NEXT: annotation.mark %[[E]] {hivm.multi_buffer = 2 : i32}
module attributes {ssbuffer.ub_budget = 20 : i32} {
  func.func @urgency_ranking(%in: tensor<512x512xf32>, %seed: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant 256 : index
    scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %in) -> (tensor<512x512xf32>) {
      %v1 = linalg.add {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%acc, %acc : tensor<512x512xf32>, tensor<512x512xf32>) outs(%acc : tensor<512x512xf32>) -> tensor<512x512xf32>
      %v2 = linalg.add {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} ins(%v1, %v1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%v1 : tensor<512x512xf32>) -> tensor<512x512xf32>
      %lm_alloc = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<64x64xf32>
      %lm = bufferization.to_tensor %lm_alloc restrict writable {gm_load_bufferable} : memref<64x64xf32>
      %v3a = linalg.add {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} ins(%lm, %lm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%lm : tensor<64x64xf32>) -> tensor<64x64xf32>
      %v3b = linalg.add {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} ins(%v2, %v2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%v2 : tensor<512x512xf32>) -> tensor<512x512xf32>
      %le_alloc = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      %le = bufferization.to_tensor %le_alloc restrict writable {gm_load_bufferable} : memref<64x64xf32>
      %v0 = linalg.add {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%le, %seed : tensor<64x64xf32>, tensor<64x64xf32>) outs(%le : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %v3b : tensor<512x512xf32>
    } {ssbuffer.main_loop}
    return
  }
}

// -----

// Case 2: Ample budget -> both double. With ub_budget=100% the low-urgency
// masked load is not sacrificed; urgency only decides order, not eligibility.
// CHECK-LABEL: func.func @ample_budget
// CHECK: %[[M:.*]] = memref.alloc() {ssbuffer.block_id = 3 {{.*}} memref<64x64xf32>
// CHECK-NEXT: annotation.mark %[[M]] {hivm.multi_buffer = 2 : i32}
// CHECK: %[[E:.*]] = memref.alloc() {ssbuffer.block_id = 5 {{.*}} memref<64x64xf32>
// CHECK-NEXT: annotation.mark %[[E]] {hivm.multi_buffer = 2 : i32}
module attributes {ssbuffer.ub_budget = 100 : i32} {
  func.func @ample_budget(%in: tensor<512x512xf32>, %seed: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant 256 : index
    scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %in) -> (tensor<512x512xf32>) {
      %v1 = linalg.add {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%acc, %acc : tensor<512x512xf32>, tensor<512x512xf32>) outs(%acc : tensor<512x512xf32>) -> tensor<512x512xf32>
      %v2 = linalg.add {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} ins(%v1, %v1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%v1 : tensor<512x512xf32>) -> tensor<512x512xf32>
      %lm_alloc = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<64x64xf32>
      %lm = bufferization.to_tensor %lm_alloc restrict writable {gm_load_bufferable} : memref<64x64xf32>
      %v3a = linalg.add {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} ins(%lm, %lm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%lm : tensor<64x64xf32>) -> tensor<64x64xf32>
      %v3b = linalg.add {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} ins(%v2, %v2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%v2 : tensor<512x512xf32>) -> tensor<512x512xf32>
      %le_alloc = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<64x64xf32>
      %le = bufferization.to_tensor %le_alloc restrict writable {gm_load_bufferable} : memref<64x64xf32>
      %v0 = linalg.add {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%le, %seed : tensor<64x64xf32>, tensor<64x64xf32>) outs(%le : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %v3b : tensor<512x512xf32>
    } {ssbuffer.main_loop}
    return
  }
}

// -----

// Case 3: User pin honored. A load already carrying hivm.multi_buffer is kept
// verbatim, independent of the urgency/budget decision.
// CHECK-LABEL: func.func @user_pinned
// CHECK: memref.alloc() {ssbuffer.block_id = 3
// CHECK-NEXT: annotation.mark {{.*}} {hivm.multi_buffer = 3 : i32}
module attributes {ssbuffer.ub_budget = 100 : i32} {
  func.func @user_pinned(%vin: tensor<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant 256 : index
    scf.for %i = %c0 to %cN step %c1 {
      %l_alloc = memref.alloc() {ssbuffer.block_id = 3 : i32} : memref<64x64xf32>
      annotation.mark %l_alloc {hivm.multi_buffer = 3 : i32} : memref<64x64xf32>
      %l = bufferization.to_tensor %l_alloc restrict writable {gm_load_bufferable} : memref<64x64xf32>
      %v = linalg.add {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} ins(%l, %vin : tensor<64x64xf32>, tensor<64x64xf32>) outs(%vin : tensor<64x64xf32>) -> tensor<64x64xf32>
    } {ssbuffer.main_loop}
    return
  }
}
