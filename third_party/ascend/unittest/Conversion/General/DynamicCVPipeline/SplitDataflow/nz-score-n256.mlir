// RUN: triton-opt %s --data-dependency-analysis --inter-core-transfer-and-sync | FileCheck %s
// N256 must keep the existing independent NZ16 transfer and V-pipe release.
// CHECK-NOT: memref<32x8x16x16xf32
// CHECK: memref.alloc() {{.*}} : memref<16x8x16x16xf32, #hivm.address_space<ub>>
// CHECK: sync_block_set {{.*}}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = [[FLAG:[0-9]+]]
// CHECK: sync_block_wait {{.*}}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = [[FLAG]]
// CHECK-NEXT: hivm.hir.fixpipe
// CHECK-NEXT: hivm.hir.sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = [[FLAG]]
// CHECK: bufferization.to_tensor {{.*}} to tensor<16x8x16x16xf32>
// CHECK: "__builtin_online_softmax_nz"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @n256(%q: tensor<128x128xf8E4M3FN>, %k: tensor<128x256xf8E4M3FN>, %out: memref<8x8x16x32xf8E4M3FN>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %lut = arith.constant dense<0> : tensor<256xi8>
    %state = arith.constant dense<0.0> : tensor<128xf32>
    scf.for %i = %c0 to %n step %c1 {
      %e = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x256xf32>
      %z = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%zero : f32) outs(%e : tensor<128x256xf32>) -> tensor<128x256xf32>
      %qk = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %k : tensor<128x128xf8E4M3FN>, tensor<128x256xf8E4M3FN>) outs(%z : tensor<128x256xf32>) -> tensor<128x256xf32>
      %nz = hivm.hir.convert_layout %qk output_shape [16, 8, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<128x256xf32>) -> tensor<16x8x16x16xf32>
      %pe = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8x8x16x32xf8E4M3FN>
      %se = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
      %p:3 = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} "__builtin_online_softmax_nz" ins(%nz, %state, %lut : tensor<16x8x16x16xf32>, tensor<128xf32>, tensor<256xi8>) outs(%pe, %se, %se : tensor<8x8x16x32xf8E4M3FN>, tensor<128xf32>, tensor<128xf32>) -> (tensor<8x8x16x32xf8E4M3FN>, tensor<128xf32>, tensor<128xf32>)
      bufferization.materialize_in_destination %p#0 in writable %out {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<8x8x16x32xf8E4M3FN>, memref<8x8x16x32xf8E4M3FN>) -> ()
    } {ssbuffer.block_id = 0 : i32}
    return
  }
}
