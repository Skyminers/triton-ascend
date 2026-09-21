// RUN: triton-opt %s --paired-f16-pv-accumulate -o %t.once
// RUN: FileCheck %s --check-prefix=FULL < %t.once
// RUN: triton-opt %t.once --paired-f16-pv-accumulate -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: sed 's/ attributes {ascend.paired_f16_pv_accumulate}//' %s > %t.off
// RUN: triton-opt %t.off -o %t.off.base
// RUN: triton-opt %t.off --paired-f16-pv-accumulate -o %t.off.pass
// RUN: diff %t.off.base %t.off.pass
// RUN: sed 's/ascend.paired_f16_pv_accumulate}/ascend.paired_f16_pv_accumulate = false}/' %s | not triton-opt --paired-f16-pv-accumulate 2>&1 | FileCheck %s --check-prefix=ATTR
// RUN: triton-opt %s --paired-f16-pv-accumulate --paired-f16-acc-ownership -o %t.k1
// RUN: FileCheck %s --check-prefix=K1 < %t.k1

// FULL-LABEL: func.func @full_pv
// FULL: hivm.hir.custom
// FULL-NOT: tensor.extract_slice
// FULL: hivm.hir.convert_layout {{.*}} output_shape [128, 512]
// FULL-COUNT-1: linalg.matmul
// FULL: tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>
// ATTR: error: ascend.paired_f16_pv_accumulate must be a unit attribute
// K1-LABEL: func.func @full_pv
// K1-NOT: ascend.paired_f16_pv_accumulate
// K1-NOT: ascend.destination_operand
// K1-COUNT-1: linalg.matmul

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @full_pv(%score: tensor<32x8x16x16xf16>, %m: tensor<128xf16>, %lut: tensor<256xi8>, %v: tensor<512x128xf8E4M3FN>, %out: memref<128x128xf32>, %state: memref<128xf16>, %sum: memref<128xf32>) attributes {ascend.paired_f16_pv_accumulate} {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 3 : i32
    %step = arith.constant 1 : i32
    %z = arith.constant 0.0 : f32
    %pe = tensor.empty() : tensor<16x8x16x32xf8E4M3FN>
    %me = tensor.empty() : tensor<128xf16>
    %se = tensor.empty() : tensor<128xf32>
    %ce = tensor.empty() : tensor<128x128xf32>
    %init = linalg.fill ins(%z : f32) outs(%ce : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.for %i = %lb to %ub step %step : i32 {
      %p:3 = hivm.hir.custom {arg_attrs = [{}, {}, {}, {}], gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>} "__builtin_online_softmax_nz" ins(%score, %m, %lut : tensor<32x8x16x16xf16>, tensor<128xf16>, tensor<256xi8>) outs(%pe, %me, %se : tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>) -> (tensor<16x8x16x32xf8E4M3FN>, tensor<128xf16>, tensor<128xf32>)
      %nd = hivm.hir.convert_layout %p#0 output_shape [128, 512] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<16x8x16x32xf8E4M3FN>) -> tensor<128x512xf8E4M3FN>
      bufferization.materialize_in_destination %p#1 in writable %state : (tensor<128xf16>, memref<128xf16>) -> ()
      bufferization.materialize_in_destination %p#2 in writable %sum : (tensor<128xf32>, memref<128xf32>) -> ()
      %pv = linalg.matmul ins(%nd, %v : tensor<128x512xf8E4M3FN>, tensor<512x128xf8E4M3FN>) outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
      bufferization.materialize_in_destination %pv in writable %out : (tensor<128x128xf32>, memref<128x128xf32>) -> ()
    }
    return
  }
}
