// RUN: triton-opt %S/Inputs/paired-f16-recurrence.mlir --paired-f16-pv-accumulate -o %t.pv
// RUN: triton-opt %t.pv --paired-f16-acc-ownership --verify-each -o %t.owned
// RUN: FileCheck %s < %t.owned
// RUN: sed 's/arith.addf %95, %94/arith.addf %95, %95/' %S/Inputs/paired-f16-recurrence.mlir | not triton-opt --paired-f16-pv-accumulate --paired-f16-acc-ownership 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/arith.extf %arg18/arith.extf %78#1/' %S/Inputs/paired-f16-recurrence.mlir | not triton-opt --paired-f16-pv-accumulate --paired-f16-acc-ownership 2>&1 | FileCheck %s --check-prefix=REJECT

// CHECK-LABEL: func.func @_attn_fwd_paired_f16
// CHECK-SAME: ascend.loop_carried_read_before_update = 0 : i64
// CHECK: %[[OLD_MAX:.*]] = arith.extf %arg18
// CHECK: hivm.hir.custom
// CHECK: arith.mulf %arg20, {{.*}} {ascend.destination_operand = 0 : i64, ascend.destination_opposite_if_forward}
// CHECK: arith.addf {{.*}} {ascend.destination_operand = 0 : i64}
// CHECK: arith.divf %{{.*}}, {{.*}} {ascend.destination_operand = 0 : i64}
// REJECT: error: paired FP16 accumulator ownership requires an exclusive loop-carried acc recurrence and terminal normalize
