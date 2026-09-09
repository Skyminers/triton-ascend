// RUN: triton-opt %s --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=BATCH
// RUN: sed 's@^    // WRITE_METADATA$@memref.store %%zero_i32, %%metadata[%%c0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>@' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=CLOBBER
// RUN: sed '/%%meta1 = memref.load/s/{ssbuffer.block_id/{volatile, ssbuffer.block_id/' %s | triton-opt --materialize-cube-page-loaders --verify-each | FileCheck %s --check-prefix=VOLATILE

// Four independent descriptors precede their DMA users. The next window
// starts after those four DMAs, bounding live scalar results.
// BATCH-LABEL: func.func @scheduled_pages
// BATCH: scf.for
// BATCH: memref.load
// BATCH-NOT: hivm.hir.nd2nz
// BATCH: memref.load
// BATCH-NOT: hivm.hir.nd2nz
// BATCH: memref.load
// BATCH-NOT: hivm.hir.nd2nz
// BATCH: memref.load
// BATCH-NOT: memref.load
// BATCH-COUNT-4: hivm.hir.nd2nz
// BATCH: memref.load
// BATCH-NOT: hivm.hir.nd2nz
// BATCH: memref.load
// BATCH-COUNT-2: hivm.hir.nd2nz
// BATCH: scf.yield

// A possibly aliased store prevents reads crossing the write.
// CLOBBER-LABEL: func.func @scheduled_pages
// CLOBBER: memref.load
// CLOBBER: hivm.hir.nd2nz
// CLOBBER: memref.store
// CLOBBER: memref.load
// CLOBBER: hivm.hir.nd2nz

// Volatile reads remain at their original position relative to page DMAs.
// VOLATILE-LABEL: func.func @scheduled_pages
// VOLATILE: hivm.hir.nd2nz
// VOLATILE: memref.load {{.*}}volatile
// VOLATILE: hivm.hir.nd2nz

func.func @scheduled_pages(%metadata: memref<?xi32>, %cache: memref<?xf16>, %q: tensor<16x96xf16>, %initial: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %zero_i32 = arith.constant 0 : i32
  %result = scf.for %iv = %c0 to %c6 step %c1 iter_args(%acc = %initial) -> tensor<16x16xf32> {
    %empty = tensor.empty() : tensor<96x16xf16>
    %c0_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %slot0 = arith.addi %iv, %c0_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta0 = memref.load %metadata[%slot0] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset0 = arith.index_cast %meta0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src0 = memref.reinterpret_cast %cache to offset: [%offset0], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page0 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src0, %page0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor0 = bufferization.to_tensor %page0 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert0 = tensor.insert_slice %tensor0 into %empty[0, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    // WRITE_METADATA
    %c1_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 1 : index
    %slot1 = arith.addi %iv, %c1_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta1 = memref.load %metadata[%slot1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset1 = arith.index_cast %meta1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src1 = memref.reinterpret_cast %cache to offset: [%offset1], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src1, %page1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor1 = bufferization.to_tensor %page1 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert1 = tensor.insert_slice %tensor1 into %insert0[16, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    %c2_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 2 : index
    %slot2 = arith.addi %iv, %c2_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta2 = memref.load %metadata[%slot2] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset2 = arith.index_cast %meta2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src2 = memref.reinterpret_cast %cache to offset: [%offset2], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page2 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src2, %page2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor2 = bufferization.to_tensor %page2 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert2 = tensor.insert_slice %tensor2 into %insert1[32, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    %c3_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 3 : index
    %slot3 = arith.addi %iv, %c3_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta3 = memref.load %metadata[%slot3] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset3 = arith.index_cast %meta3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src3 = memref.reinterpret_cast %cache to offset: [%offset3], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page3 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src3, %page3 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor3 = bufferization.to_tensor %page3 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert3 = tensor.insert_slice %tensor3 into %insert2[48, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    %c4_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 4 : index
    %slot4 = arith.addi %iv, %c4_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta4 = memref.load %metadata[%slot4] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset4 = arith.index_cast %meta4 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src4 = memref.reinterpret_cast %cache to offset: [%offset4], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page4 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src4, %page4 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor4 = bufferization.to_tensor %page4 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert4 = tensor.insert_slice %tensor4 into %insert3[64, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    %c5_page = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 5 : index
    %slot5 = arith.addi %iv, %c5_page {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : index
    %meta5 = memref.load %metadata[%slot5] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xi32>
    %offset5 = arith.index_cast %meta5 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %src5 = memref.reinterpret_cast %cache to offset: [%offset5], sizes: [16, 16], strides: [16, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
    %page5 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16>
    memref.copy %src5, %page5 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16, strided<[16, 1], offset: ?>> to memref<16x16xf16>
    %tensor5 = bufferization.to_tensor %page5 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<16x16xf16> to tensor<16x16xf16>
    %insert5 = tensor.insert_slice %tensor5 into %insert4[80, 0] [16, 16] [1, 1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf16> into tensor<96x16xf16>
    %mm = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%q, %insert5 : tensor<16x96xf16>, tensor<96x16xf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.yield %mm : tensor<16x16xf32>
  }
  return %result : tensor<16x16xf32>
}
