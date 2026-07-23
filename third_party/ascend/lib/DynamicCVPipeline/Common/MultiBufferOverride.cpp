/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/MultiBufferOverride.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

using namespace mlir;

namespace gmload {

annotation::MarkOp findAllocMark(memref::AllocOp alloc)
{
    for (Operation *user : alloc.getResult().getUsers())
        if (auto mark = dyn_cast<annotation::MarkOp>(user))
            return mark;
    return nullptr;
}

std::optional<int> getMultiBufferOverride(memref::AllocOp alloc)
{
    if (!alloc)
        return std::nullopt;
    if (annotation::MarkOp mark = findAllocMark(alloc))
        if (mark.isAnnotatedByStaticAttr(CVPipeline::kMultiBuffer))
            if (auto attr = dyn_cast_or_null<IntegerAttr>(mark.getStaticAttrValue(CVPipeline::kMultiBuffer)))
                return static_cast<int>(attr.getInt());
    return std::nullopt;
}

} // namespace gmload
