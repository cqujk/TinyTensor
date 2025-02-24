#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    size_t a_rank=A.size();
    size_t b_rank = B.size();
    size_t max_rank = std::max(a_rank, b_rank);
    // 创建补1后的形状容器
    Shape A_padded = A;
    Shape B_padded = B;
    
    // 在形状前面补1，直到维度数一致（ONNX规则）
    while (A_padded.size() < max_rank)
        A_padded.insert(A_padded.begin(), 1);
    while (B_padded.size() < max_rank)
        B_padded.insert(B_padded.begin(), 1);

    Shape result;
    for (size_t i = 0; i < max_rank; ++i) {
        const int dimA = A_padded[i];
        const int dimB = B_padded[i];
        
        // 检查维度兼容性（ONNX广播规则）
        if (dimA == dimB) {
            result.push_back(dimA);
        } else if (dimA == 1) {
            result.push_back(dimB);
        } else if (dimB == 1) {
            result.push_back(dimA);
        } else {
            // 无法广播时触发断言
            printf("Broadcast failed at dimension %zu: A=%d B=%d\n", i, dimA, dimB);
        }
    }
    return result;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
