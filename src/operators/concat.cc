#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
   // 计算指定维度的总和
   size_t sum_dim = dims[dim];
    // 遍历后续输入张量
    for (size_t i = 1; i < inputs.size(); ++i) {
        const auto &current_dims = inputs[i]->getDims();
        IT_ASSERT(current_dims.size() == rank); // 维度数必须一致

        // 校验非拼接维度的一致性
        for (size_t j = 0; j < rank; ++j) {
            if (j !=  static_cast<size_t>(dim)) {
                IT_ASSERT(current_dims[j] == dims[j]);
            }
        }
        
        // 累加拼接维度的尺寸
        sum_dim += current_dims[dim];
    }

    // 构造输出形状
    dims[dim] = sum_dim;
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
