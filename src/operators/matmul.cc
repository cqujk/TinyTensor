#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        IT_ASSERT(inputs.size() == 2);
        const auto &A = inputs[0];
        const auto &B = inputs[1];
        const auto &shapeA = A->getDims();
        const auto &shapeB = B->getDims();
        // 确保输入至少有两个维度
        IT_ASSERT(shapeA.size() >= 2 && shapeB.size() >= 2);
        // 处理转置后的最后两个维度
        int a_rows, a_cols;
        if (transA) {
            a_rows = shapeA[shapeA.size() - 1];
            a_cols = shapeA[shapeA.size() - 2];
        } else {
            a_rows = shapeA[shapeA.size() - 2];
            a_cols = shapeA[shapeA.size() - 1];
        }
        int b_rows, b_cols;
        if (transB) {
            b_rows = shapeB[shapeB.size() - 1];
            b_cols = shapeB[shapeB.size() - 2];
        } else {
            b_rows = shapeB[shapeB.size() - 2];
            b_cols = shapeB[shapeB.size() - 1];
        }
        // 验证矩阵乘法维度匹配
        if(a_cols != b_rows){
            printf("[ERROR] Matmul: matrix dimension mismatch,a shape: rows, %d,cols,%d, b shape:rows,%d,cols,%d \n",a_rows, a_cols, b_rows,b_cols);
            return std::nullopt;
        }
        // 提取前导维度（batch部分）
        Shape batchA, batchB;
        if (shapeA.size() > 2) {
            batchA = Shape(shapeA.begin(), shapeA.end() - 2);
        }
        if (shapeB.size() > 2) {
            batchB = Shape(shapeB.begin(), shapeB.end() - 2);
        }

        // 广播前导维度
        Shape batch_shape = infer_broadcast(batchA, batchB);

        // 组合最终输出形状
        Shape output_shape = batch_shape;
        output_shape.insert(output_shape.end(), {a_rows, b_cols});
        }

} // namespace infini