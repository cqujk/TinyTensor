#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
    // Step 1: Remove redundant transpose pairs
    auto it = ops.begin();
    while (it != ops.end()) {
        if (it + 1 != ops.end() && 
            (*it)->getOpType() == OpType::Transpose && 
            (*(it+1))->getOpType() == OpType::Transpose) {
            // auto &t1 = *it;
            // auto &t2 = *(it+1);
            // 动态转换为 TransposeObj 以访问转置轴属性
            auto t1 = std::dynamic_pointer_cast<TransposeObj>(*it);
            auto t2 = std::dynamic_pointer_cast<TransposeObj>(*(it + 1));
            // Check if transpose axes are the same (reverse order)
            if ( t1->getOutputs()[0] == t2->getInputs()[0]) {
                // 检查轴顺序是否互为逆操作
                auto permute1 = t1->getPermute();
                auto permute2 = t2->getPermute();
                bool isInverse = true;
                for (size_t i = 0; i < permute1.size(); ++i) {
                    if (permute1[permute2[i]] != (int)i) {
                        isInverse = false;
                        break;
                    }
                }
                if (isInverse) {
                    // 删除两个转置算子，直接连接 t1 的输入到 t2 的输出
                    auto inputTensor = t1->getInputs()[0];
                    auto outputTensor = t2->getOutputs()[0];
                    outputTensor->setSource(inputTensor->getSource());
                    removeOperator(t2);
                    it = ops.erase(it + 1);
                    removeOperator(t1);
                    it = ops.erase(it);
                    continue;
                }
            }
        }
        ++it;
    }

   // Step 2: 合并 Transpose 到 Matmul
   for (auto &op : ops) {
    // 1. 检查当前算子是否为 Matmul
    if (op->getOpType() != OpType::MatMul)
        continue;

    // 2. 动态转换为 MatmulObj 以访问 transA/transB 属性
    auto matmul = std::dynamic_pointer_cast<MatmulObj>(op);
    if (!matmul)
        continue;

    // 3. 处理输入 A 的前置 Transpose
    auto inputA = matmul->getInputs()[0];
    if (inputA->getSource() && 
        inputA->getSource()->getOpType() == OpType::Transpose) {
        auto transposeA = std::dynamic_pointer_cast<TransposeObj>(inputA->getSource());
        if (transposeA) {
            // 检查 Transpose 是否仅交换最后两个维度
            auto permute = transposeA->getPermute();
            bool isLastTwoSwapped = true;
            for (size_t i = 0; i < permute.size() - 2; ++i) {
                if (permute[i] != (int)i) {
                    isLastTwoSwapped = false;
                    break;
                }
            }
            if (isLastTwoSwapped && 
                permute[permute.size() - 2] == (int)(permute.size() - 1) &&
                permute[permute.size() - 1] == (int)(permute.size() - 2)) {
                // 合并到 Matmul 的 transA 属性
                matmul->setTransA(!matmul->getTransA());
                // 将 Matmul 的输入直接连接到 Transpose 的输入
                matmul->replaceInput(inputA, transposeA->getInputs()[0]);
                removeOperator(transposeA);
            }
        }
    }

    // 4. 处理输入 B 的前置 Transpose（逻辑同上）
    auto inputB = matmul->getInputs()[1];
    if (inputB->getSource() && 
        inputB->getSource()->getOpType() == OpType::Transpose) {
        auto transposeB = std::dynamic_pointer_cast<TransposeObj>(inputB->getSource());
        if (transposeB) {
            auto permute = transposeB->getPermute();
            bool isLastTwoSwapped = true;
            for (size_t i = 0; i < permute.size() - 2; ++i) {
                if (permute[i] != (int)i) {
                    isLastTwoSwapped = false;
                    break;
                }
            }
            if (isLastTwoSwapped && 
                permute[permute.size() - 2] == (int)(permute.size() - 1) &&
                permute[permute.size() - 1] == (int)(permute.size() - 2)) {
                matmul->setTransB(!matmul->getTransB());
                matmul->replaceInput(inputB, transposeB->getInputs()[0]);
                removeOperator(transposeB);
            }
        }
    }
    }
    }
    // void GraphObj::removeOperator(OperatorObj &op) {
    //     // 断开输入和输出的链接
    //     for (auto input : op.getInputs()) {
    //         input->removeTarget(op);
    //         if (auto source = input->getSource()) {
    //             source->removeSuccessors(op);
    //             op.removePredecessors(source);
    //         }
    //     }
    //     for (auto output : op.getOutputs()) {
    //         output->setSource(nullptr);
    //         for (auto successor = output->getTargets()) {
    //             successor->removePredecessors(op);
    //             op.removeSuccessors(successor);
    //         }
    //     }
        
    //     // 从操作符列表中移除
    //     auto it = std::find(ops.begin(), ops.end(), op);
    //     if (it != ops.end()) {
    //         ops.erase(it);
    //     }
    // }
    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================


         // 分配内存偏移量并记录
        std::unordered_map<Tensor, size_t> tensorOffsets;
        for (const auto& tensor : tensors) {
            size_t size = tensor->getBytes();
            size_t offset = allocator.alloc(size);
            tensorOffsets[tensor] = offset;
        }
        // 实际分配内存
        void* basePtr = allocator.getPtr();
           // 绑定内存到各个张量
        for (const auto& tensor : tensors) {
            size_t offset = tensorOffsets[tensor];
            void* dataPtr = static_cast<char*>(basePtr) + offset;
            // 创建Blob并设置到张量中
            tensor->setDataBlob(make_ref<BlobObj>(runtime, dataPtr));
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini