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
            it->getType() == OpCode::Transpose && 
            std::next(it)->getType() == OpCode::Transpose) {
            
            auto &t1 = *it;
            auto &t2 = *(it+1);
            
            // Check if transpose axes are the same (reverse order)
            if (t1.getAxes() == t2.getAxes()) {
                // Remove t2 first to avoid iterator invalidation
                removeOperator(t2);
                it = ops.erase(it); // erase t1 now
                continue;
            }
        }
        ++it;
    }

    // Step 2: Merge transpose into matmul
    // 遍历所有 Matmul 节点，尝试合并前置的 Transpose
    for (auto &matmul : ops) {
        if (matmul.getType() != OpCode::MatMul) continue;
        
        // Check input A and B for possible preceding Transposes
        auto *inputA = matmul.getInput(0);
        auto *inputB = matmul.getInput(1);
        
        // Check if inputA is a Transpose, and its axis affects last two dimensions
        if (inputA && inputA->getType() == OpCode::Transpose) {
            auto &tA = *inputA;
            if (tA.getAxes().empty() || tA.getAxes().size() == 1) {
                // Transpose on the last dimension (e.g., [::-1])
                matmul.setTransA(true);
                // Disconnect and remove the transpose node
                removeOperator(*inputA);
            }
        }
        
        // Similarly for inputB
        if (inputB && inputB->getType() == OpCode::Transpose) {
            auto &tB = *inputB;
            if (tB.getAxes().empty() || tB.getAxes().size() == 1) {
                matmul.setTransB(true);
                removeOperator(*inputB);
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