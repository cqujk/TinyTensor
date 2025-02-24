#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;
        heapEnd = 0;
        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 第一阶段：在空闲块中查找合适内存
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            auto [addr, blockSize] = *it;

            if (blockSize >= size) {
                // 找到足够大的块，进行分割
                freeBlocks.erase(it);  // 移除当前块

                // 如果有剩余空间，将剩余部分重新插入
                if (blockSize > size) {
                    freeBlocks.emplace(addr + size, blockSize - size);
                }

                used += size;
                peak = std::max(peak, used);
                return addr;
            }
        }
        printf("when allocate memory,there is an action requesting new memory\n");
        // 第二阶段：没有可用空闲块，从堆末端分配
        const size_t allocatedAddr = heapEnd;
        heapEnd += size;  // 移动堆末端指针
        used += size;
        peak = std::max(peak, used);
        return allocatedAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        used -= size; // 更新内存使用统计
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if(addr + size ==heapEnd){
            heapEnd = addr;
        }else{
            // 插入新空闲块并获取迭代器
            auto [newIt, success] = freeBlocks.emplace(addr, size);
            auto currentIt = newIt;
            // 前向合并：检查与前一块的连续性
            if (currentIt != freeBlocks.begin()) {
                auto prevIt = std::prev(currentIt);
                if (prevIt->first + prevIt->second == addr) {
                    // 合并到前一块
                    prevIt->second += size;
                    freeBlocks.erase(currentIt);
                    currentIt = prevIt;  // 更新当前迭代器
                    addr = prevIt->first; // 更新合并后的地址
                    size = prevIt->second; // 更新合并后的大小
                }
            }
            // 后向合并：检查与后一块的连续性
            auto nextIt = std::next(currentIt);
            if (nextIt != freeBlocks.end() && 
                addr + size == nextIt->first) {
                currentIt->second += nextIt->second;
                freeBlocks.erase(nextIt);
            }
        }
    }
    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
