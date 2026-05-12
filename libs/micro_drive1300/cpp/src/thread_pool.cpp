/**
 * ThreadPool - Implementation
 */

#include "thread_pool.h"
#include <stdexcept>

ThreadPool::ThreadPool(size_t numThreads)
    : stopped_(false)
{
    // Determine number of threads
    if (numThreads == 0) {
        // Default to hardware concurrency, minimum 1
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) {
            numThreads = 1;
        }
    }
    
    // Create worker threads
    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back(&ThreadPool::workerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    // Signal threads to stop
    stopped_ = true;
    condition_.notify_all();
    
    // Join all threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::workerThread() {
    while (true) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // Wait while queue is empty and not stopped
            condition_.wait(lock, [this] {
                return !tasks_.empty() || stopped_;
            });
            
            // Exit if stopped and no more tasks
            if (stopped_ && tasks_.empty()) {
                return;
            }
            
            // Get next task
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }
        
        // Execute task (if we got one)
        if (task) {
            try {
                task();
            } catch (...) {
                // Exceptions are captured by std::future
                // Log if needed, but don't crash
            }
        }
    }
}