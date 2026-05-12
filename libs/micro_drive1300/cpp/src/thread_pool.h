/**
 * ThreadPool - Generic thread pool for async operations
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>
#include <type_traits>

class ThreadPool {
public:
    /**
     * Construct thread pool with specified number of threads
     * @param numThreads - Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t numThreads = 0);
    
    /**
     * Destructor - waits for all tasks to complete and joins threads
     */
    ~ThreadPool();
    
    // Delete copy
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    
    /**
     * Enqueue a task to be executed asynchronously
     * Uses void return type for simplicity
     * @param f - Function to execute
     */
    template<typename F>
    void enqueue(F&& f);
    
    /**
     * Get the number of threads in the pool
     * @return Number of worker threads
     */
    size_t getThreadCount() const { return workers_.size(); }
    
    /**
     * Check if the pool is running
     * @return true if running, false if stopped
     */
    bool isRunning() const { return !stopped_; }

private:
    /**
     * Worker thread function - processes tasks from the queue
     */
    void workerThread();
    
    // Worker threads
    std::vector<std::thread> workers_;
    
    // Task queue
    std::queue<std::function<void()>> tasks_;
    
    // Synchronization
    std::mutex queueMutex_;
    std::condition_variable condition_;
    
    // State flags
    std::atomic<bool> stopped_;
};

// Simple template implementation
template<typename F>
void ThreadPool::enqueue(F&& f) {
    // Lock and enqueue
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        
        // Don't allow enqueueing after stopping the pool
        if (stopped_) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }
        
        // Add task to queue
        tasks_.emplace([f]() { f(); });
    }
    
    // Notify one worker thread
    condition_.notify_one();
}

#endif // THREAD_POOL_H