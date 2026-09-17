 

### 1\. 从“拉取”到“推送”：为什么你需要回调采集？

上次我们聊了怎么用大恒相机SDK进行基础的相机操作和控制编程，算是把相机“点亮”了，能开关、能采图。但如果你真的把那个采单帧的代码跑起来，尤其是在工业检测这种对实时性有要求的场景里，你可能会发现一个问题：**效率太低了，而且容易丢帧**。

想象一下，你写了个程序，流程是“发送采集命令 -> 等待 -> 主动去‘拉’一帧图像 -> 处理”。这个“等待”和“拉取”的过程，程序是阻塞的。如果图像处理稍微慢了一点，或者相机帧率很高，下一帧图像已经来了，你上一帧还没处理完，那新来的这帧数据可能就被覆盖或者直接丢弃了。这在检测微小瑕疵（也就是我们标题里说的“小目标”）时是致命的，漏检一个可能就是一次质量事故。

所以，我们需要换一种思路：**让相机来“推”图像给我们，而不是我们去“拉”**。这就是“回调采集”机制的核心。你可以把它理解成订报纸。采单帧就像你每天跑到报亭问：“今天报纸来了吗？”（主动查询）。而回调采集就像你订了报纸，报社会在报纸印好的第一时间，直接派送员（回调函数）送到你家门口（你的程序），你只需要在家门口放个收报箱（图像处理函数）处理就行。

在大恒的Galaxy SDK中，这个“派送员”就是 `ICaptureEventHandler` 这个虚基 类 。你只需要继承它，实现里面的 `DoOnImageCaptured` 方法，告诉SDK：“图像来了就送到我这里处理”。然后SDK内部会维护一个高效的采集线程，一旦有新的 图像数据 从相机缓冲区就绪，它会立刻“回调”你提供的这个函数，把图像数据指针“推送”过来。你的主程序可以继续去干别的，比如更新UI、记录日志、或者处理其他业务逻辑，图像处理的工作完全由这个后台回调函数接管。

这种异步机制的好处是显而易见的：**高实时、低延迟、不阻塞主线程**。对于小 目标检测 这类任务，我们需要对每一帧图像进行复杂的算法分析（比如YOLO、SSD之类的推理），耗时可能几十毫秒。如果采用同步拉取，相机的高帧率优势就完全浪费了，还会导致缓冲区堆积。而回调采集配合多线程处理，可以让采集和运算并行，最大化利用硬件性能。我实测过一个项目，从同步拉取切换到回调采集后，系统能稳定处理的帧率提升了近3倍，而且再也没出现过因为处理不及时导致的丢帧告警。

### 2\. 打造你的专属“图像派送员”：ICaptureEventHandler实战

理论说再多不如一行代码。我们来看看怎么亲手打造这个“图像派送员”。首先，你需要创建一个自己的处理类，继承自 `ICaptureEventHandler`。

```cpp
// 我的图像处理回调类class MyCaptureEventHandler : public ICaptureEventHandler{public:    // 核心回调函数，当一帧图像就绪时，SDK会自动调用此函数    void DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam) override    {        // 1. 首先检查图像状态，非常重要！        if (objImageDataPointer->GetStatus() != GX_FRAME_STATUS_SUCCESS)        {            // 不是完整帧，可能是丢包、残帧等，记录日志或计数            std::cerr << "[Warning] 收到一帧无效图像，状态码: " << objImageDataPointer->GetStatus() << std::endl;            m_nErrorFrameCount++; // 假设有个成员变量记录错误帧            return; // 直接返回，不处理        }         // 2. 获取图像基本信息        uint64_t nWidth = objImageDataPointer->GetWidth();        uint64_t nHeight = objImageDataPointer->GetHeight();        GX_PIXEL_FORMAT_ENTRY emPixelFormat = objImageDataPointer->GetPixelFormat();        size_t nPayloadSize = objImageDataPointer->GetPayloadSize(); // 图像数据总大小         // 打印一下信息（生产环境建议去掉，影响性能）        // std::cout << "收到图像: " << nWidth << "x" << nHeight         //           << ", 格式: " << emPixelFormat         //           << ", 大小: " << nPayloadSize << " bytes" << std::endl;         // 3. 获取原始图像数据指针        // 注意：这个指针的生命周期仅在本次回调函数内有效！        // 如果你需要后续异步处理（比如放入队列），必须拷贝数据。        const void* pRawImageBuffer = objImageDataPointer->GetBuffer();         // 4. 【关键】将图像数据送入处理队列        // 这是工业检测中的常见做法，避免在回调函数内做耗时操作阻塞采集线程。        if (m_imageQueue.size() < MAX_QUEUE_SIZE) // 防止队列爆掉        {            // 深拷贝图像数据到自定义结构体            ProcessImageTask task;            task.width = nWidth;            task.height = nHeight;            task.pixelFormat = emPixelFormat;            task.imageData.resize(nPayloadSize);            memcpy(task.imageData.data(), pRawImageBuffer, nPayloadSize);            task.timestamp = std::chrono::system_clock::now(); // 打上时间戳             // 加锁，将任务推入队列            std::lock_guard<std::mutex> lock(m_queueMutex);            m_imageQueue.push(std::move(task));            m_queueCondition.notify_one(); // 通知处理线程        }        else        {            std::cerr << "[Error] 图像处理队列已满，丢弃一帧！" << std::endl;            m_nDroppedFrameCount++;        }         // 5. 更新统计信息（简单示例）        m_nTotalFramesReceived++;        if (m_nTotalFramesReceived % 100 == 0)        {            std::cout << "已接收 " << m_nTotalFramesReceived << " 帧，丢弃 " << m_nDroppedFrameCount << " 帧。" << std::endl;        }    } private:    // 你的图像处理队列和相关同步工具（如互斥锁、条件变量）    std::queue<ProcessImageTask> m_imageQueue;    std::mutex m_queueMutex;    std::condition_variable m_queueCondition;    int m_nTotalFramesReceived = 0;    int m_nDroppedFrameCount = 0;    int m_nErrorFrameCount = 0;};
```

上面这个类就是一个基础的框架。有几点我踩过坑，必须提醒你：

1.  **回调函数要快进快出**：`DoOnImageCaptured` 函数是运行在SDK内部的高优先级采集线程里的。如果你在这里做复杂的图像处理（比如调用OpenCV的`imshow`显示），会严重阻塞后续图像的接收，导致缓冲区溢出和丢帧。**正确的做法是只做必要的数据校验和拷贝，然后迅速将数据转移到另一个专门的处理线程**。
2.  **一定要检查图像状态**：`GetStatus()` 返回的不是`GX_FRAME_STATUS_SUCCESS`的图像，可能是网络丢包、传输错误导致的残帧，直接处理会导致程序崩溃或算法异常。
3.  **数据指针的生命周期**：`GetBuffer()` 返回的指针指向SDK内部管理的缓冲区。一旦本次回调函数结束，这个缓冲区可能会被SDK回收用于下一帧。所以，如果你需要“慢慢处理”，必须把数据`memcpy`出来，保存到自己的内存空间。

创建好处理类之后，下一步就是把它“注册”到流对象上，让SDK知道该把图像派送给谁。

```cpp
// 假设已经成功打开设备 objDevicePtr 和流 objStreamPtrCGXDevicePointer objDevicePtr = ...;CGXStreamPointer objStreamPtr = objDevicePtr->OpenStream(0); // 1. 创建我们自定义的回调处理器实例MyCaptureEventHandler* pMyHandler = new MyCaptureEventHandler(); // 2. 注册回调函数// 第一个参数：我们的处理器对象指针// 第二个参数：用户自定义参数 (void*)，可以传递任何上下文信息进去，
```

本文转自 <https://blog.csdn.net/weixin_29018815/article/details/158639262>，如有侵权，请联系删除。