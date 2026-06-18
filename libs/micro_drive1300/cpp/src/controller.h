/**
 * DM Controller - Internal C++ Class
 * Handles all deformable mirror control logic
 */

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <cstdint>  // For uint64_t
#include "dm_control.h"  // Must include the public header
#include "thread_pool.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <memory>

#if defined(DM_PLATFORM_WINDOWS)
    #include <winsock2.h>
    typedef int socklen_t;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
#endif

namespace DM {

//============================================================================
// Internal Structures
//============================================================================
struct ControllerInfo {
#if defined(DM_PLATFORM_WINDOWS)
    SOCKET socket;
#else
    int socket;
#endif
    std::string ip;
    int port;
    bool connected;
    
    ControllerInfo() : socket(INVALID_SOCKET), port(0), connected(false) {}
};

struct ActuatorMapping {
    int controllerId;  // 1-26
    int channel;        // 0-49
};

//============================================================================
// Async Task Result
//============================================================================
struct AsyncTaskResult {
    int errorCode;
    bool completed;
    std::mutex mtx;
    std::condition_variable cv;
    
    AsyncTaskResult() : errorCode(DM_SUCCESS), completed(false) {}
};

//============================================================================
// Controller Class
//============================================================================
class Controller {
public:
    Controller();
    ~Controller();
    
    // Disable copy
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;
    
    // Allow move
    Controller(Controller&&) noexcept;
    Controller& operator=(Controller&&) noexcept;
    
    // Initialization
    int Init(const char* ipFilePath, const char* mappingFilePath);
    int Disconnect();
    bool IsConnected() const { return initialized_; }
    
    // Voltage operations (synchronous)
    int SetVoltageAllControllers(float voltage, float higher, float lower);
    int SetActuatorVoltage(int actuatorNumber, float voltage);
    int SetControllerVoltage(int controllerId, float voltage);
    int SetChannelAllControllers(int channel, float voltage);
    int SetControllerVoltageArray(int controllerId, const float* voltages);
    int SetMultipleActuators(int count, const int* actuatorNumbers, const float* voltages);
    
    // Async voltage operations
    DM_AsyncTaskId SetVoltageAllControllersAsync(float voltage, float higher, float lower);
    int WaitForAsync(DM_AsyncTaskId taskId, int timeoutMs);
    int IsAsyncDone(DM_AsyncTaskId taskId);
    int GetAsyncResult(DM_AsyncTaskId taskId);
    
    // Relay control
    int ControlRelay(int state);
    int OpenRelay() { return ControlRelay(1); }
    int CloseRelay() { return ControlRelay(0); }
    
    // Utility
    int GetControllerIP(int controllerId, char* ipBuffer, int bufferSize);
    const char* GetLastError() const { return lastError_.c_str(); }
    int GetConnectedCount() const;
    int InitAllActuators();
    int GetActuatorMapping(int actuatorNumber, int* outControllerId, int* outChannel);
    
    // Version
    static const char* GetVersion() { return "1.1.0"; }

private:
    // Private helper methods
    int validateVoltage(float voltage, float higher, float lower);
    int validateChannel(int channel);
    int validateActuator(int actuatorNumber);
    int validateControllerId(int controllerId);
    
    void convertVoltage(float voltage, uint8_t* highByte, uint8_t* lowByte);
    void buildDefaultIP(int controllerId, char* ipBuf, int bufSize);
    int getDefaultPort(int controllerId);
    
    int connectController(ControllerInfo& ctrl);
    int sendCommand(int socket, const uint8_t* data, int dataLen);  // int works for both SOCKET and int
    
    void loadIPAddresses(const char* filePath);
    void buildDefaultMapping();
    
    // Async helper
    int sendVoltageToControllerAsync(int controllerIndex, const uint8_t* cmd, uint64_t taskId);

    // Member variables
    std::vector<ControllerInfo> controllers_;
    std::vector<ActuatorMapping> actuatorMap_;
    
    bool initialized_;
    std::string lastError_;
    
    mutable std::mutex mutex_;
    
    // Thread pool for async operations
    std::unique_ptr<ThreadPool> threadPool_;
    
    // Async task tracking
    std::map<DM_AsyncTaskId, std::shared_ptr<AsyncTaskResult>> asyncTasks_;
    std::atomic<DM_AsyncTaskId> nextTaskId_;
    
    // Network init flag (Windows) - made accessible for static functions
    static bool& getWinsockInitialized() {
        static bool winsockInitialized_ = false;
        return winsockInitialized_;
    }
};

} // namespace DM

#endif // CONTROLLER_H