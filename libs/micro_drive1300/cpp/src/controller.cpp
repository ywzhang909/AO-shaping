/**
 * DM Controller - Implementation
 * Cross-platform: Windows (Winsock2) / Linux / macOS (POSIX)
 */

#include "controller.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <condition_variable>

// Static initialization - using the getter
static bool& getWinsockInitialized() {
    static bool winsockInitialized_ = false;
    return winsockInitialized_;
}

//============================================================================
// Constructor / Destructor
//============================================================================
DM::Controller::Controller()
    : initialized_(false), nextTaskId_(1)
{
    controllers_.resize(DM_CONTROLLER_COUNT);
    actuatorMap_.resize(DM_MAX_ACTUATORS);
    
    // Create thread pool for async operations (26 threads for 26 controllers)
    threadPool_ = std::make_unique<ThreadPool>(26);
}

DM::Controller::~Controller() {
    Disconnect();
}

DM::Controller::Controller(Controller&& other) noexcept
    : controllers_(std::move(other.controllers_)),
      actuatorMap_(std::move(other.actuatorMap_)),
      initialized_(other.initialized_),
      lastError_(std::move(other.lastError_)),
      threadPool_(std::move(other.threadPool_)),
      asyncTasks_(std::move(other.asyncTasks_)),
      nextTaskId_(other.nextTaskId_.load())
{
    other.initialized_ = false;
}

DM::Controller& DM::Controller::operator=(Controller&& other) noexcept {
    if (this != &other) {
        Disconnect();
        controllers_ = std::move(other.controllers_);
        actuatorMap_ = std::move(other.actuatorMap_);
        initialized_ = other.initialized_;
        lastError_ = std::move(other.lastError_);
        threadPool_ = std::move(other.threadPool_);
        asyncTasks_ = std::move(other.asyncTasks_);
        nextTaskId_ = other.nextTaskId_.load();
        other.initialized_ = false;
    }
    return *this;
}

//============================================================================
// Platform-Specific Network Initialization
//============================================================================
#if defined(DM_PLATFORM_WINDOWS)
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

static int initNetwork() {
    if (!getWinsockInitialized()) {
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            return DM_ERROR_CONNECT;
        }
        getWinsockInitialized() = true;
    }
    return DM_SUCCESS;
}

static void cleanupNetwork() {
    if (getWinsockInitialized()) {
        WSACleanup();
        getWinsockInitialized() = false;
    }
}

#define CLOSE_SOCKET closesocket

#else
#include <netdb.h>
#include <errno.h>

static int initNetwork() { return DM_SUCCESS; }
static void cleanupNetwork() {}

#define CLOSE_SOCKET close
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#endif

//============================================================================
// Initialization
//============================================================================
int DM::Controller::Init(const char* ipFilePath, const char* mappingFilePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return DM_SUCCESS;
    }
    
    // Initialize network
    int ret = initNetwork();
    if (ret != DM_SUCCESS) {
        lastError_ = "WSAStartup failed";
        return ret;
    }
    
    // Load IP addresses
    loadIPAddresses(ipFilePath);
    
    // Build actuator mapping
    (void)mappingFilePath; // Reserved for future
    buildDefaultMapping();
    
    // Connect to all controllers
    int connectedCount = 0;
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        ret = connectController(controllers_[i]);
        if (ret == DM_SUCCESS) {
            connectedCount++;
        }
    }
    
    if (connectedCount == 0) {
        lastError_ = "Failed to connect to any controller";
        return DM_ERROR_CONNECT;
    }
    
    initialized_ = true;
    return DM_SUCCESS;
}

int DM::Controller::Disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return DM_SUCCESS;
    }
    
    // Close all sockets
    for (auto& ctrl : controllers_) {
        if (ctrl.socket != INVALID_SOCKET) {
            CLOSE_SOCKET(ctrl.socket);
            ctrl.socket = INVALID_SOCKET;
            ctrl.connected = false;
        }
    }
    
    cleanupNetwork();
    
    // Clear async tasks
    {
        std::lock_guard<std::mutex> asyncLock(mutex_);
        asyncTasks_.clear();
    }
    
    initialized_ = false;
    return DM_SUCCESS;
}

//============================================================================
// Private Helper Methods
//============================================================================

void DM::Controller::buildDefaultIP(int controllerId, char* ipBuf, int bufSize) {
    snprintf(ipBuf, bufSize, "192.168.0.%d", 100 + controllerId);
}

int DM::Controller::getDefaultPort(int controllerId) {
    return 10000 + controllerId;
}

void DM::Controller::convertVoltage(float voltage, uint8_t* highByte, uint8_t* lowByte) {
    double value = (voltage + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0;
    uint16_t raw = (uint16_t)(value + 0.5);
    *highByte = (uint8_t)(raw / 256);
    *lowByte = (uint8_t)(raw % 256);
}

int DM::Controller::validateVoltage(float voltage, float higher, float lower) {
    if (voltage > higher || voltage < lower) {
        lastError_ = "Voltage out of range";
        return DM_ERROR_INVALID_VOLT;
    }
    return DM_SUCCESS;
}

int DM::Controller::validateChannel(int channel) {
    if (channel < 0 || channel > 49) {
        lastError_ = "Invalid channel (0-49)";
        return DM_ERROR_INVALID_CH;
    }
    return DM_SUCCESS;
}

int DM::Controller::validateActuator(int actuatorNumber) {
    if (actuatorNumber < 1 || actuatorNumber > DM_MAX_ACTUATORS) {
        lastError_ = "Invalid actuator number (1-1296)";
        return DM_ERROR_INVALID_ACT;
    }
    return DM_SUCCESS;
}

int DM::Controller::validateControllerId(int controllerId) {
    if (controllerId < 1 || controllerId > DM_CONTROLLER_COUNT) {
        lastError_ = "Invalid controller ID (1-26)";
        return DM_ERROR_CONNECT;
    }
    return DM_SUCCESS;
}

int DM::Controller::connectController(ControllerInfo& ctrl) {
#if defined(DM_PLATFORM_WINDOWS)
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
#endif
    
    if (sock == INVALID_SOCKET) {
        lastError_ = "Failed to create socket";
        return DM_ERROR_CONNECT;
    }
    
    // Set socket timeout
#if defined(DM_PLATFORM_WINDOWS)
    int timeout = DM_TIMEOUT_MS;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = DM_TIMEOUT_MS / 1000;
    timeout.tv_usec = (DM_TIMEOUT_MS % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
    
    struct sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons((uint16_t)ctrl.port);
    inet_pton(AF_INET, ctrl.ip.c_str(), &serverAddr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        CLOSE_SOCKET(sock);
        ctrl.socket = INVALID_SOCKET;
        ctrl.connected = false;
        return DM_ERROR_CONNECT;
    }
    
    ctrl.socket = sock;
    ctrl.connected = true;
    return DM_SUCCESS;
}

int DM::Controller::sendCommand(int socket, const uint8_t* data, int dataLen) {
    int sent = send(socket, (const char*)data, dataLen, 0);
    if (sent == SOCKET_ERROR) {
        lastError_ = "Failed to send command";
        return DM_ERROR_SEND;
    }
    return DM_SUCCESS;
}

void DM::Controller::loadIPAddresses(const char* filePath) {
    // Use defaults if no file
    if (filePath == nullptr) {
        for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
            char ip[32];
            buildDefaultIP(i + 1, ip, sizeof(ip));
            controllers_[i].ip = ip;
            controllers_[i].port = getDefaultPort(i + 1);
        }
        return;
    }
    
    // TODO: Load from file
    FILE* file = fopen(filePath, "r");
    if (file == nullptr) {
        // If file doesn't exist, fall back to defaults
        for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
            char ip[32];
            buildDefaultIP(i + 1, ip, sizeof(ip));
            controllers_[i].ip = ip;
            controllers_[i].port = getDefaultPort(i + 1);
        }
        return;
    }
    
    // Read file line by line
    char line[256];
    int controllerIndex = 0;
    while (fgets(line, sizeof(line), file) != nullptr && controllerIndex < DM_CONTROLLER_COUNT) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        // Parse line: expect format "IP PORT" or "CONTROLLER_ID IP PORT"
        int parsedControllerId = controllerIndex + 1; // Default to sequential
        char ip[32] = "";
        int port = 0;
        
        // Try to parse as "CONTROLLER_ID IP PORT"
        if (sscanf(line, "%d %31s %d", &parsedControllerId, ip, &port) == 3) {
            // Successfully parsed controller ID, IP, and port
            // Validate controller ID
            if (parsedControllerId < 1 || parsedControllerId > DM_CONTROLLER_COUNT) {
                // Invalid controller ID, treat as sequential
                parsedControllerId = controllerIndex + 1;
            }
        } else if (sscanf(line, "%31s %d", ip, &port) == 2) {
            // Parsed as "IP PORT", use sequential controller ID
            parsedControllerId = controllerIndex + 1;
        } else {
            // Failed to parse line, skip it
            continue;
        }
        
        // Validate parsed controller ID
        if (parsedControllerId >= 1 && parsedControllerId <= DM_CONTROLLER_COUNT) {
            int index = parsedControllerId - 1; // Convert to 0-based index
            controllers_[index].ip = ip;
            controllers_[index].port = port;
            controllerIndex++;
        }
    }
    
    fclose(file);
    
    // Fill any remaining controllers with defaults
    for (int i = controllerIndex; i < DM_CONTROLLER_COUNT; i++) {
        char ip[32];
        buildDefaultIP(i + 1, ip, sizeof(ip));
        controllers_[i].ip = ip;
        controllers_[i].port = getDefaultPort(i + 1);
    }
}

void DM::Controller::buildDefaultMapping() {
    // Default mapping: 36x36 grid distributed to 26 controllers
    int act = 1;
    for (int i = 0; i < 36; i++) {
        for (int j = 0; j < 36; j++) {
            int idx = act - 1;
            if (idx >= DM_MAX_ACTUATORS) break;
            
            // Simple distribution based on position
            int regionX = i / 9;
            int regionY = j / 9;
            int region = regionY * 4 + regionX;
            
            actuatorMap_[idx].controllerId = (region % 16) + 1;
            actuatorMap_[idx].channel = (i * 36 + j) % 50;
            
            act++;
        }
    }
}

//============================================================================
// Voltage Operations
//============================================================================

int DM::Controller::SetVoltageAllControllers(float voltage, float higher, float lower) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateVoltage(voltage, higher, lower);
    if (ret != DM_SUCCESS) return ret;
    
    uint8_t hv, lv;
    convertVoltage(voltage, &hv, &lv);
    
    // Command: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD
    uint8_t cmd[] = {0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD};
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (controllers_[i].connected && controllers_[i].socket != INVALID_SOCKET) {
            sendCommand(controllers_[i].socket, cmd, sizeof(cmd));
        }
    }
    
    return DM_SUCCESS;
}

DM_AsyncTaskId DM::Controller::SetVoltageAllControllersAsync(float voltage, float higher, float lower) {
    if (!initialized_) {
        lastError_ = "System not initialized";
        return 0;
    }
    
    int ret = validateVoltage(voltage, higher, lower);
    if (ret != DM_SUCCESS) return 0;
    
    // Generate task ID
    DM_AsyncTaskId taskId = nextTaskId_++;
    
    // Create shared result tracker
    auto result = std::make_shared<AsyncTaskResult>();
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        asyncTasks_[taskId] = result;
    }
    
    uint8_t hv, lv;
    convertVoltage(voltage, &hv, &lv);
    
    // Command: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD
    uint8_t cmd[] = {0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD};
    
    // Enqueue async tasks for all controllers
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (controllers_[i].connected && controllers_[i].socket != INVALID_SOCKET) {
            int controllerIndex = i;
            uint8_t* cmdCopy = new uint8_t[sizeof(cmd)];
            memcpy(cmdCopy, cmd, sizeof(cmd));
            
            threadPool_->enqueue([this, controllerIndex, cmdCopy, taskId]() {
                sendVoltageToControllerAsync(controllerIndex, cmdCopy, taskId);
                delete[] cmdCopy;
            });
        }
    }
    
    return taskId;
}

int DM::Controller::sendVoltageToControllerAsync(int controllerIndex, const uint8_t* cmd, uint64_t taskId) {
    std::shared_ptr<AsyncTaskResult> result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = asyncTasks_.find(taskId);
        if (it == asyncTasks_.end()) return DM_ERROR_THREAD;
        result = it->second;
    }
    
    int ret = sendCommand(controllers_[controllerIndex].socket, cmd, 7);
    
    std::lock_guard<std::mutex> lock(result->mtx);
    if (ret != DM_SUCCESS && result->errorCode == DM_SUCCESS) {
        result->errorCode = ret;
    }
    result->completed = true;
    result->cv.notify_all();
    
    return ret;
}

int DM::Controller::WaitForAsync(DM_AsyncTaskId taskId, int timeoutMs) {
    std::shared_ptr<AsyncTaskResult> result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = asyncTasks_.find(taskId);
        if (it == asyncTasks_.end()) return DM_ERROR_THREAD;
        result = it->second;
    }
    
    // Wait for completion
    if (timeoutMs < 0) {
        // Infinite wait
        std::unique_lock<std::mutex> lock(result->mtx);
        result->cv.wait(lock, [result] { return result->completed; });
    } else if (timeoutMs > 0) {
        // Timed wait
        std::unique_lock<std::mutex> lock(result->mtx);
        auto waitResult = result->cv.wait_for(lock, 
            std::chrono::milliseconds(timeoutMs),
            [result] { return result->completed; });
        if (!waitResult) return DM_ERROR_TIMEOUT;
    }
    // timeoutMs == 0: poll, don't wait
    
    return GetAsyncResult(taskId);
}

int DM::Controller::IsAsyncDone(DM_AsyncTaskId taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = asyncTasks_.find(taskId);
    if (it == asyncTasks_.end()) return 0;
    
    std::lock_guard<std::mutex> resultLock(it->second->mtx);
    return it->second->completed ? 1 : 0;
}

int DM::Controller::GetAsyncResult(DM_AsyncTaskId taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = asyncTasks_.find(taskId);
    if (it == asyncTasks_.end()) return DM_ERROR_THREAD;
    
    std::lock_guard<std::mutex> resultLock(it->second->mtx);
    return it->second->errorCode;
}

int DM::Controller::SetActuatorVoltage(int actuatorNumber, float voltage) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateActuator(actuatorNumber);
    if (ret != DM_SUCCESS) return ret;
    
    ret = validateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) return ret;
    
    int idx = actuatorNumber - 1;
    int ctrlId = actuatorMap_[idx].controllerId;
    int channel = actuatorMap_[idx].channel;
    
    ret = validateControllerId(ctrlId);
    if (ret != DM_SUCCESS) return ret;
    
    ret = validateChannel(channel);
    if (ret != DM_SUCCESS) return ret;
    
    if (!controllers_[ctrlId - 1].connected || 
        controllers_[ctrlId - 1].socket == INVALID_SOCKET) {
        lastError_ = "Controller not connected";
        return DM_ERROR_NO_CONNECT;
    }
    
    uint8_t hv, lv;
    convertVoltage(voltage, &hv, &lv);
    
    // Command: 0xAA 0xBB 0x04 channel hv lv 0xCC 0xDD
    uint8_t cmd[] = {0xAA, 0xBB, 0x04, (uint8_t)channel, hv, lv, 0xCC, 0xDD};
    return sendCommand(controllers_[ctrlId - 1].socket, cmd, sizeof(cmd));
}

int DM::Controller::SetControllerVoltage(int controllerId, float voltage) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateControllerId(controllerId);
    if (ret != DM_SUCCESS) return ret;
    
    ret = validateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) return ret;
    
    if (!controllers_[controllerId - 1].connected || 
        controllers_[controllerId - 1].socket == INVALID_SOCKET) {
        lastError_ = "Controller not connected";
        return DM_ERROR_NO_CONNECT;
    }
    
    uint8_t hv, lv;
    convertVoltage(voltage, &hv, &lv);
    
    // Command: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD
    uint8_t cmd[] = {0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD};
    return sendCommand(controllers_[controllerId - 1].socket, cmd, sizeof(cmd));
}

int DM::Controller::SetChannelAllControllers(int channel, float voltage) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateChannel(channel);
    if (ret != DM_SUCCESS) return ret;
    
    ret = validateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) return ret;
    
    uint8_t hv, lv;
    convertVoltage(voltage, &hv, &lv);
    
    // Command: 0xAA 0xBB 0x04 channel hv lv 0xCC 0xDD
    uint8_t cmd[] = {0xAA, 0xBB, 0x04, (uint8_t)channel, hv, lv, 0xCC, 0xDD};
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (controllers_[i].connected && controllers_[i].socket != INVALID_SOCKET) {
            sendCommand(controllers_[i].socket, cmd, sizeof(cmd));
        }
    }
    
    return DM_SUCCESS;
}

int DM::Controller::SetControllerVoltageArray(int controllerId, const float* voltages) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateControllerId(controllerId);
    if (ret != DM_SUCCESS) return ret;
    
    if (voltages == nullptr) {
        lastError_ = "Invalid voltage array";
        return DM_ERROR_INVALID_VOLT;
    }
    
    if (!controllers_[controllerId - 1].connected || 
        controllers_[controllerId - 1].socket == INVALID_SOCKET) {
        lastError_ = "Controller not connected";
        return DM_ERROR_NO_CONNECT;
    }
    
    // Build command: 0xAA 0xBB 0x09 + 50*2 voltage bytes + 0xCC 0xDD
    uint8_t cmd[4 + 100 + 2];
    cmd[0] = 0xAA;
    cmd[1] = 0xBB;
    cmd[2] = 0x09;
    
    for (int i = 0; i < 50; i++) {
        float v = voltages[i];
        if (v < -20.0f) v = -20.0f;
        if (v > 120.0f) v = 120.0f;
        uint8_t hv, lv;
        convertVoltage(v, &hv, &lv);
        cmd[3 + i * 2] = hv;
        cmd[3 + i * 2 + 1] = lv;
    }
    
    cmd[3 + 100] = 0xCC;
    cmd[3 + 101] = 0xDD;
    
    return sendCommand(controllers_[controllerId - 1].socket, cmd, sizeof(cmd));
}

int DM::Controller::SetMultipleActuators(int count, const int* actuatorNumbers, const float* voltages) {
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    if (count <= 0 || actuatorNumbers == nullptr || voltages == nullptr) {
        lastError_ = "Invalid parameters";
        return DM_ERROR_INVALID_ACT;
    }
    
    int ret = DM_SUCCESS;
    for (int i = 0; i < count; i++) {
        int r = SetActuatorVoltage(actuatorNumbers[i], voltages[i]);
        if (r != DM_SUCCESS) ret = r;
    }
    
    return ret;
}

int DM::Controller::InitAllActuators() {
    return SetVoltageAllControllers(0.0f, 120.0f, -20.0f);
}

//============================================================================
// Relay Control
//============================================================================

int DM::Controller::ControlRelay(int state) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    // Command: 0xAA 0xBB 0x06 0xCC 0xDD (open) or 0xAA 0xBB 0x07 0xCC 0xDD (close)
    uint8_t cmdOpen[] = {0xAA, 0xBB, 0x06, 0xCC, 0xDD};
    uint8_t cmdClose[] = {0xAA, 0xBB, 0x07, 0xCC, 0xDD};
    uint8_t* cmd = state ? cmdOpen : cmdClose;
    int cmdLen = sizeof(cmdOpen);
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (controllers_[i].connected && controllers_[i].socket != INVALID_SOCKET) {
            sendCommand(controllers_[i].socket, cmd, cmdLen);
        }
    }
    
    return DM_SUCCESS;
}

//============================================================================
// Utility Methods
//============================================================================

int DM::Controller::GetControllerIP(int controllerId, char* ipBuffer, int bufferSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || ipBuffer == nullptr || bufferSize <= 0) {
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateControllerId(controllerId);
    if (ret != DM_SUCCESS) return ret;
    
    strncpy(ipBuffer, controllers_[controllerId - 1].ip.c_str(), bufferSize - 1);
    ipBuffer[bufferSize - 1] = '\0';
    
    return DM_SUCCESS;
}

int DM::Controller::GetConnectedCount() const {
    int count = 0;
    for (const auto& ctrl : controllers_) {
        if (ctrl.connected) count++;
    }
    return count;
}

int DM::Controller::GetActuatorMapping(int actuatorNumber, int* outControllerId, int* outChannel) {
    if (!initialized_) {
        lastError_ = "System not initialized";
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = validateActuator(actuatorNumber);
    if (ret != DM_SUCCESS) return ret;
    
    int idx = actuatorNumber - 1;
    *outControllerId = actuatorMap_[idx].controllerId;
    *outChannel = actuatorMap_[idx].channel;
    
    return DM_SUCCESS;
}