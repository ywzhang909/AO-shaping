/**
 * DM Control - C ABI Wrapper
 * Provides C-compatible interface to C++ Controller class
 */

#include "dm_control.h"
#include "controller.h"
#include <map>
#include <mutex>

//============================================================================
// Global State: Handle Map
//============================================================================

// Mutex for handle map
static std::mutex g_handleMutex;
static std::map<DM_Handle, std::unique_ptr<DM::Controller>> g_controllers;
static DM_Handle g_nextHandle = reinterpret_cast<DM_Handle>(1);

//============================================================================
// Internal Functions
//============================================================================

static DM_Handle createHandle() {
    std::lock_guard<std::mutex> lock(g_handleMutex);
    DM_Handle handle = g_nextHandle++;
    g_controllers[handle] = std::make_unique<DM::Controller>();
    return handle;
}

static DM::Controller* getController(DM_Handle handle) {
    if (handle == nullptr) return nullptr;
    std::lock_guard<std::mutex> lock(g_handleMutex);
    auto it = g_controllers.find(handle);
    if (it == g_controllers.end()) return nullptr;
    return it->second.get();
}

static void destroyHandle(DM_Handle handle) {
    if (handle == nullptr) return;
    std::lock_guard<std::mutex> lock(g_handleMutex);
    auto it = g_controllers.find(handle);
    if (it != g_controllers.end()) {
        it->second->Disconnect();
        g_controllers.erase(it);
    }
}

//============================================================================
// C ABI Implementation
//============================================================================

DM_Handle DM_Create(void) {
    return createHandle();
}

void DM_Destroy(DM_Handle handle) {
    destroyHandle(handle);
}

int DM_Init(DM_Handle handle, const char* ipFilePath, const char* mappingFilePath) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->Init(ipFilePath, mappingFilePath);
}

int DM_Disconnect(DM_Handle handle) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->Disconnect();
}

int DM_IsConnected(DM_Handle handle) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return 0;
    return ctrl->IsConnected() ? 1 : 0;
}

int DM_SetVoltageAllControllers(DM_Handle handle, float voltage, float higher, float lower) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetVoltageAllControllers(voltage, higher, lower);
}

DM_AsyncTaskId DM_SetVoltageAllControllersAsync(DM_Handle handle, float voltage, float higher, float lower) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return 0;
    return ctrl->SetVoltageAllControllersAsync(voltage, higher, lower);
}

int DM_WaitForAsync(DM_Handle handle, DM_AsyncTaskId taskId, int timeoutMs) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->WaitForAsync(taskId, timeoutMs);
}

int DM_IsAsyncDone(DM_Handle handle, DM_AsyncTaskId taskId) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return 0;
    return ctrl->IsAsyncDone(taskId);
}

int DM_GetAsyncResult(DM_Handle handle, DM_AsyncTaskId taskId) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->GetAsyncResult(taskId);
}

int DM_SetActuatorVoltage(DM_Handle handle, int actuatorNumber, float voltage) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetActuatorVoltage(actuatorNumber, voltage);
}

int DM_SetControllerVoltage(DM_Handle handle, int controllerId, float voltage) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetControllerVoltage(controllerId, voltage);
}

int DM_SetChannelAllControllers(DM_Handle handle, int channel, float voltage) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetChannelAllControllers(channel, voltage);
}

int DM_ControlRelay(DM_Handle handle, int state) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->ControlRelay(state);
}

int DM_OpenRelay(DM_Handle handle) {
    return DM_ControlRelay(handle, 1);
}

int DM_CloseRelay(DM_Handle handle) {
    return DM_ControlRelay(handle, 0);
}

int DM_GetControllerIP(DM_Handle handle, int controllerId, char* ipBuffer, int bufferSize) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->GetControllerIP(controllerId, ipBuffer, bufferSize);
}

const char* DM_GetLastError(DM_Handle handle) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return "Invalid handle";
    return ctrl->GetLastError();
}

int DM_GetConnectedCount(DM_Handle handle) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return 0;
    return ctrl->GetConnectedCount();
}

int DM_SetControllerVoltageArray(DM_Handle handle, int controllerId, const float* voltages) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetControllerVoltageArray(controllerId, voltages);
}

int DM_InitAllActuators(DM_Handle handle) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->InitAllActuators();
}

int DM_GetActuatorMapping(DM_Handle handle, int actuatorNumber, int* outControllerId, int* outChannel) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->GetActuatorMapping(actuatorNumber, outControllerId, outChannel);
}

int DM_SetMultipleActuators(DM_Handle handle, int count, const int* actuatorNumbers, const float* voltages) {
    DM::Controller* ctrl = getController(handle);
    if (ctrl == nullptr) return DM_ERROR_NOT_INIT;
    return ctrl->SetMultipleActuators(count, actuatorNumbers, voltages);
}

const char* DM_GetVersion(void) {
    return DM::Controller::GetVersion();
}