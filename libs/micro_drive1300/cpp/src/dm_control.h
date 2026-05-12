/**
 * DM Control Library - C++ Cross-Platform Deformable Mirror Control
 * 
 * Controls 26 R50 Power controllers via TCP/IP
 * Each controller has 50 channels, total 1296 actuators (36x36 array)
 * 
 * IP Address Range: 192.168.0.101 - 192.168.0.126
 * Port: IP + 10100 (e.g., 192.168.0.101:10101)
 * 
 * Compile (Windows DLL):
 *   cl /LD /Iinclude src/*.cpp ws2_32.lib /DM_CONTROL_EXPORTS /EHsc /MD /W4
 * 
 * Compile (Linux):
 *   g++ -shared -fPIC -o libdm_control.so src/*.cpp -lpthread
 */

#ifndef DM_CONTROL_H
#define DM_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * Platform Detection
 *===========================================================================*/
#if defined(_WIN32) || defined(_WIN64)
    #define DM_PLATFORM_WINDOWS 1
    #if defined(_WIN64)
        #define DM_PLATFORM_WIN64 1
    #else
        #define DM_PLATFORM_WIN32 1
    #endif
#else
    #define DM_PLATFORM_POSIX 1
    #if defined(__APPLE__)
        #define DM_PLATFORM_MACOS 1
    #elif defined(__linux__)
        #define DM_PLATFORM_LINUX 1
    #endif
#endif

/*============================================================================
 * DLL Export/Import Macros
 *===========================================================================*/
#if defined(DM_CONTROL_EXPORTS)
    #if defined(DM_PLATFORM_WINDOWS)
        #define DM_API __declspec(dllexport)
    #else
        #define DM_API __attribute__((visibility("default")))
    #endif
#else
    #define DM_API
#endif

/*============================================================================
 * Standard Includes
 *===========================================================================*/
#include <stdint.h>
#include <stdbool.h>

/*============================================================================
 * Opaque Handle Type
 *===========================================================================*/
typedef void* DM_Handle;

/*============================================================================
 * Error Codes
 *===========================================================================*/
#define DM_SUCCESS             0
#define DM_ERROR_NOT_INIT     -1
#define DM_ERROR_CONNECT      -2
#define DM_ERROR_SEND         -3
#define DM_ERROR_INVALID_VOLT -4
#define DM_ERROR_INVALID_CH   -5
#define DM_ERROR_INVALID_ACT  -6
#define DM_ERROR_NO_CONNECT   -7
#define DM_ERROR_TIMEOUT      -8
#define DM_ERROR_THREAD       -9

/*============================================================================
 * Constants
 *===========================================================================*/
#define DM_MAX_CONTROLLERS    26
#define DM_MAX_CHANNELS       50
#define DM_MAX_ACTUATORS     1296
#define DM_VOLTAGE_MIN       -20.0f
#define DM_VOLTAGE_MAX       120.0f
#define DM_CONTROLLER_COUNT   26
#define DM_TIMEOUT_MS        5000

/*============================================================================
 * Async Task Handle
 *===========================================================================*/
typedef uint64_t DM_AsyncTaskId;

/*============================================================================
 * API Functions
 *===========================================================================*/

/**
 * Create a new DM controller instance
 * @return Handle to the controller, or NULL on failure
 */
DM_API DM_Handle DM_Create(void);

/**
 * Destroy a DM controller instance
 * @param handle - Controller handle
 */
DM_API void DM_Destroy(DM_Handle handle);

/**
 * Initialize the DM system
 * Loads IP addresses and actuator mapping, establishes TCP connections
 * 
 * @param handle - Controller handle
 * @param ipFilePath - Path to IP address file (NULL for default)
 * @param mappingFilePath - Path to actuator mapping (NULL for default)
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_Init(DM_Handle handle, const char* ipFilePath, const char* mappingFilePath);

/**
 * Disconnect and cleanup - closes all TCP connections and frees resources
 * @param handle - Controller handle
 * @return DM_SUCCESS on success
 */
DM_API int DM_Disconnect(DM_Handle handle);

/**
 * Check if system is initialized
 * @param handle - Controller handle
 * @return 1 if initialized, 0 if not
 */
DM_API int DM_IsConnected(DM_Handle handle);

/**
 * Set voltage to ALL channels on ALL 26 controllers (SYNCHRONOUS)
 * @param handle - Controller handle
 * @param voltage - Voltage value (range: -20 to 120 V)
 * @param higher - Upper limit for validation
 * @param lower - Lower limit for validation
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetVoltageAllControllers(DM_Handle handle, float voltage, float higher, float lower);

/**
 * Set voltage to ALL channels on ALL 26 controllers (ASYNCHRONOUS)
 * Non-blocking - returns immediately
 * @param handle - Controller handle
 * @param voltage - Voltage value (range: -20 to 120 V)
 * @param higher - Upper limit for validation
 * @param lower - Lower limit for validation
 * @return Async task ID for tracking, 0 on error
 */
DM_API DM_AsyncTaskId DM_SetVoltageAllControllersAsync(DM_Handle handle, float voltage, float higher, float lower);

/**
 * Wait for async task to complete
 * @param handle - Controller handle
 * @param taskId - Async task ID from DM_SetVoltageAllControllersAsync
 * @param timeoutMs - Timeout in milliseconds (-1 for infinite, 0 for poll)
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_WaitForAsync(DM_Handle handle, DM_AsyncTaskId taskId, int timeoutMs);

/**
 * Check if async task is done
 * @param handle - Controller handle
 * @param taskId - Async task ID
 * @return 1 if done, 0 if still running
 */
DM_API int DM_IsAsyncDone(DM_Handle handle, DM_AsyncTaskId taskId);

/**
 * Get async task result
 * @param handle - Controller handle
 * @param taskId - Async task ID
 * @return DM_SUCCESS if all sends succeeded, error code on failure
 */
DM_API int DM_GetAsyncResult(DM_Handle handle, DM_AsyncTaskId taskId);

/**
 * Set voltage to specific actuator (1-1296)
 * Uses actuator mapping to determine controller and channel
 * @param handle - Controller handle
 * @param actuatorNumber - Actuator number (1-1296)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetActuatorVoltage(DM_Handle handle, int actuatorNumber, float voltage);

/**
 * Set voltage to ALL 50 channels of a specific controller
 * @param handle - Controller handle
 * @param controllerId - Controller ID (1-26)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetControllerVoltage(DM_Handle handle, int controllerId, float voltage);

/**
 * Set voltage to specific channel on ALL 26 controllers
 * @param handle - Controller handle
 * @param channel - Channel number (0-49)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetChannelAllControllers(DM_Handle handle, int channel, float voltage);

/**
 * Control relay (open/close)
 * @param handle - Controller handle
 * @param state - 1 to open relay, 0 to close relay
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_ControlRelay(DM_Handle handle, int state);

/** Open relay (turn on) */
DM_API int DM_OpenRelay(DM_Handle handle);

/** Close relay (turn off) */
DM_API int DM_CloseRelay(DM_Handle handle);

/**
 * Get controller IP address
 * @param handle - Controller handle
 * @param controllerId - Controller ID (1-26)
 * @param ipBuffer - Buffer to store IP string
 * @param bufferSize - Size of buffer
 * @return DM_SUCCESS on success
 */
DM_API int DM_GetControllerIP(DM_Handle handle, int controllerId, char* ipBuffer, int bufferSize);

/** Get last error message */
DM_API const char* DM_GetLastError(DM_Handle handle);

/** Get number of connected controllers */
DM_API int DM_GetConnectedCount(DM_Handle handle);

/**
 * Set voltage array to all channels of a specific controller
 * Uses optimized batch command (0x09)
 * @param handle - Controller handle
 * @param controllerId - Controller ID (1-26)
 * @param voltages - Array of 50 voltage values
 * @return DM_SUCCESS on success
 */
DM_API int DM_SetControllerVoltageArray(DM_Handle handle, int controllerId, const float* voltages);

/** Initialize all actuators to zero voltage */
DM_API int DM_InitAllActuators(DM_Handle handle);

/**
 * Get actuator mapping (controller and channel for given actuator)
 * @param handle - Controller handle
 * @param actuatorNumber - Actuator number (1-1296)
 * @param outControllerId - Output: controller ID
 * @param outChannel - Output: channel number
 * @return DM_SUCCESS on success
 */
DM_API int DM_GetActuatorMapping(DM_Handle handle, int actuatorNumber, int* outControllerId, int* outChannel);

/**
 * Set voltage to multiple actuators at once (batch operation)
 * @param handle - Controller handle
 * @param count - Number of actuators
 * @param actuatorNumbers - Array of actuator numbers
 * @param voltages - Array of voltage values
 * @return DM_SUCCESS on success
 */
DM_API int DM_SetMultipleActuators(DM_Handle handle, int count, const int* actuatorNumbers, const float* voltages);

/** Get library version */
DM_API const char* DM_GetVersion(void);

#ifdef __cplusplus
}
#endif

#endif /* DM_CONTROL_H */