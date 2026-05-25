/**
 * DM Control Library - Cross-Platform Deformable Mirror Control
 * 
 * Controls 26 R50 Power controllers via TCP/IP
 * Each controller has 50 channels, total 1296 actuators (39x39 array, 1521 logical)
 *
 * IP Address Range: 192.168.0.101 - 192.168.0.126
 * Port: 10000 + ip_suffix (e.g., 192.168.0.101:10101)
 * 
 * Compile:
 *   Windows: cl /LD dm_control.c /I. /Ws2_32.lib
 *   Linux:   gcc -shared -fPIC -o libdm_control.so dm_control.c -lpthread
 *   macOS:   gcc -shared -fPIC -o libdm_control.dylib dm_control.c -lpthread
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
 * API Functions
 *===========================================================================*/

/**
 * Initialize the DM system
 * Loads IP addresses and actuator mapping, establishes TCP connections
 * 
 * @param ipFilePath - Path to IP address file (NULL for default)
 * @param mappingFilePath - Path to actuator mapping (NULL for default)
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_Init(const char* ipFilePath, const char* mappingFilePath);

/**
 * Disconnect and cleanup - closes all TCP connections and frees resources
 * @return DM_SUCCESS on success
 */
DM_API int DM_Disconnect(void);

/**
 * Check if system is initialized
 * @return 1 if initialized, 0 if not
 */
DM_API int DM_IsConnected(void);

/**
 * Set voltage to ALL channels on ALL 26 controllers
 * @param voltage - Voltage value (range: -20 to 120 V)
 * @param higher - Upper limit for validation
 * @param lower - Lower limit for validation
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetVoltageAllControllers(float voltage, float higher, float lower);

/**
 * Set voltage to specific actuator (1-1296)
 * Uses actuator mapping to determine controller and channel
 * @param actuatorNumber - Actuator number (1-1296)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetActuatorVoltage(int actuatorNumber, float voltage);

/**
 * Set voltage to ALL 50 channels of a specific controller
 * @param controllerId - Controller ID (1-26)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetControllerVoltage(int controllerId, float voltage);

/**
 * Set voltage to specific channel on ALL 26 controllers
 * @param channel - Channel number (0-49)
 * @param voltage - Voltage value
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_SetChannelAllControllers(int channel, float voltage);

/**
 * Control relay (open/close)
 * @param state - 1 to open relay, 0 to close relay
 * @return DM_SUCCESS on success, error code on failure
 */
DM_API int DM_ControlRelay(int state);

/** Open relay (turn on) */
DM_API int DM_OpenRelay(void);

/** Close relay (turn off) */
DM_API int DM_CloseRelay(void);

/**
 * Get controller IP address
 * @param controllerId - Controller ID (1-26)
 * @param ipBuffer - Buffer to store IP string
 * @param bufferSize - Size of buffer
 * @return DM_SUCCESS on success
 */
DM_API int DM_GetControllerIP(int controllerId, char* ipBuffer, int bufferSize);

/** Get last error message */
DM_API const char* DM_GetLastError(void);

/** Get number of connected controllers */
DM_API int DM_GetConnectedCount(void);

/**
 * Set voltage array to all channels of a specific controller
 * Uses optimized batch command (0x09)
 * @param controllerId - Controller ID (1-26)
 * @param voltages - Array of 50 voltage values
 * @return DM_SUCCESS on success
 */
DM_API int DM_SetControllerVoltageArray(int controllerId, const float* voltages);

/** Initialize all actuators to zero voltage */
DM_API int DM_InitAllActuators(void);

/**
 * Get actuator mapping (controller and channel for given actuator)
 * @param actuatorNumber - Actuator number (1-1296)
 * @param outControllerId - Output: controller ID
 * @param outChannel - Output: channel number
 * @return DM_SUCCESS on success
 */
DM_API int DM_GetActuatorMapping(int actuatorNumber, int* outControllerId, int* outChannel);

/**
 * Set voltage to multiple actuators at once (batch operation)
 * @param count - Number of actuators
 * @param actuatorNumbers - Array of actuator numbers
 * @param voltages - Array of voltage values
 * @return DM_SUCCESS on success
 */
DM_API int DM_SetMultipleActuators(int count, const int* actuatorNumbers, const float* voltages);

/** Get library version */
DM_API const char* DM_GetVersion(void);

#ifdef __cplusplus
}
#endif

#endif /* DM_CONTROL_H */