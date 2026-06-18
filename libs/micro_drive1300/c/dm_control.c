/**
 * DM Control Library - Implementation
 * Cross-platform: Windows (Winsock2) / Linux / macOS (POSIX)
 */

#include "dm_control.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*============================================================================
 * Platform-Specific Includes and Definitions
 *===========================================================================*/
#if defined(DM_PLATFORM_WINDOWS)
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
    #define SOCKET_TYPE SOCKET
    #define CLOSE_SOCKET closesocket
    #define SOCKET_INIT_NEEDED 1
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <pthread.h>
    #include <errno.h>
    #define SOCKET_TYPE int
    #define CLOSE_SOCKET close
    #define SOCKET_ERROR -1
    #define INVALID_SOCKET -1
    #define SOCKET_INIT_NEEDED 0
#endif

/*============================================================================
 * Internal Structures
 *===========================================================================*/
typedef struct {
    SOCKET_TYPE socket;
    char ip[32];
    int port;
    int connected;
} ControllerInfo;

typedef struct {
    int controllerId;  /* 1-26 */
    int channel;       /* 0-49 */
} ActuatorMapping;

/*============================================================================
 * Global State
 *===========================================================================*/
static ControllerInfo g_controllers[DM_CONTROLLER_COUNT];
static ActuatorMapping g_actuatorMap[DM_MAX_ACTUATORS];
static int g_initialized = 0;
static char g_lastError[256] = "";
static int g_winsockInitialized = 0;

/* Thread safety */
#if defined(DM_PLATFORM_WINDOWS)
    static CRITICAL_SECTION g_cs;
    static int g_criticalSectionInitialized = 0;
#else
    static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

/*============================================================================
 * Platform-Specific Helper Functions
 *===========================================================================*/
#if defined(DM_PLATFORM_WINDOWS)
static void Lock_init(void) {
    if (!g_criticalSectionInitialized) {
        InitializeCriticalSection(&g_cs);
        g_criticalSectionInitialized = 1;
    }
}
static void Lock_enter(void) {
    if (g_criticalSectionInitialized) EnterCriticalSection(&g_cs);
}
static void Lock_leave(void) {
    if (g_criticalSectionInitialized) LeaveCriticalSection(&g_cs);
}
#else
static void Lock_init(void) {}
static void Lock_enter(void) { pthread_mutex_lock(&g_mutex); }
static void Lock_leave(void) { pthread_mutex_unlock(&g_mutex); }
#endif

/*============================================================================
 * Network Initialization (Windows only)
 *===========================================================================*/
#if SOCKET_INIT_NEEDED
static int InitNetwork(void) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        snprintf(g_lastError, sizeof(g_lastError), "WSAStartup failed");
        return DM_ERROR_CONNECT;
    }
    g_winsockInitialized = 1;
    return DM_SUCCESS;
}
#else
static int InitNetwork(void) { return DM_SUCCESS; }
#endif

/*============================================================================
 * Network Cleanup (Windows only)
 *===========================================================================*/
#if SOCKET_INIT_NEEDED
static void CleanupNetwork(void) {
    if (g_winsockInitialized) {
        WSACleanup();
        g_winsockInitialized = 0;
    }
}
#else
static void CleanupNetwork(void) {}
#endif

/*============================================================================
 * Set Error Message
 *===========================================================================*/
static void SetError(const char* msg) {
    snprintf(g_lastError, sizeof(g_lastError), "%s", msg);
}

/*============================================================================
 * Build Default IP Address
 *===========================================================================*/
static void BuildDefaultIP(int controllerId, char* ipBuf, int bufSize) {
    /* IP format: 192.168.0.101 - 192.168.0.126 */
    snprintf(ipBuf, bufSize, "192.168.0.%d", 100 + controllerId);
}

/*============================================================================
 * Build Default Port
 *===========================================================================*/
static int GetDefaultPort(int controllerId) {
    /* Port: 10000 + ip_suffix, where ip_suffix = 100 + controllerId */
    /* Controller 1 -> IP 192.168.0.101 -> port 10101 */
    /* Controller 26 -> IP 192.168.0.126 -> port 10126 */
    return 10000 + (100 + controllerId); /* = 10100 + controllerId */
}

/*============================================================================
 * Convert Voltage to High/Low Bytes
 * Protocol: value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
 *
 * Byte extraction (consistent base-256):
 *   raw  = round(value)
 *   high = raw / 256  (equivalent to raw >> 8)
 *   low  = raw % 256  (equivalent to raw & 0xFF)
 *
 * This ensures high * 256 + low == raw for the full value range.
 * Matches Python: voltages_to_payload() in MicroDM.py.
 *============================================================================*/
static void ConvertVoltage(float voltage, uint8_t* highByte, uint8_t* lowByte) {
    double value = (voltage + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0;
    uint16_t raw = (uint16_t)(value + 0.5); /* Round to nearest */
    *highByte = (uint8_t)(raw / 256);
    *lowByte = (uint8_t)(raw % 256);
}

/*============================================================================
 * Validate Voltage
 *===========================================================================*/
static int ValidateVoltage(float voltage, float higher, float lower) {
    if (voltage > higher || voltage < lower) {
        SetError("Voltage out of range");
        return DM_ERROR_INVALID_VOLT;
    }
    return DM_SUCCESS;
}

/*============================================================================
 * Validate Channel
 *===========================================================================*/
static int ValidateChannel(int channel) {
    if (channel < 0 || channel > 49) {
        SetError("Invalid channel (0-49)");
        return DM_ERROR_INVALID_CH;
    }
    return DM_SUCCESS;
}

/*============================================================================
 * Validate Actuator Number
 *===========================================================================*/
static int ValidateActuator(int actuatorNumber) {
    if (actuatorNumber < 1 || actuatorNumber > DM_MAX_ACTUATORS) {
        SetError("Invalid actuator number (1-1296)");
        return DM_ERROR_INVALID_ACT;
    }
    return DM_SUCCESS;
}

/*============================================================================
 * Validate Controller ID
 *===========================================================================*/
static int ValidateControllerId(int controllerId) {
    if (controllerId < 1 || controllerId > DM_CONTROLLER_COUNT) {
        SetError("Invalid controller ID (1-26)");
        return DM_ERROR_CONNECT;
    }
    return DM_SUCCESS;
}

/*============================================================================
 * TCP Connect to Controller
 *===========================================================================*/
static int ConnectController(ControllerInfo* ctrl) {
    struct sockaddr_in serverAddr;
    
    ctrl->socket = socket(AF_INET, SOCK_STREAM, 0);
    if (ctrl->socket == INVALID_SOCKET) {
        SetError("Failed to create socket");
        return DM_ERROR_CONNECT;
    }

    /* Set socket timeout */
#if defined(DM_PLATFORM_WINDOWS)
    int timeout = DM_TIMEOUT_MS;
    setsockopt(ctrl->socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval timeout;
    timeout.tv_sec = DM_TIMEOUT_MS / 1000;
    timeout.tv_usec = (DM_TIMEOUT_MS % 1000) * 1000;
    setsockopt(ctrl->socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons((uint16_t)ctrl->port);
    inet_pton(AF_INET, ctrl->ip, &serverAddr.sin_addr);

    if (connect(ctrl->socket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        CLOSE_SOCKET(ctrl->socket);
        ctrl->socket = INVALID_SOCKET;
        ctrl->connected = 0;
        return DM_ERROR_CONNECT;
    }

    ctrl->connected = 1;
    return DM_SUCCESS;
}

/*============================================================================
 * Send Command to Controller
 *===========================================================================*/
static int SendCommand(SOCKET_TYPE socket, const uint8_t* data, int dataLen) {
    int sent = send(socket, (const char*)data, dataLen, 0);
    if (sent == SOCKET_ERROR) {
        SetError("Failed to send command");
        return DM_ERROR_SEND;
    }
    return DM_SUCCESS;
}

/*============================================================================
 * Build Actuator Mapping (Default 39x39 to 26 controllers)
 * Each controller handles ~50 channels, mapping 1296 actuators
 *
 * NOTE: This is a PLACEHOLDER mapping for basic testing.
 * The production mapping is defined in libs/micro_drive1300/wiring_map.json
 * which maps each needle pin (277-330) across multiple groups to:
 *   - Physical positions in a 39x39 actuator array
 *   - Controller IPs (via ip_suffix)
 *   - Payload byte positions (1-50) within each controller's frame
 *
 * For production use, load the wiring map and populate g_actuatorMap
 * from ChannelEntry data (see Python: MicroDM._build_channel_indices).
 *===========================================================================*/
static void BuildDefaultMapping(void) {
    /* Default mapping: 36x36 grid distributed to 26 controllers
     * Same algorithm as libs/dm_control.py and cpp/src/controller.cpp
     */
    for (int act = 0; act < DM_MAX_ACTUATORS; act++) {
        int row = act / 36;
        int col = act % 36;

        /* Divide 36x36 into 16 regions */
        int regionX = col / 9;
        int regionY = row / 9;
        int region = regionY * 4 + regionX;

        g_actuatorMap[act].controllerId = (region % 16) + 1;
        g_actuatorMap[act].channel = (act % 50);
    }
}

/*============================================================================
 * Load IP Addresses from File
 *===========================================================================*/
static int LoadIPAddresses(const char* filePath) {
    FILE* fp = NULL;
    char line[64];
    int lineNum = 0;
    
    if (filePath == NULL) {
        /* Use default IP addresses */
        for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
            BuildDefaultIP(i + 1, g_controllers[i].ip, sizeof(g_controllers[i].ip));
            g_controllers[i].port = GetDefaultPort(i + 1);
        }
        return DM_SUCCESS;
    }
    
    fp = fopen(filePath, "r");
    if (fp == NULL) {
        /* File not found, use defaults */
        for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
            BuildDefaultIP(i + 1, g_controllers[i].ip, sizeof(g_controllers[i].ip));
            g_controllers[i].port = GetDefaultPort(i + 1);
        }
        return DM_SUCCESS;
    }
    
    /* Parse IP file format from IP_1300.txt:
       Power1:
       192.168.0.101
    */
    while (fgets(line, sizeof(line), fp) != NULL) {
        /* Look for IP address line (contains dot) */
        if (strchr(line, '.') != NULL && lineNum < DM_CONTROLLER_COUNT) {
            /* Remove newline */
            line[strcspn(line, "\r\n")] = 0;
            /* Skip whitespace */
            char* ptr = line;
            while (*ptr == ' ' || *ptr == '\t') ptr++;
            
            strncpy(g_controllers[lineNum].ip, ptr, sizeof(g_controllers[lineNum].ip) - 1);
            g_controllers[lineNum].ip[sizeof(g_controllers[lineNum].ip) - 1] = '\0';
            g_controllers[lineNum].port = GetDefaultPort(lineNum + 1);
            lineNum++;
        }
    }
    fclose(fp);
    
    /* Fill remaining with defaults */
    for (int i = lineNum; i < DM_CONTROLLER_COUNT; i++) {
        BuildDefaultIP(i + 1, g_controllers[i].ip, sizeof(g_controllers[i].ip));
        g_controllers[i].port = GetDefaultPort(i + 1);
    }
    
    return DM_SUCCESS;
}

/*============================================================================
 * API Implementation
 *===========================================================================*/

DM_API int DM_Init(const char* ipFilePath, const char* mappingFilePath) {
    Lock_enter();
    
    if (g_initialized) {
        Lock_leave();
        return DM_SUCCESS;
    }
    
    /* Initialize network (Windows) */
    int ret = InitNetwork();
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    /* Initialize locks */
    Lock_init();
    
    /* Load IP addresses */
    LoadIPAddresses(ipFilePath);
    
    /* Build default actuator mapping */
    (void)mappingFilePath; /* Reserved for future Excel loading */
    BuildDefaultMapping();
    
    /* Initialize controller sockets */
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        g_controllers[i].socket = INVALID_SOCKET;
        g_controllers[i].connected = 0;
    }
    
    /* Connect to all controllers */
    int connectedCount = 0;
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        ret = ConnectController(&g_controllers[i]);
        if (ret == DM_SUCCESS) {
            connectedCount++;
        }
    }
    
    if (connectedCount == 0) {
        SetError("Failed to connect to any controller");
        Lock_leave();
        return DM_ERROR_CONNECT;
    }
    
    g_initialized = 1;
    Lock_leave();
    return DM_SUCCESS;
}

DM_API int DM_Disconnect(void) {
    Lock_enter();
    
    if (!g_initialized) {
        Lock_leave();
        return DM_SUCCESS;
    }
    
    /* Close all sockets */
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (g_controllers[i].socket != INVALID_SOCKET) {
            CLOSE_SOCKET(g_controllers[i].socket);
            g_controllers[i].socket = INVALID_SOCKET;
            g_controllers[i].connected = 0;
        }
    }
    
    CleanupNetwork();
    g_initialized = 0;
    
    Lock_leave();
    return DM_SUCCESS;
}

DM_API int DM_IsConnected(void) {
    return g_initialized ? 1 : 0;
}

DM_API int DM_SetVoltageAllControllers(float voltage, float higher, float lower) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateVoltage(voltage, higher, lower);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    uint8_t hv, lv;
    ConvertVoltage(voltage, &hv, &lv);
    
    /* Command: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD */
    uint8_t cmd[] = {0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD};
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (g_controllers[i].connected && g_controllers[i].socket != INVALID_SOCKET) {
            SendCommand(g_controllers[i].socket, cmd, sizeof(cmd));
        }
    }
    
    Lock_leave();
    return DM_SUCCESS;
}

DM_API int DM_SetActuatorVoltage(int actuatorNumber, float voltage) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateActuator(actuatorNumber);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    ret = ValidateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    int idx = actuatorNumber - 1;
    int ctrlId = g_actuatorMap[idx].controllerId;
    int channel = g_actuatorMap[idx].channel;
    
    /* Validate controller ID */
    ret = ValidateControllerId(ctrlId);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    ret = ValidateChannel(channel);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    if (!g_controllers[ctrlId - 1].connected || g_controllers[ctrlId - 1].socket == INVALID_SOCKET) {
        SetError("Controller not connected");
        Lock_leave();
        return DM_ERROR_NO_CONNECT;
    }
    
    uint8_t hv, lv;
    ConvertVoltage(voltage, &hv, &lv);
    
    /* Command: 0xAA 0xBB 0x04 channel hv lv 0xCC 0xDD */
    uint8_t cmd[] = {0xAA, 0xBB, 0x04, (uint8_t)channel, hv, lv, 0xCC, 0xDD};
    ret = SendCommand(g_controllers[ctrlId - 1].socket, cmd, sizeof(cmd));
    
    Lock_leave();
    return ret;
}

DM_API int DM_SetControllerVoltage(int controllerId, float voltage) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateControllerId(controllerId);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    ret = ValidateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    if (!g_controllers[controllerId - 1].connected || 
        g_controllers[controllerId - 1].socket == INVALID_SOCKET) {
        SetError("Controller not connected");
        Lock_leave();
        return DM_ERROR_NO_CONNECT;
    }
    
    uint8_t hv, lv;
    ConvertVoltage(voltage, &hv, &lv);
    
    /* Command: 0xAA 0xBB 0x08 hv lv 0xCC 0xDD */
    uint8_t cmd[] = {0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD};
    ret = SendCommand(g_controllers[controllerId - 1].socket, cmd, sizeof(cmd));
    
    Lock_leave();
    return ret;
}

DM_API int DM_SetChannelAllControllers(int channel, float voltage) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateChannel(channel);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    ret = ValidateVoltage(voltage, 120.0f, -20.0f);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    uint8_t hv, lv;
    ConvertVoltage(voltage, &hv, &lv);
    
    /* Command: 0xAA 0xBB 0x04 channel hv lv 0xCC 0xDD */
    uint8_t cmd[] = {0xAA, 0xBB, 0x04, (uint8_t)channel, hv, lv, 0xCC, 0xDD};
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (g_controllers[i].connected && g_controllers[i].socket != INVALID_SOCKET) {
            SendCommand(g_controllers[i].socket, cmd, sizeof(cmd));
        }
    }
    
    Lock_leave();
    return DM_SUCCESS;
}

DM_API int DM_ControlRelay(int state) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    /* Command: 0xAA 0xBB 0x06 0xCC 0xDD (open) or 0xAA 0xBB 0x07 0xCC 0xDD (close) */
    uint8_t cmdOpen[] = {0xAA, 0xBB, 0x06, 0xCC, 0xDD};
    uint8_t cmdClose[] = {0xAA, 0xBB, 0x07, 0xCC, 0xDD};
    uint8_t* cmd = state ? cmdOpen : cmdClose;
    int cmdLen = sizeof(cmdOpen);
    
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (g_controllers[i].connected && g_controllers[i].socket != INVALID_SOCKET) {
            SendCommand(g_controllers[i].socket, cmd, cmdLen);
        }
    }
    
    Lock_leave();
    return DM_SUCCESS;
}

DM_API int DM_OpenRelay(void) {
    return DM_ControlRelay(1);
}

DM_API int DM_CloseRelay(void) {
    return DM_ControlRelay(0);
}

DM_API int DM_GetControllerIP(int controllerId, char* ipBuffer, int bufferSize) {
    Lock_enter();
    
    if (!g_initialized || ipBuffer == NULL || bufferSize <= 0) {
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateControllerId(controllerId);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    strncpy(ipBuffer, g_controllers[controllerId - 1].ip, bufferSize - 1);
    ipBuffer[bufferSize - 1] = '\0';
    
    Lock_leave();
    return DM_SUCCESS;
}

DM_API const char* DM_GetLastError(void) {
    return g_lastError;
}

DM_API int DM_GetConnectedCount(void) {
    int count = 0;
    for (int i = 0; i < DM_CONTROLLER_COUNT; i++) {
        if (g_controllers[i].connected) count++;
    }
    return count;
}

DM_API int DM_SetControllerVoltageArray(int controllerId, const float* voltages) {
    Lock_enter();
    
    if (!g_initialized) {
        SetError("System not initialized");
        Lock_leave();
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateControllerId(controllerId);
    if (ret != DM_SUCCESS) {
        Lock_leave();
        return ret;
    }
    
    if (voltages == NULL) {
        SetError("Invalid voltage array");
        Lock_leave();
        return DM_ERROR_INVALID_VOLT;
    }
    
    if (!g_controllers[controllerId - 1].connected || 
        g_controllers[controllerId - 1].socket == INVALID_SOCKET) {
        SetError("Controller not connected");
        Lock_leave();
        return DM_ERROR_NO_CONNECT;
    }
    
    /* Build command: 0xAA 0xBB 0x09 + 50*2 voltage bytes + 0xCC 0xDD */
    uint8_t cmd[4 + 100 + 2]; /* Header + 50 voltages + footer */
    cmd[0] = 0xAA;
    cmd[1] = 0xBB;
    cmd[2] = 0x09;
    
    /* Convert all 50 voltages */
    for (int i = 0; i < 50; i++) {
        float v = voltages[i];
        if (v < -20.0f) v = -20.0f;
        if (v > 120.0f) v = 120.0f;
        uint8_t hv, lv;
        ConvertVoltage(v, &hv, &lv);
        cmd[3 + i * 2] = hv;
        cmd[3 + i * 2 + 1] = lv;
    }
    
    cmd[3 + 100] = 0xCC;
    cmd[3 + 101] = 0xDD;
    
    ret = SendCommand(g_controllers[controllerId - 1].socket, cmd, sizeof(cmd));
    
    Lock_leave();
    return ret;
}

DM_API int DM_InitAllActuators(void) {
    return DM_SetVoltageAllControllers(0.0f, 120.0f, -20.0f);
}

DM_API int DM_GetActuatorMapping(int actuatorNumber, int* outControllerId, int* outChannel) {
    if (!g_initialized) {
        return DM_ERROR_NOT_INIT;
    }
    
    int ret = ValidateActuator(actuatorNumber);
    if (ret != DM_SUCCESS) {
        return ret;
    }
    
    int idx = actuatorNumber - 1;
    *outControllerId = g_actuatorMap[idx].controllerId;
    *outChannel = g_actuatorMap[idx].channel;
    
    return DM_SUCCESS;
}

DM_API int DM_SetMultipleActuators(int count, const int* actuatorNumbers, const float* voltages) {
    if (!g_initialized) {
        return DM_ERROR_NOT_INIT;
    }
    
    if (count <= 0 || actuatorNumbers == NULL || voltages == NULL) {
        return DM_ERROR_INVALID_ACT;
    }
    
    int ret = DM_SUCCESS;
    for (int i = 0; i < count; i++) {
        int r = DM_SetActuatorVoltage(actuatorNumbers[i], voltages[i]);
        if (r != DM_SUCCESS) ret = r;
    }
    
    return ret;
}

DM_API const char* DM_GetVersion(void) {
    return "1.0.0";
}