/**
 * @file r50_power.c
 * @brief R50Power Micro Deformable Mirror Driver Implementation
 *
 * @author AO-Shaping
 * @date 2026
 */

#include "r50_power.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

/*******************************************************************************
 * Private Data Structures
 ******************************************************************************/

struct R50Power {
    /** IP address */
    char ip[16];

    /** TCP port */
    int port;

    /** Socket handle */
#ifdef _WIN32
    SOCKET sock;
#else
    int sock;
#endif

    /** Connection state */
    bool connected;

    /** Last voltages */
    double last_voltages[R50_POWER_NUM_CHANNELS];

    /** Relay state */
    int relay_state;

    /** Timeout (seconds) */
    double timeout;
};

/*******************************************************************************
 * Private Function Prototypes
 ******************************************************************************/

static int send_command(R50Power* dev, const uint8_t* data, int length);
static uint8_t calculate_checksum(const uint8_t* data, int length);

/*******************************************************************************
 * Implementation
 ******************************************************************************/

R50Power* r50_power_create(const char* ip_address, int port) {
    R50Power* dev;

    if (ip_address == NULL) {
        return NULL;
    }

    dev = (R50Power*)calloc(1, sizeof(R50Power));
    if (dev == NULL) {
        return NULL;
    }

    /* Initialize with defaults */
    strncpy(dev->ip, ip_address, sizeof(dev->ip) - 1);
    dev->port = port;
    dev->sock = -1;
    dev->connected = false;
    dev->relay_state = 0;
    dev->timeout = 10.0;

    /* Initialize voltages to zero */
    memset(dev->last_voltages, 0, sizeof(dev->last_voltages));

#ifdef _WIN32
    /* Initialize Winsock on Windows */
    WSADATA wsa_data;
    static bool wsa_initialized = false;
    if (!wsa_initialized) {
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
        wsa_initialized = true;
    }
#endif

    return dev;
}

void r50_power_destroy(R50Power* dev) {
    if (dev == NULL) {
        return;
    }

    if (dev->connected) {
        r50_power_close(dev);
    }

    free(dev);
}

int r50_power_open(R50Power* dev) {
    struct sockaddr_in server_addr;
#ifdef _WIN32
    SOCKET sock;
#else
    int sock;
#endif

    if (dev == NULL) {
        return R50_POWER_ERROR_CONNECTION;
    }

    if (dev->connected) {
        return R50_POWER_SUCCESS;
    }

    /* Create socket */
#ifdef _WIN32
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET) {
        return R50_POWER_ERROR_CONNECTION;
    }
#else
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return R50_POWER_ERROR_CONNECTION;
    }
#endif

    /* Set timeout */
#ifdef _WIN32
    DWORD timeout_ms = (DWORD)(dev->timeout * 1000);
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv;
    tv.tv_sec = (int)dev->timeout;
    tv.tv_usec = (int)((dev->timeout - tv.tv_sec) * 1000000);
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    /* Connect to server */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons((uint16_t)dev->port);
    server_addr.sin_addr.s_addr = inet_addr(dev->ip);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return R50_POWER_ERROR_CONNECTION;
    }

    dev->sock = sock;
    dev->connected = true;

    return R50_POWER_SUCCESS;
}

void r50_power_close(R50Power* dev) {
    if (dev == NULL) {
        return;
    }

    if (!dev->connected) {
        return;
    }

    /* Reset all voltages before closing */
    r50_power_reset_all(dev);

    /* Close socket */
#ifdef _WIN32
    if (dev->sock != INVALID_SOCKET) {
        closesocket(dev->sock);
        dev->sock = INVALID_SOCKET;
    }
#else
    if (dev->sock >= 0) {
        close(dev->sock);
        dev->sock = -1;
    }
#endif

    dev->connected = false;
}

bool r50_power_is_connected(const R50Power* dev) {
    return (dev != NULL) && dev->connected;
}

char* r50_power_get_hardware_info(const R50Power* dev) {
    char* info;
    int len;

    if (dev == NULL) {
        return NULL;
    }

    len = 256;
    info = (char*)malloc(len);
    if (info == NULL) {
        return NULL;
    }

    snprintf(info, len,
        "manufacturer=R50Power\n"
        "model=MicroDM-50\n"
        "ip_address=%s\n"
        "port=%d\n"
        "channel_count=%d\n"
        "voltage_range=[%.1f, %.1f]\n"
        "relay_state=%d\n",
        dev->ip,
        dev->port,
        R50_POWER_NUM_CHANNELS,
        R50_POWER_VOLTAGE_MIN,
        R50_POWER_VOLTAGE_MAX,
        dev->relay_state
    );

    return info;
}

int r50_power_set_channel_voltage(R50Power* dev, int channel, double voltage) {
    uint8_t high_byte, low_byte;
    uint8_t command[8];
    int result;

    /* Validate inputs */
    if (dev == NULL || !dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

    if (channel < 0 || channel >= R50_POWER_NUM_CHANNELS) {
        return R50_POWER_ERROR_INVALID_CHANNEL;
    }

    if (voltage < R50_POWER_VOLTAGE_MIN || voltage > R50_POWER_VOLTAGE_MAX) {
        return R50_POWER_ERROR_INVALID_VOLTAGE;
    }

    /* Convert voltage to bytes */
    r50_power_voltage_to_bytes(voltage, &high_byte, &low_byte);

    /* Build command: AA BB 04 channel high low CC DD */
    command[0] = R50_POWER_HEADER_AA;
    command[1] = R50_POWER_HEADER_BB;
    command[2] = R50_POWER_CMD_SET_CHANNEL_VOLTAGE;
    command[3] = (uint8_t)channel;
    command[4] = high_byte;
    command[5] = low_byte;
    command[6] = R50_POWER_FOOTER_CC;
    command[7] = R50_POWER_FOOTER_DD;

    /* Send command */
    result = send_command(dev, command, 8);
    if (result != R50_POWER_SUCCESS) {
        return result;
    }

    /* Update internal state */
    dev->last_voltages[channel] = voltage;

    return R50_POWER_SUCCESS;
}

int r50_power_set_all_channel_voltage(R50Power* dev, double voltage) {
    uint8_t high_byte, low_byte;
    uint8_t command[8];
    int result;
    int i;

    /* Validate inputs */
    if (dev == NULL || !dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

    if (voltage < R50_POWER_VOLTAGE_MIN || voltage > R50_POWER_VOLTAGE_MAX) {
        return R50_POWER_ERROR_INVALID_VOLTAGE;
    }

    /* Convert voltage to bytes */
    r50_power_voltage_to_bytes(voltage, &high_byte, &low_byte);

    /* Build command: AA BB 08 high low CC DD */
    command[0] = R50_POWER_HEADER_AA;
    command[1] = R50_POWER_HEADER_BB;
    command[2] = R50_POWER_CMD_SET_ALL_CHANNEL_VOLTAGE;
    command[3] = high_byte;
    command[4] = low_byte;
    command[5] = R50_POWER_FOOTER_CC;
    command[6] = R50_POWER_FOOTER_DD;
    command[7] = 0;  /* Padding */

    /* Send command */
    result = send_command(dev, command, 7);
    if (result != R50_POWER_SUCCESS) {
        return result;
    }

    /* Update internal state */
    for (i = 0; i < R50_POWER_NUM_CHANNELS; i++) {
        dev->last_voltages[i] = voltage;
    }

    return R50_POWER_SUCCESS;
}

int r50_power_set_all_voltage_by_arr(R50Power* dev, const double* voltages) {
    uint8_t command[4 + R50_POWER_NUM_CHANNELS * 2 + 2];
    uint8_t high_byte, low_byte;
    int result;
    int i;

    /* Validate inputs */
    if (dev == NULL || !dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

    if (voltages == NULL) {
        return R50_POWER_ERROR_INVALID_VOLTAGE;
    }

    /* Build command header: AA BB 09 */
    command[0] = R50_POWER_HEADER_AA;
    command[1] = R50_POWER_HEADER_BB;
    command[2] = R50_POWER_CMD_SET_ALL_VOLTAGE_BY_ARR;

    /* Convert each voltage with +20V offset */
    for (i = 0; i < R50_POWER_NUM_CHANNELS; i++) {
        double v = voltages[i];

        /* Clamp voltage */
        if (v < R50_POWER_VOLTAGE_MIN) v = R50_POWER_VOLTAGE_MIN;
        if (v > R50_POWER_VOLTAGE_MAX) v = R50_POWER_VOLTAGE_MAX;

        r50_power_voltage_to_bytes_offset(v, &high_byte, &low_byte);
        command[3 + i * 2] = high_byte;
        command[3 + i * 2 + 1] = low_byte;
    }

    /* Footer: CC DD */
    command[3 + R50_POWER_NUM_CHANNELS * 2] = R50_POWER_FOOTER_CC;
    command[3 + R50_POWER_NUM_CHANNELS * 2 + 1] = R50_POWER_FOOTER_DD;

    /* Send command */
    result = send_command(dev, command, 4 + R50_POWER_NUM_CHANNELS * 2 + 2);
    if (result != R50_POWER_SUCCESS) {
        return result;
    }

    /* Update internal state */
    for (i = 0; i < R50_POWER_NUM_CHANNELS; i++) {
        dev->last_voltages[i] = voltages[i];
    }

    return R50_POWER_SUCCESS;
}

int r50_power_set_relay_state(R50Power* dev, int state) {
    uint8_t command[6];
    int result;

    /* Validate inputs */
    if (dev == NULL || !dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

    /* Build command */
    if (state) {
        /* Relay ON: AA BB 06 CC DD */
        command[0] = R50_POWER_HEADER_AA;
        command[1] = R50_POWER_HEADER_BB;
        command[2] = R50_POWER_CMD_SET_RELAY_ON;
        command[3] = R50_POWER_FOOTER_CC;
        command[4] = R50_POWER_FOOTER_DD;
        command[5] = 0;  /* Padding */

        result = send_command(dev, command, 5);
        dev->relay_state = 1;
    } else {
        /* Relay OFF: AA BB 07 CC DD */
        command[0] = R50_POWER_HEADER_AA;
        command[1] = R50_POWER_HEADER_BB;
        command[2] = R50_POWER_CMD_SET_RELAY_OFF;
        command[3] = R50_POWER_FOOTER_CC;
        command[4] = R50_POWER_FOOTER_DD;
        command[5] = 0;  /* Padding */

        result = send_command(dev, command, 5);
        dev->relay_state = 0;
    }

    return result;
}

int r50_power_set_ip(R50Power* dev, const char* ip_address) {
    uint8_t command[10];
    int ip_parts[4];
    int i;
    int result;

    /* Validate inputs */
    if (dev == NULL || !dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

    if (ip_address == NULL) {
        return R50_POWER_ERROR_INVALID_IP;
    }

    /* Parse IP address */
    if (sscanf(ip_address, "%d.%d.%d.%d",
               &ip_parts[0], &ip_parts[1], &ip_parts[2], &ip_parts[3]) != 4) {
        return R50_POWER_ERROR_INVALID_IP;
    }

    /* Validate IP parts */
    for (i = 0; i < 4; i++) {
        if (ip_parts[i] < 0 || ip_parts[i] > 255) {
            return R50_POWER_ERROR_INVALID_IP;
        }
    }

    /* Build command: AA BB 06 ip(1) ip(2) ip(3) ip(4) CC DD */
    command[0] = R50_POWER_HEADER_AA;
    command[1] = R50_POWER_HEADER_BB;
    command[2] = R50_POWER_CMD_SET_IP;
    command[3] = (uint8_t)ip_parts[0];
    command[4] = (uint8_t)ip_parts[1];
    command[5] = (uint8_t)ip_parts[2];
    command[6] = (uint8_t)ip_parts[3];
    command[7] = R50_POWER_FOOTER_CC;
    command[8] = R50_POWER_FOOTER_DD;
    command[9] = 0;  /* Padding */

    /* Send command */
    result = send_command(dev, command, 9);
    if (result == R50_POWER_SUCCESS) {
        strncpy(dev->ip, ip_address, sizeof(dev->ip) - 1);
    }

    return result;
}

int r50_power_get_actuator_positions(const R50Power* dev, double* voltages) {
    if (dev == NULL || voltages == NULL) {
        return R50_POWER_ERROR_CONNECTION;
    }

    memcpy(voltages, dev->last_voltages, sizeof(dev->last_voltages));

    return R50_POWER_SUCCESS;
}

void r50_power_voltage_to_bytes(double voltage, uint8_t* high_byte, uint8_t* low_byte) {
    double value;

    /* Clamp voltage */
    if (voltage < R50_POWER_VOLTAGE_MIN) voltage = R50_POWER_VOLTAGE_MIN;
    if (voltage > R50_POWER_VOLTAGE_MAX) voltage = R50_POWER_VOLTAGE_MAX;

    /* Convert: value = (voltage + 1) / 20 / 3.4 / 3.3 * 65535.0 */
    value = (voltage + 1.0) / 20.0 / 3.4 / 3.3 * 65535.0;

    *high_byte = (uint8_t)(value / 255.0);
    *low_byte = (uint8_t)fmod(value, 255.0);
}

void r50_power_voltage_to_bytes_offset(double voltage, uint8_t* high_byte, uint8_t* low_byte) {
    double value;

    /* Clamp voltage */
    if (voltage < R50_POWER_VOLTAGE_MIN) voltage = R50_POWER_VOLTAGE_MIN;
    if (voltage > R50_POWER_VOLTAGE_MAX) voltage = R50_POWER_VOLTAGE_MAX;

    /* Convert with +20V offset: value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0 */
    value = (voltage + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0;

    *high_byte = (uint8_t)(value / 255.0);
    *low_byte = (uint8_t)fmod(value, 255.0);
}

int r50_power_reset_all(R50Power* dev) {
    return r50_power_set_all_channel_voltage(dev, 0.0);
}

const char* r50_power_error_message(int error_code) {
    switch (error_code) {
        case R50_POWER_SUCCESS:
            return "Success";
        case R50_POWER_ERROR_CONNECTION:
            return "Connection error";
        case R50_POWER_ERROR_INVALID_CHANNEL:
            return "Invalid channel number";
        case R50_POWER_ERROR_INVALID_VOLTAGE:
            return "Invalid voltage value";
        case R50_POWER_ERROR_SEND_FAILED:
            return "Failed to send command";
        case R50_POWER_ERROR_INVALID_IP:
            return "Invalid IP address";
        default:
            return "Unknown error";
    }
}

/*******************************************************************************
 * Private Functions
 ******************************************************************************/

static int send_command(R50Power* dev, const uint8_t* data, int length) {
#ifdef _WIN32
    int sent;
#else
    ssize_t sent;
#endif

    if (dev == NULL || data == NULL || length <= 0) {
        return R50_POWER_ERROR_SEND_FAILED;
    }

    if (!dev->connected) {
        return R50_POWER_ERROR_CONNECTION;
    }

#ifdef _WIN32
    sent = send(dev->sock, (const char*)data, length, 0);
    if (sent == SOCKET_ERROR) {
        return R50_POWER_ERROR_SEND_FAILED;
    }
#else
    sent = send(dev->sock, data, length, 0);
    if (sent < 0) {
        return R50_POWER_ERROR_SEND_FAILED;
    }
#endif

    return R50_POWER_SUCCESS;
}

static uint8_t calculate_checksum(const uint8_t* data, int length) {
    uint8_t checksum = 0;
    int i;

    for (i = 0; i < length; i++) {
        checksum ^= data[i];
    }

    return checksum;
}

/*******************************************************************************
 * Example Usage
 ******************************************************************************/

#if 0

int main() {
    R50Power* dm;
    double voltages[R50_POWER_NUM_CHANNELS];
    int i, result;

    printf("R50Power Micro DM Demo\n");
    printf("======================\n\n");

    /* Create device */
    dm = r50_power_create("192.168.0.101", 10101);
    if (dm == NULL) {
        fprintf(stderr, "Failed to create device\n");
        return 1;
    }

    /* Open connection */
    result = r50_power_open(dm);
    if (result != R50_POWER_SUCCESS) {
        fprintf(stderr, "Failed to open connection: %s\n",
                r50_power_error_message(result));
        r50_power_destroy(dm);
        return 1;
    }
    printf("Connected to device\n");

    /* Open relay */
    r50_power_set_relay_state(dm, 1);
    printf("Relay opened\n");

    /* Set single channel */
    r50_power_set_channel_voltage(dm, 0, 2.5);
    printf("Set channel 0 to 2.5V\n");

    /* Set all channels to random voltages */
    srand(42);
    for (i = 0; i < R50_POWER_NUM_CHANNELS; i++) {
        voltages[i] = -1.0 + (6.5 - (-1.0)) * rand() / (double)RAND_MAX;
    }
    result = r50_power_set_all_voltage_by_arr(dm, voltages);
    printf("Set %d channels by array\n", R50_POWER_NUM_CHANNELS);

    /* Set all channels to same voltage */
    r50_power_set_all_channel_voltage(dm, 0.0);
    printf("Set all channels to 0V\n");

    /* Close relay */
    r50_power_set_relay_state(dm, 0);
    printf("Relay closed\n");

    /* Close connection */
    r50_power_close(dm);
    printf("Connection closed\n");

    /* Cleanup */
    r50_power_destroy(dm);

    printf("\nDemo complete\n");

    return 0;
}

#endif