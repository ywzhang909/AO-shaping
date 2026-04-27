/**
 * @file r50_power.h
 * @brief R50Power Micro Deformable Mirror Driver (C Implementation)
 *
 * This provides a C interface for the R50Power 50-channel micro deformable mirror,
 * compatible with the MATLAB R50Power class protocol.
 *
 * @author AO-Shaping
 * @date 2026
 *
 * Reference:
 *   - docs/micro deformable mirror/R50Power.m
 *   - docs/micro deformable mirror/Demo.m
 *
 * Example:
 *   @code
 *   R50Power* dm = r50_power_create("192.168.0.101", 10101);
 *   r50_power_open(dm);
 *   r50_power_set_relay_state(dm, 1);
 *   r50_power_set_channel_voltage(dm, 0, 2.5);
 *
 *   double voltages[50];
 *   for (int i = 0; i < 50; i++) voltages[i] = 0.0;
 *   r50_power_set_all_voltage_by_arr(dm, voltages);
 *
 *   r50_power_close(dm);
 *   r50_power_destroy(dm);
 *   @endcode
 */

#ifndef R50_POWER_H
#define R50_POWER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/*******************************************************************************
 * Constants
 ******************************************************************************/

/** Number of DM channels */
#define R50_POWER_NUM_CHANNELS 50

/** Minimum voltage (V) */
#define R50_POWER_VOLTAGE_MIN -1.0

/** Maximum voltage (V) */
#define R50_POWER_VOLTAGE_MAX 6.5

/** Protocol header/footer */
#define R50_POWER_HEADER_AA 0xAA
#define R50_POWER_HEADER_BB 0xBB
#define R50_POWER_FOOTER_CC 0xCC
#define R50_POWER_FOOTER_DD 0xDD

/** Command codes */
#define R50_POWER_CMD_SET_CHANNEL_VOLTAGE    0x04
#define R50_POWER_CMD_SET_ALL_CHANNEL_VOLTAGE 0x08
#define R50_POWER_CMD_SET_ALL_VOLTAGE_BY_ARR 0x09
#define R50_POWER_CMD_SET_RELAY_ON         0x06
#define R50_POWER_CMD_SET_RELAY_OFF        0x07
#define R50_POWER_CMD_SET_IP            0x06

/*******************************************************************************
 * Data Types
 ******************************************************************************/

/** R50Power device handle */
typedef struct R50Power R50Power;

/** Relay state */
typedef enum {
    R50_POWER_RELAY_OFF = 0,
    R50_POWER_RELAY_ON  = 1
} R50PowerRelayState;

/** Error codes */
typedef enum {
    R50_POWER_SUCCESS = 0,
    R50_POWER_ERROR_CONNECTION = -1,
    R50_POWER_ERROR_INVALID_CHANNEL = -2,
    R50_POWER_ERROR_INVALID_VOLTAGE = -3,
    R50_POWER_ERROR_SEND_FAILED = -4,
    R50_POWER_ERROR_INVALID_IP = -5
} R50PowerError;

/*******************************************************************************
 * Device Functions
 ******************************************************************************/

/**
 * @brief Create R50Power device handle
 *
 * @param ip_address Device IP address (e.g., "192.168.0.101")
 * @param port    Device TCP port (default: 10101)
 * @return Device handle, or NULL on failure
 */
R50Power* r50_power_create(const char* ip_address, int port);

/**
 * @brief Destroy R50Power device handle
 *
 * @param dev Device handle
 */
void r50_power_destroy(R50Power* dev);

/**
 * @brief Open TCP connection to device
 *
 * @param dev Device handle
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_open(R50Power* dev);

/**
 * @brief Close TCP connection
 *
 * @param dev Device handle
 */
void r50_power_close(R50Power* dev);

/**
 * @brief Check if connected
 *
 * @param dev Device handle
 * @return true if connected, false otherwise
 */
bool r50_power_is_connected(const R50Power* dev);

/**
 * @brief Get hardware information
 *
 * @param dev Device handle
 * @return Hardware info string (caller must free), or NULL on failure
 */
char* r50_power_get_hardware_info(const R50Power* dev);

/*******************************************************************************
 * Voltage Control Functions
 ******************************************************************************/

/**
 * @brief Set single channel voltage
 *
 * @param dev     Device handle
 * @param channel Channel number (0-49)
 * @param voltage Voltage value (-1.0 to 6.5 V)
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_set_channel_voltage(R50Power* dev, int channel, double voltage);

/**
 * @brief Set all channels to the same voltage
 *
 * @param dev     Device handle
 * @param voltage Voltage value for all channels
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_set_all_channel_voltage(R50Power* dev, double voltage);

/**
 * @brief Set all channels by voltage array
 *
 * Note: Uses +20V offset in conversion per R50Power protocol.
 *
 * @param dev      Device handle
 * @param voltages Array of 50 voltage values
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_set_all_voltage_by_arr(R50Power* dev, const double* voltages);

/**
 * @brief Set relay state
 *
 * @param dev  Device handle
 * @param state Relay state (0=off, 1=on)
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_set_relay_state(R50Power* dev, int state);

/**
 * @brief Set device IP address
 *
 * @param dev       Device handle
 * @param ip_address New IP address (e.g., "192.168.0.101")
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_set_ip(R50Power* dev, const char* ip_address);

/*******************************************************************************
 * Query Functions
 ******************************************************************************/

/**
 * @brief Get current actuator positions
 *
 * @param dev     Device handle
 * @param voltages Output array (must be pre-allocated, 50 elements)
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_get_actuator_positions(const R50Power* dev, double* voltages);

/*******************************************************************************
 * Utility Functions
 ******************************************************************************/

/**
 * @brief Convert voltage to high/low bytes
 *
 * Based on MATLAB formula:
 *   value = (voltage + 1) / 20 / 3.4 / 3.3 * 65535.0
 *   highByte = floor(value / 255)
 *   lowByte = floor(mod(value, 255))
 *
 * @param voltage Voltage value
 * @param high_byte Output: high byte
 * @param low_byte Output: low byte
 */
void r50_power_voltage_to_bytes(double voltage, uint8_t* high_byte, uint8_t* low_byte);

/**
 * @brief Convert voltage to bytes with +20V offset (for SetAllVoltageByArr)
 *
 * @param voltage Voltage value
 * @param high_byte Output: high byte
 * @param low_byte Output: low byte
 */
void r50_power_voltage_to_bytes_offset(double voltage, uint8_t* high_byte, uint8_t* low_byte);

/**
 * @brief Reset all channels to 0V
 *
 * @param dev Device handle
 * @return R50_POWER_SUCCESS on success, error code on failure
 */
int r50_power_reset_all(R50Power* dev);

/**
 * @brief Get error message string
 *
 * @param error_code Error code
 * @return Error message string
 */
const char* r50_power_error_message(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* R50_POWER_H */