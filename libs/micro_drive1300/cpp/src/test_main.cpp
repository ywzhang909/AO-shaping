/**
 * DM Control C++ Test Program
 * Tests all API functions including async operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>  // For Sleep on Windows

#include "dm_control.h"

/* ANSI color output */
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define RESET   "\x1b[0m"

static int tests_passed = 0;
static int tests_failed = 0;

void print_header(const char* title) {
    printf("\n" BLUE "========================================\n" RESET);
    printf(BLUE "  %s\n" RESET, title);
    printf(BLUE "========================================\n" RESET);
}

void print_test(const char* name, int passed) {
    if (passed) {
        printf(GREEN "[PASS] " RESET "%s\n", name);
        tests_passed++;
    } else {
        printf(RED "[FAIL] " RESET "%s\n", name);
        tests_failed++;
    }
}

int test_init(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_Init()..." RESET "\n");
    int ret = DM_Init(handle, NULL, NULL);
    print_test("DM_Init() returns success or connect error", 
               ret == DM_SUCCESS || ret == DM_ERROR_CONNECT);
    return (ret == DM_SUCCESS || ret == DM_ERROR_CONNECT) ? 0 : -1;
}

int test_voltage_all(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetVoltageAllControllers()..." RESET "\n");
    int ret = DM_SetVoltageAllControllers(handle, 50.0f, 120.0f, -20.0f);
    print_test("Set all to 50V", ret == DM_SUCCESS);
    return ret;
}

int test_voltage_all_async(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetVoltageAllControllersAsync()..." RESET "\n");
    
    // Test async - should return immediately
    DM_AsyncTaskId taskId = DM_SetVoltageAllControllersAsync(handle, 60.0f, 120.0f, -20.0f);
    print_test("Async returns task ID (non-zero)", taskId != 0);
    
    if (taskId != 0) {
        // Wait for completion with timeout
        int waitRet = DM_WaitForAsync(handle, taskId, 5000);
        print_test("WaitForAsync completes", waitRet == DM_SUCCESS);
        
        // Check if done
        int done = DM_IsAsyncDone(handle, taskId);
        print_test("IsAsyncDone returns true", done == 1);
        
        // Get result
        int result = DM_GetAsyncResult(handle, taskId);
        print_test("GetAsyncResult succeeds", result == DM_SUCCESS);
    }
    
    return 0;
}

int test_actuator(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetActuatorVoltage()..." RESET "\n");
    int ret = DM_SetActuatorVoltage(handle, 1, 10.0f);
    print_test("Set actuator 1 to 10V", ret == DM_SUCCESS);
    
    ret = DM_SetActuatorVoltage(handle, 1296, 20.0f);
    print_test("Set actuator 1296 to 20V", ret == DM_SUCCESS);
    
    ret = DM_SetActuatorVoltage(handle, 0, 10.0f);
    print_test("Set actuator 0 (invalid) should fail", ret != DM_SUCCESS);
    
    ret = DM_SetActuatorVoltage(handle, 1297, 10.0f);
    print_test("Set actuator 1297 (invalid) should fail", ret != DM_SUCCESS);
    
    return 0;
}

int test_controller(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetControllerVoltage()..." RESET "\n");
    int ret = DM_SetControllerVoltage(handle, 1, 30.0f);
    print_test("Set controller 1 to 30V", ret == DM_SUCCESS);
    
    ret = DM_SetControllerVoltage(handle, 26, 40.0f);
    print_test("Set controller 26 to 40V", ret == DM_SUCCESS);
    
    ret = DM_SetControllerVoltage(handle, 0, 30.0f);
    print_test("Set controller 0 (invalid) should fail", ret != DM_SUCCESS);
    
    ret = DM_SetControllerVoltage(handle, 27, 30.0f);
    print_test("Set controller 27 (invalid) should fail", ret != DM_SUCCESS);
    
    return 0;
}

int test_channel(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetChannelAllControllers()..." RESET "\n");
    int ret = DM_SetChannelAllControllers(handle, 0, 25.0f);
    print_test("Set channel 0 on all to 25V", ret == DM_SUCCESS);
    
    ret = DM_SetChannelAllControllers(handle, 49, 35.0f);
    print_test("Set channel 49 on all to 35V", ret == DM_SUCCESS);
    
    ret = DM_SetChannelAllControllers(handle, -1, 25.0f);
    print_test("Set channel -1 (invalid) should fail", ret != DM_SUCCESS);
    
    ret = DM_SetChannelAllControllers(handle, 50, 25.0f);
    print_test("Set channel 50 (invalid) should fail", ret != DM_SUCCESS);
    
    return 0;
}

int test_relay(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_ControlRelay()..." RESET "\n");
    int ret = DM_OpenRelay(handle);
    print_test("DM_OpenRelay()", ret == DM_SUCCESS);
    
    ret = DM_CloseRelay(handle);
    print_test("DM_CloseRelay()", ret == DM_SUCCESS);
    
    ret = DM_ControlRelay(handle, 1);
    print_test("DM_ControlRelay(1)", ret == DM_SUCCESS);
    
    ret = DM_ControlRelay(handle, 0);
    print_test("DM_ControlRelay(0)", ret == DM_SUCCESS);
    
    return 0;
}

int test_init_actuators(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_InitAllActuators()..." RESET "\n");
    int ret = DM_InitAllActuators(handle);
    print_test("DM_InitAllActuators()", ret == DM_SUCCESS);
    return ret;
}

int test_mapping(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_GetActuatorMapping()..." RESET "\n");
    int ctrl, ch;
    int ret = DM_GetActuatorMapping(handle, 1, &ctrl, &ch);
    print_test("Get mapping for actuator 1", ret == DM_SUCCESS);
    if (ret == DM_SUCCESS) {
        printf("    -> Controller %d, Channel %d\n", ctrl, ch);
    }
    
    ret = DM_GetActuatorMapping(handle, 1296, &ctrl, &ch);
    print_test("Get mapping for actuator 1296", ret == DM_SUCCESS);
    if (ret == DM_SUCCESS) {
        printf("    -> Controller %d, Channel %d\n", ctrl, ch);
    }
    
    return 0;
}

int test_multiple(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetMultipleActuators()..." RESET "\n");
    int actuators[] = {1, 2, 3, 4, 5};
    float voltages[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    int ret = DM_SetMultipleActuators(handle, 5, actuators, voltages);
    print_test("Set multiple actuators", ret == DM_SUCCESS);
    return ret;
}

int test_voltage_array(DM_Handle handle) {
    printf("\n" YELLOW "Testing DM_SetControllerVoltageArray()..." RESET "\n");
    float voltages[50];
    for (int i = 0; i < 50; i++) {
        voltages[i] = (float)(i + 1);
    }
    int ret = DM_SetControllerVoltageArray(handle, 1, voltages);
    print_test("Set controller 1 with voltage array", ret == DM_SUCCESS);
    return ret;
}

int test_create_destroy() {
    printf("\n" YELLOW "Testing DM_Create/DM_Destroy..." RESET "\n");
    
    DM_Handle handle = DM_Create();
    print_test("DM_Create returns valid handle", handle != NULL);
    
    if (handle != NULL) {
        DM_Destroy(handle);
        print_test("DM_Destroy succeeds", 1);
    }
    
    return 0;
}

void print_summary() {
    print_header("Test Summary");
    printf("\n");
    printf(GREEN "Passed: %d\n" RESET, tests_passed);
    printf(RED "Failed: %d\n" RESET, tests_failed);
    printf("\n");
    
    if (tests_failed == 0) {
        printf(GREEN "All tests passed!\n" RESET);
    } else {
        printf(YELLOW "Some tests failed. Check your hardware connection.\n" RESET);
    }
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║         DM Control C++ DLL Test Program           ║\n");
    printf("║      Deformable Mirror Control Library v1.1        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    
    printf("\nLibrary version: %s\n", DM_GetVersion());
    
    /* Parse command line arguments */
    int run_all = 1;
    int interactive = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            interactive = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -i, --interactive    Interactive mode\n");
            printf("  -h, --help           Show this help\n");
            return 0;
        }
    }
    
    print_header("Initialization");
    
    /* Create handle */
    printf("Creating DM handle...\n");
    DM_Handle handle = DM_Create();
    if (handle == NULL) {
        printf(RED "Failed to create handle\n" RESET);
        return 1;
    }
    printf("Handle created successfully.\n");
    
    /* Initialize */
    printf("Connecting to controllers...\n");
    int init_ret = DM_Init(handle, NULL, NULL);
    if (init_ret != DM_SUCCESS) {
        printf(YELLOW "\nWarning: DM_Init returned %d\n", init_ret);
        printf("Controllers may not be reachable. Running tests anyway...\n" RESET);
    } else {
        printf(GREEN "Initialized successfully!\n" RESET);
    }
    
    printf("Connected controllers: %d\n", DM_GetConnectedCount(handle));
    
    if (interactive) {
        /* Interactive mode */
        printf("\n" YELLOW "Interactive Mode - Enter commands:\n" RESET);
        printf("  q - quit\n");
        printf("  0 - set all to 0V\n");
        printf("  a <act> <volt> - set actuator voltage\n");
        printf("  c <ctrl> <volt> - set controller voltage\n");
        printf("  r 0/1 - close/open relay\n");
        printf("  i - initialize\n");
        printf("  d - disconnect\n");
        printf("  as <volt> - async set all\n");
        printf("\n");
        
        char cmd[256];
        while (1) {
            printf("> ");
            if (fgets(cmd, sizeof(cmd), stdin) == NULL) break;
            
            if (cmd[0] == 'q') break;
            
            if (cmd[0] == '0') {
                DM_InitAllActuators(handle);
                printf("All actuators set to 0V\n");
            }
            else if (cmd[0] == 'a') {
                int act, volt;
                if (sscanf(cmd + 2, "%d %d", &act, &volt) == 2) {
                    int ret = DM_SetActuatorVoltage(handle, act, (float)volt);
                    printf("Set actuator %d to %dV: %s\n", act, volt, 
                           ret == DM_SUCCESS ? "OK" : "FAILED");
                }
            }
            else if (cmd[0] == 'c') {
                int ctrl, volt;
                if (sscanf(cmd + 2, "%d %d", &ctrl, &volt) == 2) {
                    int ret = DM_SetControllerVoltage(handle, ctrl, (float)volt);
                    printf("Set controller %d to %dV: %s\n", ctrl, volt,
                           ret == DM_SUCCESS ? "OK" : "FAILED");
                }
            }
            else if (cmd[0] == 'r') {
                int state = cmd[2] - '0';
                int ret = DM_ControlRelay(handle, state);
                printf("Relay %s: %s\n", state ? "open" : "close",
                       ret == DM_SUCCESS ? "OK" : "FAILED");
            }
            else if (cmd[0] == 'i') {
                DM_Init(handle, NULL, NULL);
                printf("Initialized, %d controllers connected\n", DM_GetConnectedCount(handle));
            }
            else if (cmd[0] == 'd') {
                DM_Disconnect(handle);
                printf("Disconnected\n");
            }
            else if (cmd[0] == 'a' && cmd[1] == 's') {
                float volt;
                if (sscanf(cmd + 3, "%f", &volt) == 1) {
                    DM_AsyncTaskId taskId = DM_SetVoltageAllControllersAsync(handle, volt, 120.0f, -20.0f);
                    printf("Async task %llu started, waiting...\n", (unsigned long long)taskId);
                    DM_WaitForAsync(handle, taskId, 5000);
                    printf("Async complete\n");
                }
            }
        }
    } else {
        /* Auto test mode */
        test_create_destroy();
        
        test_init(handle);
        test_voltage_all(handle);
        test_voltage_all_async(handle);
        test_actuator(handle);
        test_controller(handle);
        test_channel(handle);
        test_relay(handle);
        test_init_actuators(handle);
        test_mapping(handle);
        test_multiple(handle);
        test_voltage_array(handle);
        
        print_summary();
    }
    
    printf("\nCleaning up...\n");
    DM_Disconnect(handle);
    DM_Destroy(handle);
    printf("Done.\n");
    
    return tests_failed > 0 ? 1 : 0;
}