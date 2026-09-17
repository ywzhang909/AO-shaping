Doc. #: SS-19-0719-05 

Date: 12, Jul. 2022 

Ver.2.5 

# LCOS-SLM

Liquid Crystal Based Spatial Light Modulator 

## Programmer’s Guide

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/f2d96b3c46189fbe7a7a7ad32d4534addf23e9395baeeab201b9a28c82d3943b.jpg)


This document provides the application programming interface (API) for the SLMFunc.DLL function library. 

## Notes to Users

1) Copyright 2020, Santec Corporation. All rights reserved. No part of this Operation Manual may be reproduced or transmitted in any form or by any means, electronic or mechanical, for any purpose, without the prior written permission of Santec. 

2) Information in this Operation Manual is subject to change without notice. 

3) Information of this Operation Manual is prepared with careful examination, however, in the event of any mistake, please contact us. 

## Notes in Bringing This Product Out of Japan

1) When this product is brought out of Japan, some laws or regulations of a destination country may prohibit this product from being used there. In such countries, the use of this product may lead to punishment. Please note, that in such cases Santec Corporation shall not be held responsible in any way. 

2) When this product is exported (or brought out of Japan), this product is applicable to a strategic material specified in the “Foreign Exchange and Foreign Trade Control Law”, then under law of the Japanese Government, an export permit is required. 

1 Introduction....6
1.1 Description....6
1.2 Function block diagram....6
1.3 Process Flow....7
1.3.1 Set mode, wavelength, phase....7
1.3.2 DVI mode....8
1.3.3 Memory Mode....9
1.4 Attention....10
1.4.1 Display number....10
1.4.2 SLM Number....10
1.4.3 Supported OS....11
1.4.4 Development environment....11
1.4.5 Available DLL functions....12
2 Display Functions....13
2.1 Initializing....13
2.1.1 SLM_Disp_Open....13
2.2 Display....14
2.2.1 SLM_Disp_GrayScale....14
2.2.2 SLM_Disp_BMP....15
2.2.3 SLM_Disp_Data....16
2.2.4 SLM_Disp_ReadBMP....18
2.2.5 SLM_Disp_ReadBMP_A....19
2.2.6 SLM_Disp_ReadCSV....20
2.2.7 SLM_Disp_ReadCSV_A....21
2.3 SLM Finalizing....22
2.3.1 SLM_Disp_Close....22
2.4 Others....23
2.4.1 SLM_Disp_Info....23
2.4.2 SLM_Disp_Info2....24
3 Control Functions....25
3.1 Initializing....25
3.1.1 SLM_Ctrl_Open....25
3.2 Control....26
3.2.1 SLM_Ctrl_ReadSU....26
3.2.2 SLM_Ctrl_WriteVI....27
3.2.3 SLM_Ctrl_ReadVI....28
3.2.4 SLM_Ctrl_WriteWL....29
3.2.5 SLM_Ctrl_ReadWL....30
3.2.6 SLM_Ctrl_WriteAW....31 

3.2.7 SLM_Ctrl_WriteGS 32
3.2.8 SLM_Ctrl_ReadGS 33
3.2.9 SLM_Ctrl_WriteMC 34
3.2.10 SLM_Ctrl_WriteMI 35
3.2.11 SLM_Ctrl_WriteMI_BMP 36
3.2.12 SLM_Ctrl_WriteMI_BMP_A 37
3.2.13 SLM_Ctrl_WriteMI_CSV 38
3.2.14 SLM_Ctrl_WriteMI_CSV_A 39
3.2.15 SLM_Ctrl_WriteME 40
3.2.16 SLM_Ctrl_WriteMT 41
3.2.17 SLM_Ctrl_ReadMS 42
3.2.18 SLM_Ctrl_WriteMR 43
3.2.19 SLM_Ctrl_ReadMR 44
3.2.20 SLM_Ctrl_WriteMP 45
3.2.21 SLM_Ctrl_WriteMZ 46
3.2.22 SLM_Ctrl_WriteMW 47
3.2.23 SLM_Ctrl_ReadMW 48
3.2.24 SLM_Ctrl_WriteDS 49
3.2.25 SLM_Ctrl_ReadDS 50
3.2.26 SLM_Ctrl_WriteDR 51
3.2.27 SLM_Ctrl_WriteDB 52
3.2.28 SLM_Ctrl_WriteTI 53
3.2.29 SLM_Ctrl_ReadTI 54
3.2.30 SLM_Ctrl_WriteTM 55
3.2.31 SLM_Ctrl_ReadTM 56
3.2.32 SLM_Ctrl_WriteTC 57
3.2.33 SLM_Ctrl_ReadTC 58
3.2.34 SLM_Ctrl_WriteTS 59
3.2.35 SLM_Ctrl_ReadT 60
3.2.36 SLM_Ctrl_ReadTD 61
3.2.37 SLM_Ctrl_ReadTO 62
3.2.38 SLM_Ctrl_ReadEDO 63
3.2.39 SLM_Ctrl_ReadED 64
3.2.40 SLM_Ctrl_ReadEO 65
3.2.41 SLM_Ctrl_ReadSDO 66
3.2.42 SLM_Ctrl_ReadSD 67
3.2.43 SLM_Ctrl_ReadSO 68
3.2.44 SLM_Ctrl_WritePN 69
3.2.45 SLM_Ctrl_ReadPN 70
3.2.46 SLM_Ctrl_ReadVR 71 

3.2.47 SLM_Ctrl_ReadPS....72
3.2.48 SLM_Ctrl_ReadLS....73
3.3 Finalizing....74
3.3.1 SLM_Ctrl_Close....74
3.4 Other....75
3.4.1 SLM_Ctrl_Reboot....75
3.5 SLM_STATUS....76
3.6 BMP, CSV, Data Flags....77
3.7 CSV Format....78
3.8 Display table setting....79
4 Samples....80
4.1 VB.net....80
4.1.1 Project Setting....80
4.1.2 Sample source....81
4.2 Python 3.6 Sample source....83
4.3 Other sample source....85
5 Revision History....86
6 Contact....87 

## 1 Introduction

## 1.1 Description

Santec provides DLL application interface for SLM control drivers. 

This document provides application programming interface (API) for function library. 

## 1.2 Function block diagram

This product consists of LCOS unit, Drive board and Option board. The function of each part is as follows. 

<Option board> 

Convert input data from each interface to data format of Drive board. 

<Drive board> 

Display converted data on LCOS unit. 

<LCOS unit> 

Display phase pattern. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/37f9d3f8e14bb4059dead5028d0e8e3003d52101abc468b930b8c8f8073d4221.jpg)



Fig. 1.2-1 Function block diagram


## 1.3 Process Flow

Place the SLMFunc.dll in the same folder as user program, and display data by calling SLM function in user’s program. 

User's Program 

## SLMFunc.dll

## SLM

•Initialize 

•Setting,Display 

•Covert 10bit 

•Finalize 

•Display on SLM 

•Phase Control 

•USB Communications •Trigger In/Out 

•Settings 

## 1.3.1 Set mode, wavelength, phase

SLM has two modes, which can be changed by functions. And, the wavelength and phase need to be set only once. 

Change mode 

Use functions. 

Initializing USB 

･3.1.1 SLM_Ctrl_Open 

SLM_BS 

Read Status 

･3.2.1 SLM_Ctrl_ReadSU 

SLM_OK 

Set Memory/DVI mode 

･3.2.2 SLM_Ctrl_WriteVI 

Set wavelength, phase *1 Save wavelength, phase *1 ･3.2.4 SLM_Ctrl_WriteWL ･3.2.6 SLM_Ctrl_WriteAW 

Finalizing USB 

･3.3.1 SLM_Ctrl_Close 

End 

*1 Once saved, there is no need to set them each time. 

## 1.3.2 DVI mode

In DVI mode, display on LCOS by DVI input. 

Search display number 

Initializing display 

Display phase pattern 

Finalizing display 

Use functions. 

･2.4.1 SLM_Disp_Info 

･2.4.2 SLM_Disp_Info2 

･2.1.1 SLM_Disp_Open 

･2.2.1 SLM_Disp_GrayScale ･2.2.2 SLM_Disp_BMP ･2.2.3 SLM_Disp_Data ･2.2.4 SLM_Disp_ReadBMP ･2.2.5 _A ･2.2.6 SLM_Disp_ReadCSV ･2.2.7 _A 

･2.3.1 SLM_Disp_Close 

## 1.3.3 Memory Mode

In the memory mode, the phase data is transferred to the memory to SLM and displayed on LCOS by specifying the memory number. 

Set Memory mode 

Set wavelength, phase *1 Save wavelength, phase *1 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/5cc824405ec0739a5bbf2fe28cfb467e15fd0073dd4d363a79bcd61aa06dc71d.jpg)


Clear LCOS display *2 

Transfer phase pattern 

Set display order *3 

Display phase pattern 

Use functions. 

･3.1.1 SLM_Ctrl_Open 

･3.2.1 SLM_Ctrl_ReadSU 

･3.2.2 SLM_Ctrl_WriteVI 

･3.2.4 SLM_Ctrl_WriteWL 

･3.2.6 SLM_Ctrl_WriteAW 

･3.2.7 SLM_Ctrl_WriteGS 

･3.2.10 SLM_Ctrl_WriteMI ･3.2.11 SLM_Ctrl_WriteMI_BMP ･3.2.12 SLM_Ctrl_WriteMI_BMP_A ･3.2.13 SLM_Ctrl_WriteMI_CSV ･3.2.14 SLM_Ctrl_WriteMI_CSV_A ･3.2.16 SLM_Ctrl_WriteMT ･3.2.18 SLM_Ctrl_WriteMR ･3.2.20 SLM_Ctrl_WriteMP ･3.2.21 SLM_Ctrl_WriteMZ ･3.2.22 SLM_Ctrl_WriteMW ･3.2.32 SLM_Ctrl_WriteTC Select display ･3.2.24 SLM_Ctrl_WriteDS Continuous display ･3.2.22 SLM_Ctrl_WriteMW ･3.2.26 SLM_Ctrl_WriteDR ･3.2.27 SLM_Ctrl_WriteDB Trigger display ･Trigger Input or 3.2.34 SLM_Ctrl_WriteTS ･3.3.1 SLM_Ctrl_Close 

*1 Once saved, there is no need to set them each time. 

*2. Clear the display because SLM_Ctrl_MI** command cannot be used to write to the displayed Memory number. 

*3 Not required when using SLM_Ctrl_WriteDS function. 

## 1.4 Attention

## 1.4.1 Display number

When SLM is connected to a notebook computer, it is recognized as Display 2. In the case of desktop computer, you need to check what display SLM recognizes. If there are more than two displays, check the SLM's display number with 

” 2.4.2 SLM_Disp_Info2” function. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/643ede456f1d9191a3dff09d3e37f95c1452d4ceea5ae8cb7000c2294f736cfa.jpg)



Figure 1.4-1 Display number


## 1.4.2 SLM Number

SLM Number is automatically allocated by Windows. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/8c8680d61f27c618aca16123cb8c405acdcaa327e9d5c2799a70f8e1c127024b.jpg)


## 1.4.3 Supported OS

Windows 10 

1.4.4 Development environment 

Recommended development environment: 

・Visual Studio

・Python

・MATLAB

・LabVIEW

Select a DLL according to 64-bit and 32-bit development environment. 

¥x64¥SLMFunc.dll 

･･･ 64bit development environment. 

¥x64¥FTD3XX.dll 

･･･ 64bit development environment. 

¥x86¥SLMFunc.dll 

･･･ 32bit development environment. 

¥x86¥FTD3XX.dl 

･･･ 32bit development environment. 

Place the SLMFunc.dll and FTD3XX.dl in the same folder as user program. 

Check that the FT601 recognizes the following when connecting the PC and SLM via USB. 

Device Manager 

File Action View Help 

>System devices 

Universal Serial Bus controllers 

FTDI FT601 USB 3.0 Bridge Device 

Generic USB Hub 

ED Intel(R) USB 3.0 eXtensible Host Controller - 1.0 (Microsoft) 

日 Intel(R) USB 3.1 eXtensible Host Controller - 1.10 (Microsoft)

D Realtek USB 3.0 Card Reader 

USB Composite Device 

USB Composite Device 

USB Composite Device 

If the FT601 does not recognize, turn off the SLM or refer to “USB driver installation procedure” section in the “SLM-200 OPERATIONAL MANUAL”. 

## 1.4.5 Available DLL functions


Table 1.4-1 Available DLL functions


<table><tr><td></td><td>Functions</td><td>GUI Software</td><td>DLL Functions</td></tr><tr><td rowspan="3">DVI I/F</td><td>Display CSV or BMP file</td><td>√</td><td>√</td></tr><tr><td>Display array data</td><td></td><td>√</td></tr><tr><td>Full screen contrast</td><td>√</td><td>√</td></tr><tr><td rowspan="5">USB I/F</td><td>Set wavelength</td><td>√</td><td>√</td></tr><tr><td>Continuous display</td><td>√</td><td>√</td></tr><tr><td>Pattern capture</td><td>√</td><td>√</td></tr><tr><td>Set trigger</td><td>√</td><td>√</td></tr><tr><td>Display mode select</td><td>√</td><td>√</td></tr><tr><td>Other</td><td>CGH generator</td><td>√</td><td></td></tr></table>

## 2 Display Functions

## 2.1 Initializing

2.1.1 SLM_Disp_Open 

```cmake
SLM_STATUS
SLM_Disp_Open(
DWORD DisplayNumber
) 
```

## Summary

SLM display initializing. 

## Parameters

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
Example
    // Open Display2
    if(SLM_Disp_Open(2) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

## 2.2 Display

```sql
SLM_STATUS
SLM_Disp_GrayScale(
DWORD DisplayNumber,
DWORD Flags,
USHORT GrayScale
) 
```

## Summary

Drawing the entire display with GrayScale input. 

## Parameters

```txt
DisplayNumber: Specify display number (1, 2, 3...).
Flags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
GrayScale: Specify grayscale from 0 to 1023 (0π - 2 π). 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```cpp
example
    // Display all pixel 512
    if(SLM_Disp_GrayScale(2,0,512) == SLM_OK){
    // OK
    }

    // Display grayscale on the 120Hz SLM.
    if(SLM_Disp_GrayScale(2, FLAG_RATE120,512) == SLM_OK){
    // OK
    } 
```

```txt
2.2.2 SLM_Disp_BMP
SLM_STATUS
SLM_Disp_BMP(
DWORD DisplayNumber,
DWORD Flags,
HBITMAP bmp
) 
```

```txt
DisplayNumber: Specify display number (1, 2, 3...).
Flags: This parameter is for future option (default value 0). 
```

## Summary

Display BMP on the SLM. 

## Parameters

Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”) If you use 120Hz model, use FLAGS_RATE120. bmp: Pointer to bmp. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```c
// Display bmp on the SLM.
HBITMAP hbmp;
hbmp=(HBITMAP)LoadImage(hinst,_T("c:¥¥test.bmp"),
IMAGE_BITMAP,0,0,LR_CREATEDIBSECTION | LR_LOADFROMFILE);
if(SLM_Disp_BMP(2,0,hbmp) == SLM_OK){
    // OK
} 
```

```txt
2.2.3 SLM_Disp_Data
SLM_STATUS
SLM_Disp_Data(
DWORD DisplayNumber,
USHORT width,
USHORT height,
DWORD Flags,
USHORT* data
) 
```

## Summary

Display array data on the SLM. 

```txt
parameters
DisplayNumber: Specify display number (1, 2, 3...).
width: Specify display width value.
height: Specify display height value.
Flags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
data: Pointer to array of unsigned short data.(width * height * 2byte) 
```

## Return Value

```txt
SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to "3.5 SLM_STATUS") 
```

```c
Example
// Display array data on the SLM.
USHORT *dat, *pos;
dat = pos = (USHORT*)malloc(sizeof(short) * 1920 * 1200);
for (int y = 0; y < 1200; y++) {
    for (int x = 0; x < 1920; x++) {
    *pos = (USHORT)(rand() * 1023); // All data random
    pos++;
    }
}
if (SLM_Disp_Data(1, 1920, 1200, 0, dat) == SLM_OK) {
    // OK
}

// Display bmp on the 120Hz SLM.
if (SLM_Disp_Data(1, 1920, 1200, FLAG_RATE120, dat) == SLM_OK) {
    // OK 
```

<table><tr><td>Functions</td><td>santec</td></tr><tr><td>}</td><td></td></tr></table>

## 2.2.4 SLM_Disp_ReadBMP

```txt
SLM_STATUS
SLM_Disp_ReadBMP(
DWORD DisplayNumber,
DWORD BMPFlags,
LPCWSTR FileName
) 
```

## Summary

Display bmpfile (Unicode) data on the SLM. 

## Parameters

```txt
DisplayNumber: Specify display number (1, 2, 3...).
BMPFlags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
FileName: Pointer to buffer containing Unicode bmpfile name. 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // bmp file
    if (SLM_Disp_ReadBMP(2, 0, _T("C:¥¥test.bmp")) == SLM_OK) {
    // OK
    }

    // Display bmp on the 120Hz SLM.
    if (SLM_Disp_ReadBMP(2, FLAG_RATE120, _T("C:¥¥test.bmp")) == SLM_OK) {
    // OK
    } 
```

```txt
SLM_STATUS
SLM_Disp_ReadBMP_A(
DWORD DisplayNumber,
DWORD BMPFlags,
LPCSTR FileName
) 
```

## 2.2.5 SLM_Disp_ReadBMP_A

## Summary

Display bmpfile(ANSI code) data on the SLM. 

## Parameters

```txt
DisplayNumber: Specify display number (1, 2, 3...).
BMPFlags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
FileName: Pointer to buffer containing ANSI code bmpfile name. 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // bmp file
    if (SLM_Disp_ReadBMP_A(2, 0, ("C:¥¥test.bmp")) == SLM_OK) {
    // OK
    }

    // Display bmp on the 120Hz SLM.
    if (SLM_Disp_ReadBMP_A(2, FLAG_RATE120, ("C:¥¥test.bmp")) == SLM_OK) {
    // OK
    } 
```

## 2.2.6 SLM_Disp_ReadCSV

```txt
SLM_STATUS
SLM_Disp_ReadCSV(
DWORD DisplayNumber,
DWORD CSVFlags,
LPCWSTR FileName
) 
```

## Summary

Display csvfile(Unicode) data on the SLM. 

## Parameters

```txt
DisplayNumber: Specify display number (1, 2, 3...).
CSVFlags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
FileName: Pointer to buffer containing Unicode csvfile name.
(Refer to “CSV Format3.7 CSV Format”) 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // CSV file
    if (SLM_Disp_ReadCSV(2, 0, _T("C:¥¥test.csv")) == SLM_OK) {
    // OK
    }

    // Display csv on the 120Hz SLM.
    if (SLM_Disp_ReadCSV(2, FLAG_RATE120, _T("C:¥¥test.csv")) == SLM_OK) {
    // OK
    } 
```

```txt
2.2.1 SLM_Disp_ReadCSV_A
SLM_STATUS
SLM_Disp_ReadCSV_A(
DWORD DisplayNumber,
DWORD CSVFlags,
LPCSTR FileName
) 
```

## 2.2.7 SLM_Disp_ReadCSV_A

## Summary

Display csvfile(ANSI code) data on the SLM. 

## Parameters

```txt
DisplayNumber: Specify display number (1, 2, 3...).
CSVFlags: Use this to change the display method. (Refer to “3.6 BMP, CSV, Data Flags”)
If you use 120Hz model, use FLAGS_RATE120.
FileName: Pointer to buffer containing ANSI code csvfile name.
(Refer to “CSV Format3.7 CSV Format”) 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // csv file
    if (SLM_Disp_ReadCSV_A(2, 0, ("C:¥¥test.csv")) == SLM_OK) {
    // OK
    }

    // Display csv on the 120Hz SLM.
    if (SLM_Disp_ReadCSV_A(2, FLAG_RATE120, ("C:¥¥test.csv")) == SLM_OK) {
    // OK
    } 
```

```txt
SLM_STATUS
SLM_Disp_Close(
DWORD DisplayNumber 
```

```txt
Example
    // Close Display2
    if(SLM_Disp_Close(2) == SLM_OK){
    // OK
    } 
```

## 2.3 SLM Finalizing

## Summary

SLM display finalizing. 

## Parameters

DisplayNumber: 

Specify display number (1, 2, 3…). 

## Return Value

## 2.4 Others

2.4.1 SLM_Disp_Info 

```txt
SLM_STATUS
SLM_Disp_Info(
DWORD DisplayNumber,
USHORT *width,
USHORT *height
) 
```

## Summary

Read width and height of the display. 

## Parameters

width: 

height: 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Display2 Information
UShort width, height;
if(SLM_Disp_Info(2, &width, &height) == SLM_OK){
    // OK
    // width = 1920, height = 1200
} 
```

```txt
2.4.2 SLM_Disp_Info2
SLM_STATUS
SLM_Disp_Info(
DWORD DisplayNumber,
USHORT *width,
USHORT *height,
LPSTR DisplayName
) 
```

## Summary

Read width and height, DisplayName of display. 

## Parameters

```txt
parameters
DisplayNumber: Specify display number (1, 2, 3...).
width: Pointer to unsigned short to store width value.
height: Pointer to unsigned short to store height value.
DisplayName Pointer to a 128-byte buffer to store DisplayName.
DisplayName format is “UserFriendlyName, ManufactreName, ProductCodeID, SerialNumberID”
e.g. DisplayName = “LCOS-SLM, SOC, 8001, 2018021001” 
```

## Return Value

```txt
SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to "3.5 SLM_STATUS") 
```

## Example

```txt
// Display2 Information
UShort width, height;
char DisplayName[128];

if(SLM_Disp_Info2(1, &width, &height, DisplayName) == SLM_OK){
    // OK
    // width = 1920, height = 1080
    // DisplayName = "ABC Z246,ABC,0001,ABC0123456789"
}    if(SLM_Disp_Info2(2, &width, &height, DisplayName) == SLM_OK){
    // OK
    // width = 1920, height = 1200
    // DisplayName = "LCOS-SLM,SOC,8001,2018021001"
} 
```

## 3 Control Functions

## 3.1 Initializing

3.1.1 SLM_Ctrl_Open 

```txt
SLM_STATUS
SLM_Ctrl_Open (
DWORD SLMNumber,
) 
```

## Summary

Open USB interface. 

## Parameters

SLMNumber: Specify SLM number (1-8). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```rust
// Open USB interface
if(SLM_Ctrl_Open(1)==SLM_OK){
    // OK
} 
```

```txt
SLM_STATUS
SLM_Ctrl_ReadSU(
DWORD SLMNumber 
```

## 3.2 Control

## Summary

Read status of SLM. Busy or Ready. 

## Parameters

SLMNumber: Specify SLM number (1, 2, 3…). 

## Return Value

## Notes

It will be in BUSY state for about 40 seconds for power ON and phase table expand. 

```c
Example
// Reads status
SLM_STATUS ret;

for(int i = 0; i < 100; i++) {
    Sleep(1000);
    ret = SLM_Ctrl_ReadSU(1);
    if(ret == SLM_OK) return true;
    else if(ret == SLM_BS) continue;
    else return false; // error
}

return false; // timeout 
```

## 3.2.2 SLM_Ctrl_WriteVI

SLM_STATUS SLM_Ctrl_WriteVI( DWORD SLMNumber, DWORD mode 

## Summary

Write video mode DVI or Memory mode. 

## Parameters

SLMNumber: Specify SLM number (1-8). mode: Specify mode value. 0:Memory mode, 1:DVI mode, Default 1 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Notes

It takes 40 seconds to respond to expand phase table. 

## Example

// Set video mode 0 if(SLM_Ctrl_WriteVI (1,0) == SLM_OK){ // OK } 

```txt
3.2.3 SLM_Ctrl_ReadVI
SLM_STATUS
SLM_Ctrl_ReadVI(
DWORD SLMNumber,
DWORD *mode
) 
```

Summary Read display mode DVI or Memory mode. 

## Parameters

SLMNumber: Specify SLM number (1-8). mode: Pointer to unsigned int(32bit) to store mode value. 0: Memory mode, 1: DVI mode 

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // Read display mode
    DWORD mode;
    if(SLM_Ctrl_ReadVI(1,&mode) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

```txt
SLM_STATUS
SLM_Ctrl_WriteWL(
DWORD SLMNumber,
DWORD wavelength,
DWORD phase
) 
```

## 3.2.4 SLM_Ctrl_WriteWL

## Summary

Write wavelength and phase value. It cannot be set to a value that causes internal calculation result of SLM to be abnormal. e.g. Set phase 2.00 => calculation result 2.01 

## Parameters

SLMNumber: Specify SLM number (1-8). wavelength: Specify wavelength value.(e.g. 1500) phase: Specify phase value multiplied by 100 (0-999). e.g. 2.00 => 200. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Notes

It takes 40 seconds to respond to expand phase table. 

```txt
Example
// Set wavelength (1500nm) and phase (2π)
if(SLM_Ctrl_WriteWL(1,1500,200)==SLM_OK){
    // OK
} 
```

```txt
3.2.5 SLM_Ctrl_ReadWL 
```

```sql
SLM_STATUS
SLM_Ctrl_ReadWL(
DWORD SLMNumber,
DWORD *wavelength,
DWORD *phase
) 
```

## Summary

Read wavelength and phase value. 

## Parameters

```txt
SLMNumber: Specify SLM number (1,2,3...8).
wavelength: Pointer to unsigned int(32bit) to store wavelength value.(450-1600)
phase: Pointer to unsigned int(32bit) to store phase value multiplied by 100.
(0-999) 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read display mode
DWORD wavelength, phase;
if(SLM_Ctrl_ReadWL(1,&wavelength,&phase) == SLM_OK){
    // OK
    printf("%d nm, %0.2f pai", wavelength, ((float)phase)/100);
}
else{
    // Error
} 
```

## 3.2.6 SLM_Ctrl_WriteAW

## Summary

Save wavelength and phase settings. The settings are retained even when power is turned off. 

Parameters SLMNumber: Specify SLM number (1-8). 

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

// Set video mode 0 SLM_Ctrl_WriteWL(1,1500,200); 

```txt
3.2.7 SLM_Ctrl_WriteGS
SLM_STATUS
SLM_Ctrl_WriteGS(
DWORD SLMNumber,
USHORT GrayScale
) 
```

Summary Display specified grayscale on the entire display. 

Parameters SLMNumber: Specify SLM number (1, 2, 3…). GrayScale: Specify grayscale from 0 to 1023 (0π - 2 π) 

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // entire display 1023
    if(SLM_Ctrl_WriteGS(1,1023) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

```cmake
3.2.8 SLM_Ctrl_ReadGS
SLM_STATUS
SLM_Ctrl_ReadGS(
DWORD SLMNumber,
USHORT *GrayScale
) 
```

Summary Read grayscale on display. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
GrayScale: Specify grayscale from 0 to 1023 (0π - 2 π). 
```

```txt
Return Value
SLM_OK if successful, otherwise SLM_STATUS error code is returned.
(Refer to “3.5 SLM_STATUS”) 
```

```txt
Example
    // read grayscale
    USHORT gray-scale;
    if(SLM_Ctrl_ReadGS(1,&grayscale) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

## 3.2.9 SLM_Ctrl_WriteMC

## Summary

Transfer phase pattern input from the DVI input to internal memory. 

## Parameters

SLMNumber: Specify SLM number (1-8). MemoryNumber: Specify Memory number (1-128). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

// DVI input to internal memory 10. if(SLM_Ctrl_WriteMC (1,10) == SLM_OK){ // OK } 

```txt
3.2.10 SLM_Ctrl_WriteMI 
```

```txt
SLM_STATUS
SLM_Ctrl_WriteMI(
DWORD SLMNumber,
DWORD MemoryNumber,
USHORT width,
USHORT height,
DWORD Flags,
USHORT* data
) 
```

## Summary

Transfer array data to SLM memory. 

## Parameters

Flags: This parameter is for future option (default value 0). 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. 

```txt
(Refer to "3.5 SLM_STATUS") 
```

## Note

Since memory number displayed with SLM_Ctrl_WriteDS function cannot be overwritten, you need to change displayed memory number or change the displayed content with SLM_Ctrl_WriteGS function to write. 

## Example

```c
// write array data to memory number 1
USHORT *dat, *pos;
dat = pos = (USHORT*)malloc(sizeof(short) * 1920 * 1200);
for (int y = 0; y < 1200; y++) {
    for (int x = 0; x < 1920; x++) {
    *pos = (USHORT)(rand() * 1023); // All data random
    pos++;
    }
}
if (SLM_Ctrl_WriteMI(1, 1, 1920, 1200, 0, dat) == SLM_OK) { // OK} 
```

```txt
3.2.11 SLM_Ctrl_WriteMI_BMP 
```

```sql
SLM_STATUS
SLM_Ctrl_WriteMI_BMP(
DWORD SLMNumber,
DWORD MemoryNumber,
DWORD Flags,
LPCWSTR FileName
) 
```

## Summary

Transfer BMP file(Unicode) to SLM memory. 

## Parameters

DisplayNumber: Specify display number (1, 2, 3…). 

MemoryNumber: Specify memory number(1-128). 

BMPFlags: Specify color mode. See “3.6 BMP, CSV, Data Flags” 

FileName: Pointer to buffer containing Unicode bmpfile name. 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Note

Since memory number displayed with SLM_Ctrl_WriteDS function cannot be overwritten, you need to change displayed memory number or change displayed content with SLM_Ctrl_WriteGS function to write. 

```txt
Example
// bmp file to memory number 1
if (SLM_Ctrl_WriteMI_BMP(1,1,0,_T("C:¥¥test.bmp")) == SLM_OK) {
    // OK
} 
```

```txt
DisplayNumber: Specify display number (1, 2, 3...).
MemoryNumber: Specify memory number(1-128).
BMPFlags: Specify color mode. See “3.6 BMP, CSV, Data Flags”
FileName: Pointer to buffer containing Unicode bmpfile name. 
```

```txt
3.2.12 SLM_Ctrl_WriteMI_BMP_A 
```

```sql
SLM_STATUS
SLM_Ctrl_WriteMI_BMP_A(
DWORD SLMNumber,
DWORD MemoryNumber,
DWORD Flags,
LPCSTR FileName
) 
```

## Summary

Transfer BMP file(Unicode) to SLM memory. 

## Parameters

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Note

Since memory number displayed with SLM_Ctrl_WriteDS function cannot be overwritten, you need to change displayed memory number or change displayed content with SLM_Ctrl_WriteGS function to write. 

```cpp
Example
// bmp file to memory number 1
if (SLM_Ctrl_WriteMI_BMP_A(1,1,0, "C:¥¥test.bmp") == SLM_OK) {
    // OK
} 
```

```txt
3.2.13 SLM_Ctrl_WriteMI_CSV 
```

```sql
SLM_STATUS
SLM_Ctrl_WriteMI_CSV(
DWORD SLMNumber,
DWORD MemoryNumber,
DWORD CSVFlags,
LPCWSTR FileName
) 
```

## Summary

Transfer CSV file(Unicode) to SLM memory. 

## Parameters

DisplayNumber: Specify display number (1, 2, 3…). MemoryNumber: Specify memory number(1-128). CSVFlags: This parameter is for future option (default value 0). FileName: Pointer to buffer containing Unicode csvfile name. (Refer to “CSV Format3.7 CSV Format”) 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Note

Since memory number displayed with SLM_Ctrl_WriteDS function cannot be overwritten, you need to change displayed memory number or change displayed content with SLM_Ctrl_WriteGS function to write. 

```c
Example
// csv file to memory number 1
if (SLM_Ctrl_WriteMI_CSV(1,1,0,_T("C:¥¥test.csv")) == SLM_OK) {
    // OK
} 
```

```txt
3.2.14 SLM_Ctrl_WriteMI_CSV_A 
```

```sql
SLM_STATUS
SLM_Ctrl_WriteMI_CSV_A(
DWORD SLMNumber,
DWORD MemoryNumber,
DWORD CSVFlags,
LPCSTR FileName
) 
```

## Summary

Transfer CSV file(ANSI code) to SLM memory. 

## Parameters

DisplayNumber: Specify display number (1, 2, 3…). MemoryNumber: Specify memory number (1-128). CSVFlags: This parameter is for future option (default value 0). FileName: Pointer to buffer containing ANSI code csvfile name. (Refer to “CSV Format3.7 CSV Format”) 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Note

Since memory number displayed with SLM_Ctrl_WriteDS function cannot be overwritten, you need to change displayed memory number or change displayed content with SLM_Ctrl_WriteGS function to write. 

```c
Example
// csv file to memory number 1
if (SLM_Ctrl_WriteMI_CSV_A(1,1,0, "C:¥¥test.csv") == SLM_OK) {
    // OK
} 
```

## 3.2.15 SLM_Ctrl_WriteME

## Summary

Invalidates phase pattern stored in internal memory. 

## Parameters

SLMNumber: Specify SLM number (1-8). MemoryNumber: Specify memory number (1-128). 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

// Invalidates phase pattern stored in internal memory. if(SLM_Ctrl_WriteME (1,1) == SLM_OK){ // OK } 

## Summary

Replace memory number set in display table. 

## Parameters

For detailed information about table number and memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```rust
// replace memory number
if(SLM_Ctrl_WriteMT(1,1,2)==SLM_OK){
    // OK
} 
```

## 3.2.17 SLM_Ctrl_ReadMS

## Summary

Read memory number set in display table. 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
TableNumber: Specify table number (1-128).
MemoryNumber: Pointer to unsigned int(32bit) to store memory number value. 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read memory mode
DWORD memorynumber;
if(SLM_Ctrl_ReadMS(1,1,& memorynumber) == SLM_OK){
    // OK
}
else{
    // Error
} 
```

## 3.2.18 SLM_Ctrl_WriteMR

## Summary

Write effective range of display table. 

## Parameters

TableNumber1: 

TableNumber2: 

For detailed information about table number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// effective range
if(SLM_Ctrl_WriteMR(1,1,128)==SLM_OK){
    // OK
} 
```

```txt
3.2.19 SLM_Ctrl_ReadMR 
```

```sql
SLM_STATUS
SLM_Ctrl_ReadMR(
DWORD SLMNumber,
DWORD *TableNumber1,
DWORD *TableNumber2
) 
```

## Summary

Read effective range of display table. 

## Parameters

```txt
SLMNumber: Specify SLM number (1-8).
TableNumber1: Pointer to unsigned int(32bit) to store table number value.
TableNumber2: Pointer to unsigned int(32bit) to store table number value. 
```

For detailed information about table number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// effective range
DWORD st, ed;
if(SLM_Ctrl_ReadMR(1,&st,&ed)==SLM_OK){
    // OK
} 
```

```txt
SLMNumber: Specify SLM number (1-8).
TableNumber: Specify table number (1-128). 
```

## 3.2.20 SLM_Ctrl_WriteMP

## Summary

Write table number of display table to be displayed first. 

## Parameters

For detailed information about table number refer to “3.8 Display table setting”. 

## Return Value

## Example

```txt
// first display table
if(SLM_Ctrl_WriteMP(1,1)==SLM_OK){
    // OK
} 
```

## 3.2.21 SLM_Ctrl_WriteMZ

## Summary

Set contents of display table to default settings. 

## Parameters

SLMNumber: 

For detailed information about table number refer to “3.8 Display table setting”. 

## Return Value

## Example

```txt
// table default setting
if(SLM_Ctrl_WriteMZ(1)==SLM_OK){
    // OK
} 
```

## 3.2.22 SLM_Ctrl_WriteMW

SLM_STATUS SLM_Ctrl_WriteMW( DWORD SLMNumber, DWORD frames 

## Summary

Write interval for switching pattern display by number of frames. Setting is specified by number of frames. 

## Parameters

SLMNumber: 

frames: 

Specify SLM number (1-8). Specify frames value (0-120). e.g. 16.7ms per frame if SLM frame rate is 60Hz. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
// 1s interval
if(SLM_Ctrl_WriteMW(1,60) == SLM_OK){
    // OK
} 
```

```cmake
3.2.23 SLM_Ctrl_ReadMW
SLM_STATUS
SLM_Ctrl_ReadMW(
DWORD SLMNumber,
DWORD *frames
) 
```

Summary Read interval for switching pattern display by number of frames. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
frames: Pointer to unsigned int(32bit) to store frames value.(0-120) 
```

```txt
Return Value
SLM_OK if successful, otherwise SLM_STATUS error code is returned.
(Refer to "3.5 SLM_STATUS") 
```

```txt
Example
    // frames
    DWORD frames;
    if(SLM_Ctrl_ReadMW(1,&frames) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

## 3.2.24 SLM_Ctrl_WriteDS

## Summary

Specify memory number to display internal memory phase pattern. 

## Parameters

SLMNumber: Specify SLM number (1, 2, 3…). MemoryNumber: Specify memory number (1-128). 

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// display memory number1
if(SLM_Ctrl_WriteDS(1,1) == SLM_OK){
    // OK
}
else{
    // Error
} 
```

```txt
3.2.25 SLM_Ctrl_ReadDS
SLM_STATUS
SLM_Ctrl_ReadDS(
DWORD SLMNumber,
DWORD *MemoryNumber
) 
```

## Summary

Read displayed memory number. 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
MemoryNumber: Pointer to unsigned int(32bit) to store memory number value. 
```

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read display memory number
DWORD MemoryNumber;
if(SLM_Ctrl_ReadDS(1,&MemoryNumber) == SLM_OK){
    // OK
}
else{
    // Error
} 
```

```txt
3.2.26 SLM_Ctrl_WriteDR
SLM_STATUS
SLM_Ctrl_WriteDR(
DWORD SLMNumber,
DWORD order
) 
```

## Summary

Display phase patterns stored in internal memory in order of display table. Display order, range, and start position follow display table settings. Continuous display continues until stopped by SLM_Ctrl_WriteDB function. Some communication commands are invalid during continuous display. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
order: Specify order value (0-1).
0: Descending order, 1: Ascending order 
```

For detailed information about memory number refer to “3.8 Display table setting”. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// continuous display
if(SLM_Ctrl_WriteDR(1,1) == SLM_OK){
    // OK
}
else{
    // Error
}
Sleep(1000);

// stop
if(SLM_Ctrl_WriteDB(1) == SLM_OK){
    // OK
}
else{
    // Error
} 
```

```txt
3.2.27 SLM_Ctrl_WriteDB
SLM_STATUS
SLM_Ctrl_WriteDB(
DWORD SLMNumber
) 
```

```cpp
example
    // continuous display
    if(SLM_Ctrl_WriteDR(1,1) == SLM_OK){
    // OK
    }
    else{
    // Error
    }
    Sleep(1000);

    // stop
    if(SLM_Ctrl_WriteDB(1) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

```txt
// trigger input on
if(SLM_Ctrl_WriteTI(1,0)==SLM_OK){
    // OK
} 
```

## 3.2.28 SLM_Ctrl_WriteTI

## Summary

Write ON / OFF of trigger input value. 

## Parameters

SLMNumber: Specify SLM number (1-8). onoff : Specify onoff value(0:off,1:on). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
3.2.29 SLM_Ctrl_ReadTI
SLM_STATUS
SLM_Ctrl_ReadTI(
DWORD SLMNumber,
DWORD *onoff
) 
```

Summary Read ON / OFF of trigger input value. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
onoff: Pointer to unsigned int(32bit) to store mode value(0:off,1:on). 
```

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // read trigger input
    DWORD onoff;
    if(SLM_Ctrl_ReadTI(1,&onoff)==SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

```rust
// trigger output on
if(SLM_Ctrl_WriteTM(1,1)==SLM_OK){
    // OK
} 
```

## 3.2.30 SLM_Ctrl_WriteTM

## Summary

Write ON / OFF of trigger output value. 

## Parameters

SLMNumber: Specify SLM number (1-8). onoff : Specify onoff value(0:off,1:on). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
3.2.31 SLM_Ctrl_ReadTM
SLM_STATUS
SLM_Ctrl_ReadTM(
DWORD SLMNumber,
DWORD *onoff
) 
```

Summary Read ON / OFF of trigger output value. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
onoff: Pointer to unsigned int(32bit) to store mode value(0:off, 1:on). 
```

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```cpp
Example
    // read trigger output
    DWORD onoff;
    if(SLM_Ctrl_ReadTM(1,&onoff)==SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

## Summary

Write ascending / descending order of pattern display by trigger input. 

## Parameters

SLMNumber: Specify SLM number (1-8). 

order : Specify order value(0-1). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// trigger display order
if(SLM_Ctrl_WriteTC(1,0)==SLM_OK){
    // OK
} 
```

```txt
3.2.33 SLM_Ctrl_ReadTC
SLM_STATUS
SLM_Ctrl_ReadTC(
DWORD SLMNumber,
DWORD *order
) 
```

Summary Read ascending / descending order of pattern display by trigger input. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
order: Pointer to unsigned int(32bit) to store order value.
0: Descending order, 1: Ascending order 
```

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // Read trigger order
    DWORD order;
    if(SLM_Ctrl_ReadTC(1,&order) == SLM_OK){
    // OK
    }
    else{
    // Error
    } 
```

## 3.2.34 SLM_Ctrl_WriteTS

## Summary

Performs same operation as trigger input. 

## Parameters

SLMNumber: Specify SLM number (1-8). 

## Return Value

## Example

## 3.2.35 SLM_Ctrl_ReadT

```sql
SLM_STATUS
SLM_Ctrl_ReadT(
DWORD SLMNumber,
INT32 *driveboardTemp,
INT32 *optionboardTemp
) 
```

## Summary

Read drive board and option board Celsius temperatures. 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
driveboardTemp: Pointer to int(32bit) to store driveboardTemp value multiplied by 10.
optionboardTemp: Pointer to int(32bit) to store optionboardTemp value multiplied by 10. 
```

## Return Value

## Example

```txt
// Read drive and option board temperatures
int dTemp,oTemp;
if(SLM_Ctrl_ReadT(1,&dTemp,&oTemp) == SLM_OK){
    // OK
    printf("Drive Board %0.1f degrees, Option Board %0.1f degrees", ((float) dTemp)/10, ((float) oTemp)/10);
}
else{
    // Error
} 
```

## 3.2.36 SLM_Ctrl_ReadTD

## Summary

Read drive board Celsius temperature. 

## Parameters

## Return Value

## Example

```txt
// Read option board temperature
int dTemp;
if(SLM_Ctrl_ReadTD(1,&oTemp) == SLM_OK){
    // OK
    printf("Drive Board %0.1f degrees, Option Board %0.1f degrees", ((float) dTemp)/10);
}
else{
    // Error
} 
```

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
optionboardTemp: Pointer to int(32bit) to store optionboardTemp value multiplied by 10. 
```

## 3.2.37 SLM_Ctrl_ReadTO

## Summary

Read option board Celsius temperature. 

## Parameters

## Return Value

## Example

```txt
// Read option board temperature
int oTemp;
if(SLM_Ctrl_ReadTO(1,&oTemp) == SLM_OK){
    // OK
    printf("Drive Board %0.1f degrees, Option Board %0.1f degrees", ((float) oTemp)/10);
}
else{
    // Error
} 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/889b0dca2833783383092849a44dd04f587ea2d4ce7bca42e9d2cc061c70978a.jpg)



Fig. 3.2-1 drive board error



Fig. 3.2-2 option board error


```txt
Return Value
SLM_OK if successful, otherwise SLM_STATUS error code is returned.
(Refer to "3.5 SLM_STATUS")

Example
    // Read error
    DWORD driveerr, optionerr;
    if(SLM_Ctrl_ReadEDO(1,&driveerr,&optionerr) == SLM_OK){
    // OK
    printf("Drive Board Error %4X, Option Board Error %4X¥n",driveerr, optionerr);
    }
    else{
    // Error
    } 
```

```csv
3.2.39 SLM_Ctrl_ReadED
SLM_STATUS
SLM_Ctrl_ReadEDO(
DWORD SLMNumber,
DWORD *driveboardError
) 
```

## Summary

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
driveboardError: Pointer to unsigned int(32bit) to store driveboard error value. 
```

<table><tr><td>7</td><td>6</td><td>5</td><td>4</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>(4)</td><td>(3)</td><td>(2)</td><td>(1)</td></tr></table>


Fig. 3.2-3 drive board error


## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // Read error
    DWORD driveerr;
    if(SLM_Ctrl_ReadED(1,&driveerr) == SLM_OK){
    // OK
    printf("Drive Board Error %4X¥n",driveerr);
    }
    else{
    // Error
    } 
```

```sql
3.2.40 SLM_Ctrl_ReadEO
SLM_STATUS
SLM_Ctrl_ReadEDO(
DWORD SLMNumber,
DWORD *optionboardError
) 
```

Summary Read error flags of "Option board". These error values are output in hexadecimal. 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
optionboardError: Pointer to unsigned int(32bit) to store optionboard error value. 
```

<table><tr><td>7</td><td>6</td><td>5</td><td>4</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>(4)</td><td>(3)</td><td>(2)</td><td>(1)</td></tr></table>


Fig. 3.2-4 option board error


## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

```txt
Example
    // Read error
    DWORD optionerr;
    if(SLM_Ctrl_ReadEO(1,&optionerr) == SLM_OK){
    // OK
    printf("Option Board Error %4X¥n", optionerr);
    }
    else{
    // Error
    } 
```

## Summary

Read identification numbers of "Drive board" and "Option board". 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
driveboardID: Pointer to a 16-byte buffer to store driveboardID
optionboardID: Pointer to a 16-byte buffer to store optionboardID 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read drive and option board id
char driveboardID[16];
char optionboardID[16];
if(SLM_Ctrl_ReadSDO(1,driveboardID, optionboardID) == SLM_OK){
    // OK
    // driveboardID => "18050004"
    // optionboardID => "18050004"
}
else{
    // Error
} 
```

## 3.2.42 SLM_Ctrl_ReadSD

Summary Read identification numbers of "Drive board". 

```txt
Parameters
SLMNumber: Specify SLM number (1, 2, 3...).
driveboardID: Pointer to a 16-byte buffer to store driveboardID 
```

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read drive board id
char driveboardID[16];
if(SLM_Ctrl_ReadSD(1,driveboardID) == SLM_OK){
    // OK
    // driveboardID => "18050004"
}
else{
    // Error
} 
```

```txt
3.2.43 SLM_Ctrl_ReadSO
SLM_STATUS
SLM_Ctrl_ReadSDO(
DWORD SLMNumber,
LPSTR optionboardID
) 
```

## Summary

Read identification numbers of "Option board". 

## Parameters

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read option board id
char optionboardID[16];
if(SLM_Ctrl_ReadSO(1, optionboardID) == SLM_OK){
    // OK
    // optionboardID => "18050004"
}
else{
    // Error 
```

## 3.2.44 SLM_Ctrl_WritePN

## Summary

Write Max 13-digit Display Name. This name is the same information as SLM_Disp_Info2 and is the EDID information of the display. 

## Parameters

SLMNumber: Specify SLM number (1, 2, 3…). DisplayName: Pointer to buffer containing DisplayName. 

## Return Value Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

// Write Display Name if(SLM_Ctrl_WritePN(1 , “SLM-LCOS_001”) == SLM_OK){ // OK } else{ // Error } 

```cmake
3.2.45 SLM_Ctrl_ReadPN
SLM_STATUS
SLM_Ctrl_ReadPN(
DWORD SLMNumber,
LPSTR DisplayName
) 
```

## Summary

Read Max 13-digit Display Name. This name is the same information as SLM_Disp_Info2 and is the EDID information of the display. 

Parameters SLMNumber: Specify SLM number (1, 2, 3…). DisplayName: Pointer to a 16-byte buffer to store DisplayName. 

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read Display Name
char DisplayName [16];
if(SLM_Ctrl_ReadLS(1, 0, DisplayName) == SLM_OK){
    // OK
    // DisplayName => "LCOS-SLM"
}
else{
    // Error
} 
```

## 3.2.46 SLM_Ctrl_ReadVR

```txt
SLM_STATUS
SLM_Ctrl_ReadVR(
DWORD SLMNumber,
LPSTR DLL_DRIVE_OPTION_FPGA_ver
) 
```

## Summary

Read version information. DLL version, drive board firmware version, option board firmware version, FPGA firmware version 

## Parameters

SLMNumber: Specify SLM number (1, 2, 3…). DLL_DRIVE_OPTION_FPGA_ver: Pointer to a 64-byte buffer to store version data. 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read lcos product serial number
char versions [64];
if(SLM_Ctrl_ReadLS(1, versions) == SLM_OK){
    // OK
    // versions => "DLL:2.5.0,Drive:0322,Option:0321,FPGA:0110"
}
else{
    // Error
} 
```

```txt
3.2.47 SLM_Ctrl_ReadPS
SLM_STATUS
SLM_Ctrl_ReadPS(
DWORD SLMNumber,
DWORD BoardNo,
LPSTR ProductSerialNo
) 
```

## Summary

Read 12-digit product serial number of "Drive board" or "Option board". This number is the number on the product label. This function is valid for drive board 0322 and option board 0321 or later. 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
BoardNo: 0: Driver board, 1: Option board.
ProductSerialNo: Pointer to a 16-byte buffer to store ProductSerialNo. 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read product serial number
char ProductSerialNo [16];
if(SLM_Ctrl_ReadSO(1, ProductSerialNo) == SLM_OK){
    // OK
    // ProductSerialNo => "123456789012"
}
else{
    // Error
} 
```

```txt
3.2.48 SLM_Ctrl_ReadLS

SLM_STATUS
SLM_Ctrl_ReadLS(
DWORD SLMNumber,
DWORD BoardNo,
LPSTR LCOSSerialNo
) 
```

## Summary

Read Max 20-digit LCOS product serial number of "Drive board" or "Option board". This number is the number on the product label. This function is valid for drive board 0322 and option board 0321 or later. 

## Parameters

```txt
SLMNumber: Specify SLM number (1, 2, 3...).
BoardNo: 0: Driver board, 1: Option board.
LCOSSerialNo: Pointer to a 32-byte buffer to store LCOSSerialNo. 
```

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## Example

```txt
// Read lcos product serial number
char LCOSSerialNo [32];
if(SLM_Ctrl_ReadLS(1, 0, LCOSSerialNo) == SLM_OK){
    // OK
    // LCOSSerialNo => "00000000000000000000"
}
else{
    // Error
} 
```

```txt
SLM_STATUS
SLM_Ctrl_Close (
DWORD SLMNumber, 
```

## 3.3 Finalizing

Summary Close USB interface. 

## Parameters

SLMNumber: Specify SLM number (1-8). 

## Return Value

SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

Example // Open USB interface if(SLM_Ctrl_Close (1) == SLM_OK){ // OK } 

```txt
Example
    // Reboot
    if(SLM_Ctrl_Reboot(1)==SLM_OK){
    // OK
    } 
```

## 3.4 Other

```txt
SLM_STATUS
SLM_Ctrl_Reboot (
DWORD SLMNumber, 
```

## Summary

Reboot SLM. 

After using this function, connect again with SLM_Ctrl_Open. 

## Parameters

SLMNumber: Specify SLM number (1-8). 

Return Value SLM_OK if successful, otherwise SLM_STATUS error code is returned. (Refer to “3.5 SLM_STATUS”) 

## 3.5 SLM_STATUS

SLM_STATUS is obtained as a return value when SLM function is executed. 

You can use this return value to check SLM status. 


Table 3.5-1 SLM_STATUS


<table><tr><td>Defined name</td><td>Return Value</td><td>Note</td></tr><tr><td>SLM_OK</td><td>0</td><td></td></tr><tr><td>SLM_NG</td><td>1</td><td>NG</td></tr><tr><td>SLM_BS</td><td>2</td><td>SLM is Busy</td></tr><tr><td>SLM_ER</td><td>3</td><td>parameter error</td></tr><tr><td></td><td></td><td></td></tr><tr><td>SLM_INVAID_MONITOR</td><td>-1</td><td>not find display no</td></tr><tr><td>SLM_NOT_OPEN_MONITOR</td><td>-2</td><td>not open display</td></tr><tr><td>SLM_OPEN_WINDOW_ERR</td><td>-3</td><td>window open error</td></tr><tr><td>SLM_DATA_FORMAT_ERR</td><td>-4</td><td>data format error</td></tr><tr><td>SLM_FILE_READ_ERR</td><td>-101</td><td>over 1023</td></tr><tr><td>SLM_NOT_OPEN_USB</td><td>-200</td><td>not open USB</td></tr><tr><td>SLM_OTHER_ERROR</td><td>-1000</td><td>other error</td></tr><tr><td></td><td></td><td></td></tr><tr><td>FT_INVALID_HANDLE</td><td>-10001</td><td>USB driver error.</td></tr><tr><td>FT_DEVICE_NOT_FOUND</td><td>-10002</td><td>Check connected device&#x27;s power.If connected, reset the power.</td></tr><tr><td>FT_DEVICE_NOT_OPENED</td><td>-10003</td><td>Already opened.</td></tr><tr><td>FT_IO_ERROR</td><td>-10004</td><td>USB driver error.</td></tr><tr><td>FT_INSUFFICIENT_RESOURCES</td><td>-10005</td><td>USB driver error.</td></tr><tr><td>FT_INVALID_PARAMETER</td><td>-10006</td><td>USB driver error.</td></tr><tr><td>FT_INVALID_BAUD_RATE</td><td>-10007</td><td>USB driver error.</td></tr><tr><td>FT_DEVICE_NOT_OPENED_FOR_ERASE</td><td>-10008</td><td>USB driver error.</td></tr><tr><td>FT_DEVICE_NOT_OPENED_FOR_WRITE</td><td>-10009</td><td>USB driver error.</td></tr><tr><td>FT_FAILED_TO_WRITE_DEVICE</td><td>-10010</td><td>USB driver error.</td></tr><tr><td>FT_EEPROM_READ_FAILED</td><td>-10011</td><td>USB driver error.</td></tr><tr><td>FT_EEPROM_WRITE_FAILED</td><td>-10012</td><td>USB driver error.</td></tr><tr><td>FT_EEPROM_ERASE_FAILED</td><td>-10013</td><td>USB driver error.</td></tr><tr><td>FT_EEPROM_NOT_PRESENT</td><td>-10014</td><td>USB driver error.</td></tr><tr><td>FT_EEPROM_NOT_PROGRAMMED</td><td>-10015</td><td>USB driver error.</td></tr><tr><td>FT_INVALID_ARGS</td><td>-10016</td><td>USB driver error.</td></tr><tr><td>FT_NOT_SUPPORTED</td><td>-10017</td><td>USB driver error.</td></tr><tr><td>FT_NO_MORE_ITEMS</td><td>-10018</td><td>USB driver error.</td></tr><tr><td>FT_TIMEOUT</td><td>-10019</td><td>USB driver error.</td></tr><tr><td>FT_OPERATION_ABORTED</td><td>-10020</td><td>USB driver error.</td></tr><tr><td>FT_RESERVED_PIPE</td><td>-10021</td><td>USB driver error.</td></tr><tr><td>FT_INVALID_CONTROL_REQUEST_DIRECTION</td><td>-10022</td><td>USB driver error.</td></tr><tr><td>FT_INVALID_CONTROL_REQUEST_TYPE</td><td>-10023</td><td>USB driver error.</td></tr><tr><td>FT_IO_PENDING</td><td>-10024</td><td>USB driver error.</td></tr><tr><td>FT_IO_INCOMPLETE</td><td>-10025</td><td>USB driver error.</td></tr><tr><td>FT_HANDLE_EOF</td><td>-10026</td><td>USB driver error.</td></tr><tr><td>FT_BUSY</td><td>-10027</td><td>USB driver error.</td></tr><tr><td>FT_NO_SYSTEM_RESOURCES</td><td>-10028</td><td>USB driver error.</td></tr><tr><td>FT_DEVICE_LIST_NOT_READY</td><td>-10029</td><td>USB driver error.</td></tr><tr><td>FT_DEVICE_NOT_CONNECTED</td><td>-10030</td><td>USB driver error.</td></tr><tr><td>FT_INCORRECT_DEVICE_PATH</td><td>-10031</td><td>USB driver error.</td></tr><tr><td>FT_OTHER_ERROR</td><td>-10032</td><td>USB driver error.</td></tr></table>

## 3.6 BMP, CSV, Data Flags


Table 3.6-1 : BMP Flags


<table><tr><td>Status Name</td><td>Flags Value 8bit color</td><td>Flags Value 10bit color (10bit is simply 4 times 8bit)</td><td>define name</td><td>Note</td></tr><tr><td>Original color display</td><td>—</td><td>0x0</td><td>FLAGS_COLOR_NOP</td><td>BMP only.</td></tr><tr><td>Only red color display</td><td>0x01</td><td>0x101</td><td>FLAGS_COLOR_R</td><td>BMP only.</td></tr><tr><td>Only green color display</td><td>0x02</td><td>0x102</td><td>FLAGS_COLOR_G</td><td>BMP only.</td></tr><tr><td>Only blue color display</td><td>0x04</td><td>0x104</td><td>FLAGS_COLOR_B</td><td>BMP only.</td></tr><tr><td>Convert grayscale (Y=0.299R+0.587G+0.114B)</td><td>0x08</td><td>0x108</td><td>FLAGS_COLOR_GRAY</td><td>BMP only.</td></tr><tr><td>Rate 120Hz SLM</td><td>0x20000000</td><td>0x20000100</td><td>FLAGS_RATE120</td><td></td></tr></table>


BMP Data Format


<table><tr><td>bit</td><td>7</td><td>6</td><td>5</td><td>4</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>Red</td><td>R7</td><td>R6</td><td>R5</td><td>R4</td><td>R3</td><td>R2</td><td>R1</td><td>R0</td></tr><tr><td>Green</td><td>G7</td><td>G6</td><td>G5</td><td>G4</td><td>G3</td><td>G2</td><td>G1</td><td>G0</td></tr><tr><td>Bule</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td><td>B3</td><td>B2</td><td>B1</td><td>B0</td></tr></table>


Original color display (Use Red 3bit,Green 3bit,Bule 4bit)


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R7</td><td>R6</td><td>R5</td><td>G7</td><td>G6</td><td>G5</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td></tr></table>


Only red color display


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>0</td><td>0</td><td>R7</td><td>R6</td><td>R5</td><td>R4</td><td>R3</td><td>R2</td><td>R1</td><td>R0</td></tr></table>


Only red color display & 10bit color


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R7</td><td>R6</td><td>R5</td><td>R4</td><td>R3</td><td>R2</td><td>R1</td><td>R0</td><td>0</td><td>0</td></tr></table>


Only green color display


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>0</td><td>0</td><td>G7</td><td>G6</td><td>G5</td><td>G4</td><td>G3</td><td>G2</td><td>G1</td><td>G0</td></tr></table>


Only green color display & 10bit color


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>G7</td><td>G6</td><td>G5</td><td>G4</td><td>G3</td><td>G2</td><td>G1</td><td>G0</td><td>0</td><td>0</td></tr></table>


Only blue color display


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>0</td><td>0</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td><td>B3</td><td>B2</td><td>B1</td><td>B0</td></tr></table>


Only blue color display & 10bit color


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td><td>B3</td><td>B2</td><td>B1</td><td>B0</td><td>0</td><td>0</td></tr></table>


Convert grayscale


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>0</td><td>0</td><td>G7</td><td>G6</td><td>G5</td><td>G4</td><td>G3</td><td>G2</td><td>G1</td><td>G0</td></tr></table>


Convert grayscale & 10bit color


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>G7</td><td>G6</td><td>G5</td><td>G4</td><td>G3</td><td>G2</td><td>G1</td><td>G0</td><td>0</td><td>0</td></tr></table>

120Hz model 

FLAGS_RATE120 bit OFF. 


First frame


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R7</td><td>R6</td><td>R5</td><td>G7</td><td>G6</td><td>G5</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td></tr><tr><td colspan="11">Next frame</td></tr><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R3</td><td>R2</td><td>R1</td><td>G3</td><td>G2</td><td>G1</td><td>B3</td><td>B2</td><td>B1</td><td>B0</td></tr></table>

FLAGS_RATE120 bit ON. 


First frame


<table><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R7</td><td>R6</td><td>R5</td><td>G7</td><td>G6</td><td>G5</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td></tr><tr><td colspan="11">Next frame</td></tr><tr><td>bit</td><td>9</td><td>8</td><td>7</td><td>6</td><td>5</td><td>6</td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>LCOS Format</td><td>R7</td><td>R6</td><td>R5</td><td>G7</td><td>G6</td><td>G5</td><td>B7</td><td>B6</td><td>B5</td><td>B4</td></tr></table>


Fig. 3.6-1 10bit encoding format in RGB color


## 3.7 CSV Format

The CSV file with data format made in accordance with Fig. 3.7-1 can be opened. The data can be edited using spreadsheet like Microsoft excel. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/9dc418b43c6b8142fc140b31b0a6f55113f3b132b82850bd3baa1eac2f658bd5.jpg)



Y: Vertical pixel number



Gray scale level for each pixel: 0~1023 (10 bit)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/dd364f101efaec336f9e47bcf47700d93bdbfaea10debe22adc89996629746aa.jpg)



(a) All-in-one model



0 to 1023 corresponds to 0 to 2pi at specified wave length.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/186476b54e46998dc9cc1a9511b358996656bf6d414249b6a561a2b49942cb0d.jpg)



(b) Separate model (LCOS unit)



Fig. 3.7-1 : Data format of pattern files.


## 3.8 Display table setting

<Default> 

<table><tr><td>Table number</td><td>Table area</td><td>Memory number</td><td>Memory area</td></tr><tr><td>1</td><td>Memory number 1</td><td>1</td><td>Phase pattern</td></tr><tr><td>2</td><td>Memory number 2</td><td>2</td><td>Phase pattern</td></tr><tr><td>3</td><td>Memory number 3</td><td>3</td><td>Phase pattern</td></tr><tr><td>4</td><td>Memory number 4</td><td>4</td><td>Phase pattern</td></tr><tr><td>5</td><td>Memory number 5</td><td>5</td><td>Phase pattern</td></tr><tr><td>.</td><td>.</td><td>.</td><td>.</td></tr><tr><td>.</td><td>.</td><td>.</td><td>.</td></tr><tr><td>.</td><td>.</td><td>.</td><td>.</td></tr><tr><td>124</td><td>Memory number 124</td><td>124</td><td>Phase pattern</td></tr><tr><td>125</td><td>Memory number 125</td><td>125</td><td>Phase pattern</td></tr><tr><td>126</td><td>Memory number 126</td><td>126</td><td>Phase pattern</td></tr><tr><td>127</td><td>Memory number 127</td><td>127</td><td>Phase pattern</td></tr><tr><td>128</td><td>Memory number 128</td><td>128</td><td>Phase pattern</td></tr></table>


Fig. 3.8-1 : Default table



<Display table change> 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/c0c7f990b7c80bebc80233dd7ab5205ce97bfd920427f645e4bdd2790def6567.jpg)



Fig. 3.8-2 : Display table changed


## 4 Samples

## 4.1 VB.net

## 4.1.1 Project Setting

32bit: “Compile -> Target CPU” to x86. 

64bit: “Compile -> Target CPU” to x64. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/972a70e35288c7be6ebda37988fecc71846409918217820c889f53a6bac018ed.jpg)


```vba
4.1.2 Sample source
Imports System.Runtime.InteropServices
Module SLMFunc

'/ SLM Status Codes
*************************/
Public Enum SLM_STATUS As Integer
    SLM_OK = 0    ' OK
    SLM_INVALID_MONITOR = -1    ' not find display no
    SLM_NO_OPEN_MONITOR = -2    ' not open display
    SLM_OPEN_WINDOW_ERR = -3    ' window open error
    SLM_DATA_FORMAT_ERR = -4    ' data format error

    SLM_FILE_READ_ERR = -101    ' not find file
    SLM_OTHER_ERROR = -1000    ' other Error
End Enum

Private Const DLLFileName As String = "SLMFunc.dll"

<System.Runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_Info(ByVal DisplayNumber As UInt32, ByVal width As UShort, ByVal height As UShort) As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_Open(ByVal DisplayNumber As UInt32) As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_Close(ByVal DisplayNumber As UInt32) As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_GrayScale(ByVal DisplayNumber As UInt32, ByVal Flags As UInt32, ByVal GrayScale As UShort)
As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_BMP(ByVal DisplayNumber As UInt32, ByVal Flags As UInt32, ByVal b As IntPtr) As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_Data(ByVal DisplayNumber As UInt32, ByVal width As UInt32, ByVal height As UInt16, ByVal Flags
As UInt32, ByVal data() As UShort) As Int32
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_ReadBMP(ByVal DisplayNumber As Integer, ByVal Flags As UInt32,
<MarshalAs(UnmanagedType.LPWStr)> ByVal Para1 As String) As Integer
End Function

<System.runtime.InteropServices.DllImport(DLLFileName,
CallingConvention:=Runtime.InteropServices.CallingConvention.Cdecl)>
Function SLM_Disp_ReadCSV(ByVal DisplayNumber As Integer, ByVal Flags As UInt32,
<MarshalAs(UnmanagedType.LPWStr)> ByVal Para1 As String) As Integer
End Function 
```

Public Class Form1 

Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click 

If (SLM_Disp_Open(2) = SLM_STATUS.SLM_OK) Then 

SLM_Disp_GrayScale(2, 0, 100) 

System.Threading.Thread.Sleep(1000) 

SLM_Disp_Close(2) 

End If 

End Sub 

End Class 

Form1 

Button 1 

Displayed all pixel in grayscale 100 when click “Button 1”. 

## 4.2 Python 3.6 Sample source

```python
# ================= Record
# SLMFunc.dll Sample Program for Python
# ================= Record
# coding:utf-8
import ctypes
import time
import numpy as np

# ================= Record
# 2D gradation
# ================= Record
def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
    return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
    return np.tile(np.linspace(start, stop, height), (width, 1)).T

# ================= Record
# Main
# ================= Record
dNo = 1
sampleFolder = 'C:\workspace\¥SLM\¥Files\¥2k\¥'
# ================= Record
# set dll
# ================= Record
dll = ctypes.cdll.LoadLibrary('SLMFunc.dll')

# ================= Record
# open
# SLM_STATUS SLM_Disp_Open(DWORD dNo)
# ================= Record
dll.SLM_Disp_Open(ctypes.c_int32(dNo))

# ================= Record
# grayscale test
# SLM_STATUS SLM_Disp_GrayScale(DWORD dNo, DWORD type, USHORTOR GrayScale)
# ================= Record
for i in range(10):
    ret = dll.SLM_Disp_GrayScale(ctypes.c_int32(dNo),ctypes.c_int32(0),ctypes.c_int(i*10))
    if(ret != 0): print(ret)
    time.sleep(0.05)
time.sleep(0.1)

# ================= Record
# array data test
# SLM_STATUS SLM_Disp_Data(DWORD dNo, USHORTORT width, USHORTORT height, DWORD type, short* data)
# ================= Record
n = get_gradation_2d(0,1023,1920,1200,1)
n1 = n.astype(np.int16)
n_h, n_w = n1.shape # height, width

for i in range(5):
    n1 = np.roll(n1,10)
    c = n1.ctypes.data_as(ctypes.POINTER((ctypes.c_int16 * n_h) * n_w)).contents # convert
    ret = dll.SLM_Disp_Data(ctypes.c_int32(dNo),ctypes.c_int16(n_w),ctypes.c_int16(n_h),ctypes.c_int32(0),c)
    if(ret != 0): print(ret)
    time.sleep(0.05)

time.sleep(0.5)

# ================= Record
# bmp data test
# SLM_STATUS SLM_Disp_BMP(DWORD dNo, DWORD type, HBITMAP bmp)
# no sample
# ================= Record
#time.sleep(0.5)

# ================= Record 
```

## 4.3 Other sample source

Sample files are in the following location after extracting distribution file (SLM_DLL_ver.x.x.zip). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/48956a7fba89878883e4508aeec62ec62b528756b6898f2d2657b7328c44ebb5.jpg)


## 5 Revision History


Table 5 Revision History


<table><tr><td>Revision</td><td>Changes</td><td>Date</td></tr><tr><td>2.0</td><td>Initial Release</td><td>2020.04.13</td></tr><tr><td>2.4</td><td>Add 120Hz Option.SLM_Disp_GrayScale, SLM_Disp_Data, SLM_Disp_ReadBMP,SLM_Disp_ReadBMP_A, SLM_Disp_ReadCSV, SLM_Disp_ReadCSV_A</td><td>2021.07.13</td></tr><tr><td>2.5</td><td>Add Functions.SLM_Ctrl_ReadTD, SLM_Ctrl_ReadTO, SLM_Ctrl_ReadED, SLM_Ctrl_ReadEO,SLM_Ctrl_ReadSD, SLM_Ctrl_ReadSO, SLM_Ctrl_WritePN, SLM_Ctrl_ReadPN,SLM_Ctrl_ReadVR, SLM_Ctrl_ReadPS, SLM_Ctrl_ReadLS, SLM_Ctrl_Reboot</td><td>2021.07.12</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-07/42e9ee5b-1146-40df-8171-8ac8bd17c287/c305a7ac5ca14904de2866bdd604109ab54dceeb5a6dc6936a4305216f4ef73e.jpg)


6 Contact 

## WARNING!

In the event of any trouble with this product, turn the unit off in accordance with the procedures to shut off the power described in this operation manual, disconnect the power source cord, record the product name and serial number described on the name plate of the product, and then contact our dealer at your place or directly contact us at Santec Photonics Laboratories. Our telephone number and facsimile number are shown below. However, we are not responsible for any trouble arising from your own repair or modification on this product. 

5823 Ohkusa-Nenjyozaka, Komaki, Aichi 485-0802, Japan 

## SANTEC CORPORATION

Tel. +81-568-79-1959 

Fax +81-568-79-1718 

433 Hackensack Ave., Hackensack, NJ 07601, U.S.A. 

SANTEC U.S.A. CORPORATION 

Toll Free +1-201-488-5505 

Fax +1-201-488-7702 

Grand Union Studios, 332 Ladbroke Grove, London W10 5AD 

SANTEC EUROPE LIMITED 

Tel. +44-20-3176-1550 

11F Room E, Hua Du Bldg., No.838 Zhangyang Road, Pudong, Shanghai 200122 China 

SANTEC (SHANGHAI) Co., Ltd 

Tel. +86-21-58361261, +86-21-58361262 

Fax +86-21-58361263 

www.santec.com 