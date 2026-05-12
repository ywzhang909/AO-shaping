#pragma once

#ifdef __cplusplus
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT __declspec(dllexport)
#endif





/*
		  bit31                       bit0
		  HHHHHHHHLLLLLLLLHHHHHHHHLLLLLLLL
		  [3]     [2]     [1]     [0]
		  00000000RRRRRRRRGGGGGGGGBBBBBBBB
								9876543210
								RRRGGGBBBB   10bitData
		  00000000RRR0rrr0GGG0ggg0BBBBbbbb   LCOS FORMAT

*/
#define MASK_R 0b1110000000
#define MASK_G 0b0001110000
#define MASK_B 0b0000001111
#define SHIFT_R 14
#define SHIFT_G 9
#define SHIFT_B 4
#define SHIFT_R2 10
#define SHIFT_G2 5
#define SHIFT_B2 0


// 10bit -> 24bit
#define LCOSRGB2(b)     ((COLORREF)((((((DWORD)(b)&MASK_B)<<SHIFT_B )|((DWORD)((b)&MASK_G)<<SHIFT_G )|((DWORD)((b)&MASK_R)<<SHIFT_R))) | \
									 ((((DWORD)(b)&MASK_B)<<SHIFT_B2)|((DWORD)((b)&MASK_G)<<SHIFT_G2)|((DWORD)((b)&MASK_R)<<SHIFT_R2))))

#define LCOSRGB(b)      ((COLORREF)((((DWORD)(b)&MASK_B)<<SHIFT_B)|((DWORD)((b)&MASK_G)<<SHIFT_G)|((DWORD)((b)&MASK_R)<<SHIFT_R)))						// 10bit -> 24bit

// 24bit -> 10bit
#define RGBLCOS(b)		((USHORT)((((b)&(((DWORD)MASK_B)<<4))>>SHIFT_B)|(((b)&(((DWORD)MASK_G)<<9))>>9)|(((b)&(((DWORD)MASK_R)<<14))>>SHIFT_R)))	






#define MAX_WIDTH  1920
#define MAX_HEIGHT 1200


/*****************************************************************************
// SLM Flags
*****************************************************************************/
const DWORD FLAGS_INCWORD = 0x10000000UL;


const DWORD FLAGS_COLOR_MASK      = 0x000000FFUL;
const DWORD FLAGS_COLOR_NOP       = 0x00000000UL;
const DWORD FLAGS_COLOR_R         = 0x00000001UL;
const DWORD FLAGS_COLOR_G         = 0x00000002UL;
const DWORD FLAGS_COLOR_B         = 0x00000004UL;
const DWORD FLAGS_COLOR_GRAY      = 0x00000008UL;
const DWORD FLAGS_COLOR_10BIT     = 0x00000100UL;
const DWORD FLAGS_COLOR_FILL_MASK = 0x00FFF000UL;		// 000-3FF:0-1023 glayscale, FFF:Transparent color
const DWORD FLAGS_RATE120		  = 0x20000000UL;


/*****************************************************************************
// SLM Status Codes
*****************************************************************************/
typedef enum _SLM_STATUS
{
	SLM_OK = 0,							// OK
	SLM_NG = 1,							// NG
	SLM_BS = 2,							// Busy
	SLM_ER = 3,							// parameter ER
	SLM_INVAID_MONITOR = -1,		// not find display no
	SLM_NOT_OPEN_MONITOR = -2,		// not open display
	SLM_OPEN_WINDOW_ERR = -3,		// window open error
	SLM_DATA_FORMAT_ERR = -4,		// data foramt error


	SLM_FILE_READ_ERR = -101,		// not find  file

	SLM_NOT_OPEN_USB = -200,		// not open usb


	SLM_OTHER_ERROR = -1000,		// other error
	SLM_FTDI_ERROR  = -10000		// -10000 -> -10032


};

typedef int SLM_STATUS;

EXPORT SLM_STATUS SLM_Disp_Info(DWORD DisplayNumber, USHORT *width, USHORT *height);						// Info
EXPORT SLM_STATUS SLM_Disp_Info2(DWORD DisplayNumber, USHORT *width, USHORT *height, LPSTR DisplayName );	// Info2
EXPORT SLM_STATUS SLM_Disp_Open(DWORD DisplayNumber);													// Open	SLM Display
EXPORT SLM_STATUS SLM_Disp_Open2(DWORD DisplayNumber, SHORT x, SHORT y, USHORT width, USHORT height);
EXPORT SLM_STATUS SLM_Disp_Close(DWORD DisplayNumber);													// Close SLM Display
EXPORT SLM_STATUS SLM_Disp_GrayScale(DWORD DisplayNumber, DWORD Flags, USHORT GrayScale);				// All Pixel Set GrayScale
EXPORT SLM_STATUS SLM_Disp_BMP(DWORD DisplayNumber, DWORD Flags, HBITMAP bmp);							// BMP data
EXPORT SLM_STATUS SLM_Disp_Data(DWORD DisplayNumber, USHORT width, USHORT height, DWORD Flags, USHORT* data);	// Array data

EXPORT SLM_STATUS SLM_Disp_ReadBMP(DWORD DisplayNumber, DWORD Flags, LPCWSTR FileName);					// BMP filename(unicode)
EXPORT SLM_STATUS SLM_Disp_ReadCSV(DWORD DisplayNumber, DWORD Flags, LPCWSTR FileName);					// CSV filename(unicode)

EXPORT SLM_STATUS SLM_Disp_ReadBMP_A(DWORD DisplayNumber, DWORD Flags, LPCSTR FileName);				// BMP filename(ansi)
EXPORT SLM_STATUS SLM_Disp_ReadCSV_A(DWORD DisplayNumber, DWORD Flags, LPCSTR FileName);				// CSV filename(ansi)


EXPORT SLM_STATUS SLM_Ctrl_Open(DWORD SLMNumber);													// USB Open
EXPORT SLM_STATUS SLM_Ctrl_Close(DWORD SLMNumber);													// USB Close

EXPORT SLM_STATUS SLM_Ctrl_WriteXX(DWORD SLMNumber, BYTE* send, USHORT send_len, BYTE* recv, USHORT* recv_len, DWORD retry);
EXPORT SLM_STATUS SLM_Ctrl_Read(DWORD SLMNumber, BYTE* recv, USHORT* recv_len);



EXPORT SLM_STATUS SLM_Ctrl_WriteVI(DWORD SLMNumber, DWORD mode);
EXPORT SLM_STATUS SLM_Ctrl_ReadVI(DWORD SLMNumber, DWORD *mode);
EXPORT SLM_STATUS SLM_Ctrl_WriteWL(DWORD SLMNumber, DWORD wavelength, DWORD phase);
EXPORT SLM_STATUS SLM_Ctrl_ReadWL(DWORD SLMNumber, DWORD *wavelength, DWORD *phase);
EXPORT SLM_STATUS SLM_Ctrl_WriteAW(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteTI(DWORD SLMNumber, DWORD onoff);
EXPORT SLM_STATUS SLM_Ctrl_ReadTI(DWORD SLMNumber, DWORD *onoff);
EXPORT SLM_STATUS SLM_Ctrl_WriteTM(DWORD SLMNumber, DWORD onoff);
EXPORT SLM_STATUS SLM_Ctrl_ReadTM(DWORD SLMNumber, DWORD *onoff);
EXPORT SLM_STATUS SLM_Ctrl_WriteTC(DWORD SLMNumber, DWORD order);
EXPORT SLM_STATUS SLM_Ctrl_ReadTC(DWORD SLMNumber, DWORD *order);
EXPORT SLM_STATUS SLM_Ctrl_WriteTS(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteMC(DWORD SLMNumber, DWORD MemoryNumber);


EXPORT SLM_STATUS SLM_Ctrl_WriteMI(DWORD SLMNumber, DWORD MemoryNumber, USHORT width, USHORT height, DWORD Flags, USHORT* data);
EXPORT SLM_STATUS SLM_Ctrl_WriteMI_BMP(DWORD SLMNumber, DWORD MemoryNumber, DWORD BMPFlags, LPCWSTR FileName);
EXPORT SLM_STATUS SLM_Ctrl_WriteMI_CSV(DWORD SLMNumber, DWORD MemoryNumber, DWORD CSVFlags, LPCWSTR FileName);
EXPORT SLM_STATUS SLM_Ctrl_WriteMI_BMP_A(DWORD SLMNumber, DWORD MemoryNumber, DWORD BMPFlags, LPCSTR FileName);
EXPORT SLM_STATUS SLM_Ctrl_WriteMI_CSV_A(DWORD SLMNumber, DWORD MemoryNumber, DWORD CSVFlags, LPCSTR FileName);
EXPORT SLM_STATUS SLM_Ctrl_WriteME(DWORD SLMNumber, DWORD MemoryNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteMT(DWORD SLMNumber, DWORD TableNumber, DWORD MemoryNumber);

EXPORT SLM_STATUS SLM_Ctrl_ReadMS(DWORD SLMNumber, DWORD TableNumber, DWORD *MemoryNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteMR(DWORD SLMNumber, DWORD TableNumber1, DWORD TableNumber2);
EXPORT SLM_STATUS SLM_Ctrl_ReadMR(DWORD SLMNumber, DWORD *TableNumber1, DWORD *TableNumber2);
EXPORT SLM_STATUS SLM_Ctrl_WriteMP(DWORD SLMNumber, DWORD TableNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteMZ(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteMW(DWORD SLMNumber, DWORD frames);
EXPORT SLM_STATUS SLM_Ctrl_ReadMW(DWORD SLMNumber, DWORD *frames);
EXPORT SLM_STATUS SLM_Ctrl_WriteDS(DWORD SLMNumber, DWORD MemoryNumber);
EXPORT SLM_STATUS SLM_Ctrl_ReadDS(DWORD SLMNumber, DWORD *MemoryNumber);
EXPORT SLM_STATUS SLM_Ctrl_WriteDR(DWORD SLMNumber, DWORD order);

EXPORT SLM_STATUS SLM_Ctrl_WriteDB(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_WriteGS(DWORD SLMNumber, USHORT GrayScale);
EXPORT SLM_STATUS SLM_Ctrl_ReadGS(DWORD SLMNumber, USHORT *GrayScale);

EXPORT SLM_STATUS SLM_Ctrl_ReadT(DWORD SLMNumber, INT32 *driveTemp, INT32 *optionTemp);
EXPORT SLM_STATUS SLM_Ctrl_ReadTO(DWORD SLMNumber, INT32 *Temp);
EXPORT SLM_STATUS SLM_Ctrl_ReadTD(DWORD SLMNumber, INT32 *Temp);

EXPORT SLM_STATUS SLM_Ctrl_ReadEDO(DWORD SLMNumber, DWORD *driveError, DWORD *optionError);
EXPORT SLM_STATUS SLM_Ctrl_ReadED(DWORD SLMNumber, DWORD *Error);
EXPORT SLM_STATUS SLM_Ctrl_ReadEO(DWORD SLMNumber, DWORD *Error);

EXPORT SLM_STATUS SLM_Ctrl_ReadSU(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_ReadSDO(DWORD SLMNumber, LPSTR driveID, LPSTR optionID);
EXPORT SLM_STATUS SLM_Ctrl_ReadSD(DWORD SLMNumber, LPSTR ID);
EXPORT SLM_STATUS SLM_Ctrl_ReadSO(DWORD SLMNumber, LPSTR ID);

EXPORT SLM_STATUS SLM_Ctrl_ReadPS(DWORD SLMNumber, DWORD BoardNo, LPSTR ProductSerialNo);
EXPORT SLM_STATUS SLM_Ctrl_ReadLS(DWORD SLMNumber, DWORD BoardNo, LPSTR LCOSSerialNo);
EXPORT SLM_STATUS SLM_Ctrl_Reboot(DWORD SLMNumber);

EXPORT SLM_STATUS SLM_Ctrl_ReadVR(DWORD SLMNumber, LPSTR DLL_DRIVE_OPTION_FPGA_ver);
EXPORT SLM_STATUS SLM_Ctrl_ReadPN(DWORD SLMNumber, LPSTR DisplayName);
EXPORT SLM_STATUS SLM_Ctrl_WritePN(DWORD SLMNumber, LPCSTR DisplayName);




EXPORT SLM_STATUS SLM_Ctrl_ReadSN(DWORD SLMNumber, LPSTR SerialNo);
EXPORT SLM_STATUS SLM_Debug(DWORD onoff);

