R50Object = R50Power('192.168.0.101', 10101); %连接设备

R50Object.SetRelayState(1); %打开继电器
R50Object.SetChannelVoltage(0, 0.1); %设置指定通道电压（通道，电压）
tempArr = 1:3:150;
R50Object.SetAllVoltageByArr(tempArr); %通过数组设置所有通道电压
R50Object.SetAllChannelVoltage(0); %把所有通道电压设置为指定值
R50Object.SetRelayState(0); %关闭继电器
t = -1 + (6.5 - (-1)) * rand(); %生成一个[-1,6.5]的随机数
R50Object.SetChannelVoltage(0, t); %通道0设置为随机电压
t = -1 + (6.5 - (-1)) * rand(1, 50); %生成50个随机数数组
R50Object.SetAllVoltageByArr(t); %设置所有通道



