driver = R50PowerV1("192.168.0.110",10110);
driver.SetRelayState(1);
driver.SetAllChannelVoltage(5.0);
% % driver.SetChannelVoltage(5, 20.0);
% % driver.SetAllVoltageByArr(1:50);
% driver.SetRelayState(0);

driver.delete();