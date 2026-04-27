classdef R50Power
    properties
        tcpClient
        localIP
        localPort
        deviceIP
        devicePort
    end
    
    methods
        function obj = R50Power(deviceIP, devicePort, localIP, localPort)
            if nargin < 4
                localIP = '0.0.0.0';
            end
            if nargin < 5
                localPort = 0;
            end

            obj.deviceIP = deviceIP;
            obj.devicePort = devicePort;
            obj.localIP = localIP;
            obj.localPort = localPort;
                %'LocalPort', localPort, 'LocalHost', localIP, ...
            obj.tcpClient = tcpclient(deviceIP, devicePort, ...
                'Timeout', 10);
        end

        function delete(obj)
        end
        
        function [highByte, lowByte] = ConvertVoltage(obj, voltage)
            value = (voltage + 1) / 20 / 3.4 / 3.3 * 65535.0;
            highByte = floor(value / 255);
            lowByte = floor(mod(value, 256));
        end

        function SetChannelVoltage(obj, channel, voltage)
            if channel < 0 || channel > 49
                error('error channel');
            end
            [hv, lv] = obj.ConvertVoltage(voltage);
            dp = [0xAA, 0xBB, 0x04, channel, hv, lv, 0xCC, 0xDD];
            write(obj.tcpClient, dp);
        end

        function SetAllVoltageByArr(obj, voltageArr)
            if length(voltageArr) ~= 50
                error('arr length not 50');
            end

            dp = [0xAA, 0xBB, 0x09];
            for i = 1:50
                [hv, lv] = obj.ConvertVoltage(voltageArr(i) + 20);
                dp = [dp, hv, lv];
            end

            dp = [dp, 0xCC, 0xDD];

            write(obj.tcpClient, dp);
        end

        function SetAllChannelVoltage(obj, voltage)
            [hv, lv] = obj.ConvertVoltage(voltage);
            dp = [0xAA, 0xBB, 0x08, hv, lv, 0xCC, 0xDD];
            write(obj.tcpClient, dp);
        end

        function SetRelayState(obj, state)
            if (state == 1)
                dp = [0xAA, 0xBB, 0x06, 0xCC, 0xDD];
            else
                dp = [0xAA, 0xBB, 0x07, 0xCC, 0xDD];
            end
            write(obj.tcpClient, dp);
        end

        function SetIP(obj, ipAddress)
            ipf = sscanf(ipAddress, '%d.%d.%d.%d');

            dp = [0xAA, 0xBB, 0x06, ipf(1), ipf(2), ipf(3), ipf(4), 0xCC, 0xDD];

            write(obj.tcpClient, dp);
        end
    end
end

