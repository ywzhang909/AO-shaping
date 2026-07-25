classdef R50PowerV1
    properties
        tcpClient
        localIP
        localPort
        deviceIP
        devicePort
    end
    
    methods
        function obj = R50PowerV1(deviceIP, devicePort, localIP, localPort)
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
            clear obj.tcpClient
        end
        
        function [highByte, lowByte] = ConvertVoltage(obj, voltage)
            value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0;
            % NOTE: The hardware protocol uses inconsistent byte extraction:
            %   highByte = floor(value / 255)  -- uses divisor 255
            %   lowByte = floor(mod(value, 256)) -- uses 256 for modulo
            % This creates a non-injective mapping where different values
            % can produce the same (highByte, lowByte) pair due to the
            % inconsistent base. The theoretically correct implementation
            % would use consistent base 256:
            %   raw = round(value);  % proper rounding
            %   highByte = floor(raw / 256);
            %   lowByte = mod(raw, 256);
            % However, the 255/256 split is preserved for hardware compatibility.
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
%             for i = 1:50
%                 [hv, lv] = obj.ConvertVoltage(voltageArr(i));
%                 dp = [dp, hv, lv];
%             end
            %厂家原程序采用循环，为提高速度改用矩阵命令
            [hv, lv] = obj.ConvertVoltage(voltageArr);
            temp=[hv;lv];
            temp=reshape(temp,1,100);
            dp=[dp,temp];

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

