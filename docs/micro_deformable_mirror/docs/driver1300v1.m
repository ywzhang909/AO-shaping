function mydriver=driver1300v1
mydriver.duiying=@duiying;
mydriver.huanshu=@huanshu;
mydriver.lianjie=@lianjie;
mydriver.duankai=@duankai;
mydriver.zhengti=@zhengti;
mydriver.fendianyuanzhengti=@fendianyuanzhengti;
mydriver.fentongdaozhengti=@fentongdaozhengti;
mydriver.danyuan=@danyuan;
mydriver.fast=@fast;
mydriver.different=@different;
mydriver.apply=@apply;
mydriver.voliss=@voliss;
mydriver.relaystate=@relaystate;
end

function duiying
global volume issue page
str=readcell('1300-5.xlsx','Sheet','Sheet1');
[row,~]=size(str);
for i=1:row
    page(i)=str{i,1};
    volume(i)=str{i,2};
    issue(i)=str{i,3};
end
volume=volume-100;
end

function fanhui_array=huanshu(initial_actuator,chicun)
array=1:1296;
array=reshape(array,36,36);
array=array';
[x,y]=find(array==initial_actuator);
fanhui_array=array(x:x+chicun-1,y:y+chicun-1);
end

function lianjie
global R50Object dianyuan
dianyuan=[1 2];
fileID=fopen('IP_1300.txt','r');
index=1;
IP=cell(26,1);
while(fgetl(fileID)~=-1)
    IP{index}=fgetl(fileID);
    index=index+1;
end
fclose(fileID);
R50Object=cell(26,1);
for i=1:length(dianyuan)
    R50Object{dianyuan(i)}=R50PowerV1(IP{dianyuan(i)},dianyuan(i)+10100);
end
end

function duankai
fclose(instrfindall);
end

% 设置所有单元,voltage是电压，higher是上限，lower是下限
function zhengti(voltage,higher,lower)
global R50Object dianyuan
if voltage<=higher && voltage>=lower
    for i=dianyuan
       R50Object{i}.SetAllChannelVoltage(voltage);
    end
else
    errordlg('电压超出范围','警告');
end
end

function fendianyuanzhengti(voltage,element)
global R50Object
if voltage<=140 && voltage>=-20
    R50Object{element}.SetAllChannelVoltage(voltage);
else
    errordlg('电压超出范围','警告');
end
end

function fentongdaozhengti(voltage,element)
global R50Object dianyuan
if voltage<=140 && voltage>=-20
    cellfun(@(x) x.SetChannelVoltage(element,voltage),R50Object);
else
    errordlg('电压超出范围','警告');
end
end

% 设置某个单元电压,number为1~1296的行号（不是电极号），voltage是想要加的电压，higher是电压上限，lower是电压下限
function danyuan(number,voltage,higher,lower)
global page volume issue R50Object
a=find(page==number);
if length(a)~=0
    if voltage<=higher && voltage>=lower
        R50Object{volume(a)}.SetChannelVoltage(issue(a), voltage);
    else
        errordlg('电压超出范围','警告');
    end
end
end

function fanhui_matrix=fast(numbers,voltages)
global volume issue
matrix=zeros(16,324);
for i=1:length(numbers)
    switch volume(numbers(i))
        case 1
            matrix(1,issue(numbers(i)))=voltages(i);
        case 2
            matrix(2,issue(numbers(i)))=voltages(i);
        case 3
            matrix(3,issue(numbers(i)))=voltages(i);
        case 4
            matrix(4,issue(numbers(i)))=voltages(i);
        case 5
            matrix(5,issue(numbers(i)))=voltages(i);
        case 6
            matrix(6,issue(numbers(i)))=voltages(i);
        case 7
            matrix(7,issue(numbers(i)))=voltages(i);
        case 8
            matrix(8,issue(numbers(i)))=voltages(i);
        case 9
            matrix(9,issue(numbers(i)))=voltages(i);
        case 10
            matrix(10,issue(numbers(i)))=voltages(i);
        case 11
            matrix(11,issue(numbers(i)))=voltages(i);
        case 12
            matrix(12,issue(numbers(i)))=voltages(i);
        case 13
            matrix(13,issue(numbers(i)))=voltages(i);
        case 14
            matrix(14,issue(numbers(i)))=voltages(i);
        case 15
            matrix(15,issue(numbers(i)))=voltages(i);
        case 16
            matrix(16,issue(numbers(i)))=voltages(i);
    end
end
fanhui_matrix=matrix;
end

function different(matrix)
global Power1 Power2 Power3 Power4 Power5 Power6 Power7 Power8 Power9 Power10 Power11 Power12 Power13 Power14 Power15 Power16
for i=1:324
    Power1.SetOutValue(i,matrix(1,i));
    Power2.SetOutValue(i,matrix(2,i));
    Power3.SetOutValue(i,matrix(3,i));
    Power4.SetOutValue(i,matrix(4,i));
    Power5.SetOutValue(i,matrix(5,i));
    Power6.SetOutValue(i,matrix(6,i));
    Power7.SetOutValue(i,matrix(7,i));
    Power8.SetOutValue(i,matrix(8,i));
    Power9.SetOutValue(i,matrix(9,i));
    Power10.SetOutValue(i,matrix(10,i));
    Power11.SetOutValue(i,matrix(11,i));
    Power12.SetOutValue(i,matrix(12,i));
    Power13.SetOutValue(i,matrix(13,i));
    Power14.SetOutValue(i,matrix(14,i));
    Power15.SetOutValue(i,matrix(15,i));
    Power16.SetOutValue(i,matrix(16,i));
end
end

function apply
end

function [shebei,tongdao]=voliss(number)
global volume issue page
a=find(page==number);
shebei=volume(a);
tongdao=issue(a);
end

% 继电器打开关闭，zhuangtai=1打开，zhuangtai=0关闭
function relaystate(zhuangtai)
global R50Object dianyuan
for i=dianyuan
    R50Object{i}.SetRelayState(zhuangtai);
end
end