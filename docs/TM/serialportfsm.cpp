#include "serialportfsm.h"

SerialPortFSM::SerialPortFSM(QObject *parent): QObject(parent), MaxPosition(1510.00), MinPosition(-1510.00), m_serialportfsm(new QSerialPort(this)){
    FSMPositionCommadArray.resize(13);
    // 连接接收数据的信号
    connect(m_serialportfsm, &QSerialPort::readyRead, this, &SerialPortFSM::onDataReceived, Qt::DirectConnection);
}

bool SerialPortFSM::init(const QString &portName,
                             int baudRate,
                             QSerialPort::DataBits dataBits,
                             QSerialPort::Parity parity,
                             QSerialPort::StopBits stopBits,
                             QSerialPort::FlowControl flowControl)
{
    // 检查串口是否存在
    if (!isPortAvailable(portName)) {
        qDebug() << "串口" << portName << "不存在！";
        return false;
    }
    m_serialportfsm->setPortName(portName);
    m_serialportfsm->setBaudRate(baudRate);
    m_serialportfsm->setDataBits(dataBits);
    m_serialportfsm->setParity(parity);
    m_serialportfsm->setStopBits(stopBits);
    m_serialportfsm->setFlowControl(flowControl);
    // 尝试打开串口
    if (m_serialportfsm->open(QIODevice::ReadWrite)) {
        qDebug() << "串口连接成功：" << portName;
        return true;
    } else {
        qDebug() << "串口连接失败：" << portName;
        return false;
    }
}

void SerialPortFSM::closePort()
{
    if (m_serialportfsm->isOpen()) {
        m_serialportfsm->close();
        qDebug() << "串口关闭";
    }
}


void SerialPortFSM::FSMPositionFLOAT2HEX(float PositionX, float PositionY, quint8 *pData){
    if(pData){
        // 限制移动量范围
        if (PositionX > MaxPosition)
            PositionX = MaxPosition;
        if (PositionX < MinPosition)
            PositionX = MinPosition;
        if (PositionY > MaxPosition)
            PositionY = MaxPosition;
        if (PositionY < MinPosition)
            PositionY = MinPosition;
        int16_t xInt = static_cast<int16_t>(std::round(PositionX/0.05));
        int16_t yInt = static_cast<int16_t>(std::round(PositionY/0.05));

        pData[0] = static_cast<quint8>((xInt >> 8) & 0xFF);  // X轴高8位
        pData[1] = static_cast<quint8>(xInt & 0xFF);         // X轴低8位
        pData[2] = static_cast<quint8>((yInt >> 8) & 0xFF);  // Y轴高8位
        pData[3] = static_cast<quint8>(yInt & 0xFF);         // Y轴低8位
    }
}


bool SerialPortFSM::sendFSMPositionData(float PositionX, float PositionY)
{
    if (m_serialportfsm->isOpen()) {
        quint8 *fsmpositioncommad = reinterpret_cast<quint8*>(FSMPositionCommadArray.data());
        memset(fsmpositioncommad, 0, FSMPositionCommadArray.size());
        fsmpositioncommad[0] = 0x7E;
        fsmpositioncommad[1] = 0xE7;
        fsmpositioncommad[2] = 0x01;
        FSMPositionFLOAT2HEX(PositionX,PositionY,fsmpositioncommad+3);
        fsmpositioncommad[7] = 0x00;
        fsmpositioncommad[8] = 0x00;
        fsmpositioncommad[9] = 0x00;
        fsmpositioncommad[10] = 0x00;
        fsmpositioncommad[11] = 0x00;

        quint8 checkSum = 0;
        for (int i = 2; i <= 11; ++i) {  // 注意：协议中校验范围是"除帧头外"，即字节3~12（索引2~11）
            checkSum += fsmpositioncommad[i];
        }
        checkSum = ~checkSum & 0xFF;  // 取反并保留低8位
        fsmpositioncommad[12] = checkSum;  // 校验位（第13字节）

        // 打印原始数据（十六进制）
        QString hexStr;
        for (int i = 0; i < 13; ++i) {
            hexStr += QString("%1 ").arg(fsmpositioncommad[i], 2, 16, QLatin1Char('0')).toUpper();
        }
        qDebug() << "发送数据（十六进制）：" << hexStr.trimmed();

        qint64 bytesWritten = m_serialportfsm->write(reinterpret_cast<const char*>(fsmpositioncommad),13);

        return bytesWritten == 13;
    } else {
        qDebug() << "串口未打开，无法发送数据";
        return false;
    }
}

QByteArray SerialPortFSM::readData()
{
    QByteArray data = m_serialportfsm->readAll();
    return data;
}

bool SerialPortFSM::isPortAvailable(const QString &portName)
{
    // 获取系统中的所有串口信息
    QList<QSerialPortInfo> availablePorts = QSerialPortInfo::availablePorts();

    for (const QSerialPortInfo &port : availablePorts) {
        if (port.portName() == portName) {
            return true;  // 找到指定的串口
        }
    }
    return false;  // 没有找到指定的串口
}

QList<QSerialPortInfo> SerialPortFSM::getAllPort()
{
    return QSerialPortInfo::availablePorts();
}

// 获取波特率
QList<QString> SerialPortFSM::getAllBaudRates()
{
    QList<QString> baudRates;
    QMetaEnum metaEnum = QMetaEnum::fromType<QSerialPort::BaudRate>();
    int count = metaEnum.keyCount();
    for (int i = 0; i < count; ++i) {
        //QString key = metaEnum.key(i);
        int value = metaEnum.value(i);
        baudRates.append(QString::number(value));
        // 处理key和value
    }

    return baudRates;
}

// 获取数据位
QList<QString> SerialPortFSM::getAllDataBits()
{
    QList<QString> dataBitsList;
    QMetaEnum metaEnum = QMetaEnum::fromType<QSerialPort::DataBits>();
    int count = metaEnum.keyCount();
    for (int i = 0; i < count; ++i) {
        //QString key = metaEnum.key(i);
        int value = metaEnum.value(i);
        dataBitsList.append(QString::number(value));
        // 处理key和value
    }
    return dataBitsList;
}

// 获取停止位
QList<QString> SerialPortFSM::getAllStopBits()
{
    QList<QString> stopBitsList;
    QMetaEnum metaEnum = QMetaEnum::fromType<QSerialPort::StopBits>();
    int count = metaEnum.keyCount();
    for (int i = 0; i < count; ++i) {
        QString key = metaEnum.key(i);
        stopBitsList.append(key);
        //int value = metaEnum.value(i);
        // 处理key和value
    }
    return stopBitsList;
}

// 获取校验位
QList<QString> SerialPortFSM::getAllParity()
{
    QList<QString> parityList;
    QMetaEnum metaEnum = QMetaEnum::fromType<QSerialPort::Parity>();
    int count = metaEnum.keyCount();
    for (int i = 0; i < count; ++i) {
        QString key = metaEnum.key(i);
        parityList.append(key);
        //int value = metaEnum.value(i);
        // 处理key和value
    }
    return parityList;
}

void SerialPortFSM::setDataReceivedCallback(std::function<void(const QByteArray &)> callback)
{
    m_dataReceivedCallback = callback;
}

void SerialPortFSM::onDataReceived()
{
    qDebug()<<"开始读取数据";
    // 从串口读取数据
    QByteArray data = m_serialportfsm->readAll();
    // 如果回调函数已设置，调用回调函数
    if (m_dataReceivedCallback) {
        m_dataReceivedCallback(data);
    }
}

