# tm/ - 定时模块驱动

定时模块 (Timing Module) 驱动模块, 提供串口通信的 FSM 控制。

## 结构

```
tm/
├── serial_port_fsm.py   # 串口定时模块 (FSM 控制)
└── __init__.py
```

## 关键类

| 类 | 文件 | 说明 |
|----|------|------|
| `SerialPortFSM` | `serial_port_fsm.py` | 串口定时模块 (有限状态机) |

## SerialPortFSM 接口

| 方法 | 说明 |
|------|------|
| `open()` | 打开串口连接 |
| `close()` | 关闭连接 |
| `send(x, y)` | 发送位置命令 |
| `send_in_queue(x, y)` | 队列方式发送 |
| `get_rx()` | 获取回读数据 |
| `wait_rx(timeout)` | 等待回读 |
| `list_port()` | 列出可用串口 (static) |
| `validate_received_data(data)` | 验证接收数据完整性 |
| `parse_position_data(data)` | 解析位置数据 |

详细接口文档见 [`INTERFACE_DOCS.md`](../INTERFACE_DOCS.md) §定时模块/TM 接口。
