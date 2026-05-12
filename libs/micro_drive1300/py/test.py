"""
DM Control Python Test Script
变形镜控制Python测试脚本

运行: python test.py
"""

from dm_control import DMController, ErrorCode, init, disconnect

def test_basic():
    """基础测试"""
    print("=" * 50)
    print("DM Control Python Test")
    print("=" * 50)
    
    # 使用上下文管理器
    with DMController() as dm:
        print(f"\n1. 初始化...")
        ret = dm.init()
        if ret != ErrorCode.SUCCESS:
            print(f"   初始化失败: {ret}")
            return
        
        print(f"   已连接控制器: {dm.connected_count}")
        
        print(f"\n2. 获取控制器IP...")
        for i in range(1, 27):
            ip = dm.get_controller_ip(i)
            print(f"   控制器 {i:2d}: {ip}")
        
        print(f"\n3. 设置所有通道为0V...")
        dm.init_all_actuators()
        print("   完成")
        
        print(f"\n4. 设置驱动器1为10V...")
        ret = dm.set_actuator_voltage(1, 10.0)
        print(f"   结果: {ret.name if hasattr(ret, 'name') else ret}")
        
        print(f"\n5. 获取驱动器1映射...")
        ctrl, ch = dm.get_actuator_mapping(1)
        print(f"   -> 控制器 {ctrl}, 通道 {ch}")
        
        print(f"\n6. 设置控制器1所有通道为50V...")
        dm.set_controller_voltage(1, 50.0)
        print("   完成")
        
        print(f"\n7. 控制继电器...")
        dm.open_relay()
        print("   继电器已打开")
        dm.close_relay()
        print("   继电器已关闭")
        
        print(f"\n8. 测试批量设置...")
        dm.set_voltage_all_controllers(30.0)
        print("   所有控制器设为30V")
        
        print("\n" + "=" * 50)
        print("测试完成!")
        print("=" * 50)


def test_with_excel_mapping():
    """测试使用Excel映射"""
    print("\n使用Excel映射测试...")
    
    with DMController() as dm:
        # 尝试加载Excel映射
        try:
            ret = dm.init(mapping_file="1300-5.xlsx")
            print(f"加载Excel映射: {ret.name if hasattr(ret, 'name') else ret}")
            
            # 显示前10个映射
            print("\n前10个驱动器映射:")
            for i in range(1, 11):
                ctrl, ch = dm.get_actuator_mapping(i)
                print(f"  驱动器 {i}: 控制器{ctrl}, 通道{ch}")
        except Exception as e:
            print(f"Excel加载失败: {e}")


def test_multi_actuators():
    """测试批量设置多个驱动器"""
    print("\n批量设置驱动器测试...")
    
    with DMController() as dm:
        dm.init()
        
        # 批量设置
        actuators = list(range(1, 101))  # 1-100
        voltages = [10.0] * 100
        
        ret = dm.set_multiple_actuators(actuators, voltages)
        print(f"批量设置100个驱动器: {ret.name if hasattr(ret, 'name') else ret}")


if __name__ == "__main__":
    import sys
    
    print("Python版本:", sys.version)
    print("正在导入dm_control模块...")
    
    try:
        test_basic()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 可选测试
    # test_with_excel_mapping()
    # test_multi_actuators()