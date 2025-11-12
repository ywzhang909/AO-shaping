from ao_shaping.drivers.tm import TM

def test_port_list():
    print(TM.list_port())

def test_tm():
    tm = TM()
    tm.open()
    tm.send_pos(-200, 110)
    tm.close()
