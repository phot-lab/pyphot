from commpy.modulation import QAMModem


class Modem:
    def __init__(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol
        pass

    def modulate(self, data_x, data_y):
        # QAM调制器
        modem = QAMModem(2 ** self.bits_per_symbol)
        symbols_x = modem.modulate(data_x).reshape((-1, 1))  # X偏振信号
        symbols_y = modem.modulate(data_y).reshape((-1, 1))  # Y偏振信号
        return symbols_x, symbols_y
