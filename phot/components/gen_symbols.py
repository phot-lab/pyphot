import numpy as np
from commpy.modulation import QAMModem


def gen_symbols(num_symbols, bits_per_symbol):
    data_x = np.random.randint(0, 2, (num_symbols * bits_per_symbol, 1))  # 采用randint函数随机产生0 1码元序列x
    data_y = np.random.randint(0, 2, (num_symbols * bits_per_symbol, 1))  # 采用randint函数随机产生0 1码元序列y

    # QAM调制器
    modem = QAMModem(2 ** bits_per_symbol)
    tx_symbols_x = modem.modulate(data_x).reshape((-1, 1))  # X偏振信号
    tx_symbols_y = modem.modulate(data_y).reshape((-1, 1))  # Y偏振信号

    return tx_symbols_x, tx_symbols_y
