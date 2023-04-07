import phot
import time
import pandas as pd
from tabulate import tabulate


def transmit_end():
    # 设置全局系统仿真参数
    num_symbols = 2**16  # 符号数目
    bits_per_symbol = 4  # 2 for QPSK; 4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
    total_baud = 10e9  # 信号波特率，符号率
    up_sampling_factor = 2  # 上采样倍数
    sampling_rate = up_sampling_factor * total_baud  # 信号采样率

    """ 程序从此开始为发射端的仿真代码 """

    # 首先产生发射端X/Y双偏振信号

    bits = phot.gen_bits(num_symbols * bits_per_symbol)  # 生成两列随机二进制序列

    # QAM调制器
    symbols = phot.qam_modulate(bits, bits_per_symbol)

    # 此处先存储发射端原始发送信号，作为最后比较BER
    prev_symbols = symbols

    RRC_ROLL_OFF = 0.02  # RRC脉冲整形滚降系数
    shaper = phot.PulseShaper(
        up_sampling_factor=up_sampling_factor,
        len_filter=128 * up_sampling_factor,
        alpha=RRC_ROLL_OFF,
        ts=1 / total_baud,
        fs=sampling_rate,
    )

    signals = shaper.tx_shape(symbols)

    """ 加入AWG中DAC的量化噪声 """
    sampling_rate_awg = 96e9  # DAC采样率
    dac_resolution_bits = 8  # DAC的bit位数

    signals = phot.dac_noise(signals, sampling_rate_awg, sampling_rate, dac_resolution_bits)

    """ 加入发射端激光器产生的相位噪声 """
    linewidth_tx = 150e3  # 激光器线宽
    signals = phot.phase_noise(signals, sampling_rate / total_baud, linewidth_tx, total_baud)

    """ 根据设置的OSNR来加入高斯白噪声 """

    osnr_db = 30  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB
    signals = phot.gaussian_noise(signals, osnr_db, sampling_rate)
    return signals, sampling_rate


if __name__ == "__main__":
    phot.config(plot=False, backend="cupy")  # 全局开启画图，backend 使用 numpy

    signals, sampling_rate = transmit_end()

    # 实际情况：1000公里 10米一步
    num_spans = 5  # 多少个 span (每个span经过一次放大器)
    span_length = 75  # 一个 span 的长度 (km)
    delta_z = 1  # 单步步长 (km)
    alpha = 0.2
    beta2 = 21.6676e-24
    gamma = 1.3

    # 先把 GPU 连接一下
    phot.fiber(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)

    delta_z_list = [1, 0.1, 0.01]
    times = []
    for delta_z in delta_z_list:
        start = time.time()
        phot.fiber(signals, sampling_rate, num_spans, beta2, delta_z, gamma, alpha, span_length)
        duration = time.time() - start
        times.append(f"{duration:.2f}")
        print(f"Finished delta_z: {delta_z}, time: {duration:.2f} s")

    df = pd.DataFrame({"单步步长": delta_z, "time (s)": times})
    print(tabulate(df, tablefmt="github", headers="keys"))
