import phot

if __name__ == '__main__':
    """ 本代码为程序主函数 本代码主要适用于 QPSK，16QAM，32QAM，64QAM 调制格式的单载波相干背靠背（B2B）信号 """

    # phot.config(plot=False)  # 全局关闭画图

    # 设置全局系统仿真参数
    num_symbols = 2 ** 18  # 符号数目
    bits_per_symbol = 4  # 2 for QPSK;4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
    total_baud = 64e9  # 信号波特率，符号率
    up_sampling_factor = 2  # 上采样倍数
    sampling_rate = up_sampling_factor * total_baud  # 信号采样率

    """ 程序从此开始为发射端的仿真代码 """

    # 首先产生发射端X/Y双偏振信号

    data_x, data_y = phot.gen_bits(num_symbols * bits_per_symbol)  # 生成两列随机二进制序列

    # QAM调制器
    modem = phot.Modem(bits_per_symbol)
    symbols_x, symbols_y = modem.modulate(data_x, data_y)

    # 此处先存储发射端原始发送信号，作为最后比较BER
    prev_symbols_x, prev_symbols_y = symbols_x, symbols_y

    RRC_ROLL_OFF = 0.1  # RRC脉冲整形滚降系数
    shaper = phot.PulseShaper(up_sampling_factor=up_sampling_factor, len_filter=128 * up_sampling_factor,
                              alpha=RRC_ROLL_OFF, ts=1 / total_baud, fs=sampling_rate)

    signal_x, signal_y = shaper.tx_shape(symbols_x, symbols_y)

    """ 加入AWG中DAC的量化噪声 """
    sampling_rate_awg = 96e9  # DAC采样率
    dac_resolution_bits = 8  # DAC的bit位数

    noise = phot.DacNoise(sampling_rate_awg, sampling_rate, dac_resolution_bits)

    signal_x, signal_y = noise.add(signal_x, signal_y)

    """ 加入发射端激光器产生的相位噪声 """
    linewidth_tx = 150e3  # 激光器线宽
    noise = phot.PhaseNoise(sampling_rate / total_baud, linewidth_tx, total_baud)

    signal_x, signal_y = noise.add(signal_x, signal_y)

    """ 根据设置的OSNR来加入高斯白噪声 """

    osnr_db = 25  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB
    noise = phot.GaussianNoise(osnr_db, sampling_rate)

    signal_x, signal_y = noise.add(signal_x, signal_y)

    """ 添加接收端激光器产生的相位噪声 """
    linewidth_rx = 150e3  # 激光器线宽
    noise = phot.PhaseNoise(sampling_rate / total_baud, linewidth_rx, total_baud)

    signal_x, signal_y = noise.add(signal_x, signal_y)

    """
    发射端代码此处截止
    以下开始为接收端的代码
    """

    """ 添加收发端激光器造成的频偏，就是发射端激光器和接收端激光器的中心频率的偏移差 """

    frequency_offset = 2e9  # 设置频偏，一般激光器的频偏范围为 -3G~3G Hz

    signal_x, signal_y = phot.add_freq_offset(signal_x, signal_y, frequency_offset, sampling_rate)

    """ 模拟接收机造成的I/Q失衡，主要考虑幅度失衡和相位失衡，这里将两者都加在虚部上 """

    signal_x, signal_y = phot.add_iq_imbalance(signal_x, signal_y)

    """ 加入ADC的量化噪声 """
    adc_sample_rate = 160e9  # ADC采样率
    adc_resolution_bits = 8  # ADC的bit位数

    signal_x, signal_y = phot.add_adc_noise(signal_x, signal_y, sampling_rate, adc_sample_rate, adc_resolution_bits)

    """ IQ正交化补偿，就是将之前的I/Q失衡的损伤补偿回来 """

    signal_x, signal_y = phot.iq_freq_offset_and_compensation(signal_x, signal_y, sampling_rate)

    """ 接收端相应的RRC脉冲整形，具体的参数代码与发射端的RRC滤波器是一致的 """
    signal_x, signal_y = shaper.rx_shape(signal_x, signal_y)

    """ 帧同步，寻找与发射端原始信号头部对应的符号 """
    syncer = phot.FrameSyncer(up_sampling_factor)
    signal_x, signal_y, prev_symbols_x, prev_symbols_y = syncer.sync(signal_x, signal_y, prev_symbols_x, prev_symbols_y)

    """ 自适应均衡，此处采用恒模算法（CMA）对收敛系数进行预收敛，再拿收敛后的滤波器系数对正式的信号使用半径定向算法（RDE）进行均衡收敛，总的思想采用梯度下降法 """

    num_tap = 25  # 均衡器抽头数目，此处均衡器内部是采用FIR滤波器，具体可查阅百度或者论文，
    ref_power_cma = 2  # 设置CMA算法的模
    cma_convergence = 30000  # CMA预均衡收敛的信号长度
    step_size_cma = 1e-9  # CMA的更新步长，梯度下降法的步长
    step_size_rde = 1e-9  # RDE的更新步长，梯度下降法的步长  ， %% CMA和RDE主要就是损失函数不同

    equalizer = phot.AdaptiveEqualizer(num_tap, cma_convergence, ref_power_cma, step_size_cma, step_size_rde,
                                       up_sampling_factor, bits_per_symbol, total_baud)

    signal_x, signal_y = equalizer.equalize(signal_x, signal_y)

    """ 相位恢复  采用盲相位搜索算法（BPS）进行相位估计和补偿 """

    num_test_angle = 64  # BPS算法的测试角数目，具体算法原理可以参考函数内部给的参考文献
    block_size = 100  # BPS算法的块长设置

    bps = phot.BPS(num_test_angle, block_size, bits_per_symbol)
    signal_x, signal_y = bps.restore(signal_x, signal_y)

    """ 此处开始计算误码率 """

    ber = phot.BER(bits_per_symbol)
    ber.count(signal_x, signal_y, prev_symbols_x, prev_symbols_y)
