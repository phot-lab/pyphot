import numpy as np
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter
from scipy.signal import lfilter, resample_poly
import phot

if __name__ == '__main__':
    """ 本代码为程序主函数 本代码主要适用于 QPSK，16QAM，32QAM，64QAM 调制格式的单载波相干背靠背（B2B）信号 """

    # 设置全局系统仿真参数
    num_symbols = 2 ** 16  # 符号数目
    bits_per_symbol = 4  # 2 for QPSK;4 for 16QAM; 5 for 32QAM; 6 for 64QAM  设置调制格式
    total_baud = 64e9  # 信号波特率，符号率
    up_sampling_factor = 2  # 上采样倍数
    sampling_rate = up_sampling_factor * total_baud  # 信号采样率

    """ 程序从此开始为发射端的仿真代码 """

    # 首先产生发射端X/Y双偏振信号
    data_x = np.random.randint(0, 2, (num_symbols * bits_per_symbol, 1))  # 采用randint函数随机产生0 1码元序列x
    data_y = np.random.randint(0, 2, (num_symbols * bits_per_symbol, 1))  # 采用randint函数随机产生0 1码元序列y

    modem = QAMModem(2 ** bits_per_symbol)
    tx_symbols_x = modem.modulate(data_x).reshape((-1, 1))  # X偏振信号
    tx_symbols_y = modem.modulate(data_y).reshape((-1, 1))  # Y偏振信号

    scm_matrix_x = tx_symbols_x
    scm_matrix_y = tx_symbols_y

    tx_scm_matrix_x = scm_matrix_x  # 此处先存储发射端原始发送信号，作为最后比较BER
    tx_scm_matrix_y = scm_matrix_y

    """ RRC shaping   对信号进行RRC脉冲整形 """

    RRC_ROLL_OFF = 0.1  # RRC脉冲整形滚降系数

    # 先进行插0上采样，上采样倍数为 up_sampling_factor
    upsampled_matrix_x = phot.upsample(scm_matrix_x, up_sampling_factor)
    upsampled_matrix_y = phot.upsample(scm_matrix_y, up_sampling_factor)

    # 产生RRC滤波器
    time_idx, rrc_filter = rrcosfilter(N=128 * up_sampling_factor, alpha=RRC_ROLL_OFF, Ts=1 / total_baud,
                                       Fs=sampling_rate)  # up_sampling_factor*128为滤波器的长度
    rrc_filter = (rrc_filter * np.sqrt(2))

    # 对信号使用RRC滤波器脉冲整形
    rrc_matrix_x = lfilter(rrc_filter, 1, upsampled_matrix_x, axis=0)
    rrc_matrix_y = lfilter(rrc_filter, 1, upsampled_matrix_y, axis=0)

    """ 加入AWG中DAC的量化噪声 """

    sampling_rate_awg = 96e9  # DAC采样率
    dac_resolution_bits = 8  # DAC的bit位数

    # 重采样信号采样率为DAC的采样率，模拟进入DAC
    tx_samples_x = resample_poly(rrc_matrix_x, int(sampling_rate_awg), int(sampling_rate))
    tx_samples_y = resample_poly(rrc_matrix_y, int(sampling_rate_awg), int(sampling_rate))

    # 对信号进行量化，模拟加入量化噪声
    tx_samples_x_dac = phot.dac_resolution(tx_samples_x, dac_resolution_bits)
    tx_samples_y_dac = phot.dac_resolution(tx_samples_y, dac_resolution_bits)

    # 减去信号量化后的直流，也就是均值
    tx_samples_x_dac = tx_samples_x_dac - np.mean(tx_samples_x_dac)
    tx_samples_y_dac = tx_samples_y_dac - np.mean(tx_samples_y_dac)

    # 重采样信号采样率为原来的采样率，模拟出DAC后采样率，resample_poly 为SciPy库的函数
    tx_samples_x_dac = resample_poly(tx_samples_x_dac, int(sampling_rate), int(sampling_rate_awg)).reshape((-1, 1))
    tx_samples_y_dac = resample_poly(tx_samples_y_dac, int(sampling_rate), int(sampling_rate_awg)).reshape((-1, 1))

    # 采样率转换信号 = resample(经过DAC转换后的信号，进行下一步信号处理的采样率，DAC的采样率)

    """ 加入发射端激光器产生的相位噪声 """

    linewidth_tx = 150e3  # 激光器线宽
    phase_noise = phot.linewidth_induced_noise(len(tx_samples_x_dac), sampling_rate / total_baud, linewidth_tx,
                                               total_baud)  # 通过线宽计算相位噪声方差，产生随机的相位噪声

    # 添加相位噪声
    tx_samples_x_dac = tx_samples_x_dac * np.exp(1j * phase_noise)
    tx_samples_y_dac = tx_samples_y_dac * np.exp(1j * phase_noise)

    """ 根据设置的OSNR来加入高斯白噪声 """

    OSNR_dB = 25  # 设置系统OSNR，也就是光信号功率与噪声功率的比值，此处单位为dB

    # 生成均值为0，方差为1的随机噪声,此处直接产生两个偏振的噪声
    noise_x, noise_y, noise_power = phot.load_awgn(len(tx_samples_x_dac))

    # 计算当前信号功率
    original_avg_power = np.mean(np.square(np.abs(tx_samples_x_dac)) + np.square(np.abs(tx_samples_y_dac)))

    OSNR = np.power(10, OSNR_dB / 10)  # 将OSNR的单位由dB转为常量单位

    # 先计算OSNR对应的SNR，再通过SNR计算所需要达到的目标的信号功率，12.5e9为信号的中心频率，此处是根据公式计算，可参考通信原理或者百度或者论文
    target_avg_power = noise_power * OSNR * 12.5e9 / sampling_rate

    # 改变当前信号功率为目标功率
    tx_signal_x = np.sqrt(target_avg_power / original_avg_power) * tx_samples_x_dac
    tx_signal_y = np.sqrt(target_avg_power / original_avg_power) * tx_samples_y_dac

    # 对信号添加随机噪声
    tx_signal_x = noise_x + tx_signal_x
    tx_signal_y = noise_y + tx_signal_y

    """ 添加接收端激光器产生的相位噪声 """

    linewidth_rx = 150e3  # 激光器线宽

    # 通过线宽计算相位噪声方差，产生随机的相位噪声
    phase_noise = phot.linewidth_induced_noise(len(tx_signal_x), sampling_rate / total_baud, linewidth_rx, total_baud)

    # 添加相位噪声
    tx_signal_x = tx_signal_x * np.exp(1j * phase_noise)
    tx_signal_y = tx_signal_y * np.exp(1j * phase_noise)

    """
    发射端代码此处截止
    以下开始为接收端的代码
    """

    """ 添加收发端激光器造成的频偏，就是发射端激光器和接收端激光器的中心频率的偏移差 """

    frequency_offset = 2e9  # 设置频偏，一般激光器的频偏范围为 -3G~3G Hz

    # 2*pi*N*V*T   通过公式计算频偏造成的相位，N表示每个符号对应的序号，[1:length(TxSignal_X)]
    phase_carrier_offset = (
            np.arange(1, len(tx_signal_x) + 1).T * 2 * np.pi * frequency_offset / sampling_rate).reshape((-1, 1))

    # 添加频偏，频偏也可以看作一个相位
    tx_signal_x = tx_signal_x * np.exp(1j * phase_carrier_offset)
    tx_signal_y = tx_signal_y * np.exp(1j * phase_carrier_offset)

    """ 模拟接收机造成的I/Q失衡，主要考虑幅度失衡和相位失衡，这里将两者都加在虚部上 """

    rx_xi_tem = np.real(tx_signal_x)  # 取信号实部
    rx_xq_tem = np.imag(tx_signal_x)  # 取信号虚部
    rx_yi_tem = np.real(tx_signal_y)  # 取信号实部
    rx_yq_tem = np.imag(tx_signal_y)  # 取信号虚部

    amplitude_imbalance = np.power(10, 3 / 20)  # 10^(3/20)为幅度失衡因子
    phase_imbalance = np.pi * 80 / 180  # 80/180为相位失衡因子

    # 对虚部信号乘一个幅度失衡因子
    rx_xq_tem = rx_xq_tem * amplitude_imbalance
    rx_yq_tem = rx_yq_tem * amplitude_imbalance

    # 对虚部信号添加一个失衡相位,并将实部虚部组合为复数信号
    tx_signal_x = rx_xi_tem + np.exp(1j * phase_imbalance) * rx_xq_tem
    tx_signal_y = rx_yi_tem + np.exp(1j * phase_imbalance) * rx_yq_tem

    """ 加入ADC的量化噪声 """
    adc_sample_rate = 160e9  # ADC采样率
    adc_resolution_bits = 8  # ADC的bit位数

    # 重采样改变采样率为ADC采样率，模拟进入ADC
    re_x = resample_poly(tx_signal_x, int(adc_sample_rate), int(sampling_rate))
    re_y = resample_poly(tx_signal_y, int(adc_sample_rate), int(sampling_rate))

    # 对信号量化，添加ADC造成的量化噪声
    re_x = phot.adc_resolution(re_x, adc_resolution_bits)
    re_y = phot.adc_resolution(re_y, adc_resolution_bits)

    # 减去信号量化后的直流
    re_x = re_x - np.mean(re_x)
    re_y = re_y - np.mean(re_y)

    # 将信号采样率重采样为原来的采样率
    re_x = resample_poly(re_x, int(sampling_rate), int(adc_sample_rate))
    re_y = resample_poly(re_y, int(sampling_rate), int(adc_sample_rate))

    """ IQ正交化补偿，就是将之前的I/Q失衡的损伤补偿回来 """

    # 利用GSOP算法对I/Q失衡进行补偿，具体算法原理可看函数内部给的参考文献或论文
    rx_xi_tem, rx_xq_tem = phot.gram_schmidt_orthogonalize(np.real(re_x), np.imag(re_x))
    rx_yi_tem, rx_yq_tem = phot.gram_schmidt_orthogonalize(np.real(re_y), np.imag(re_y))

    # 对补偿后的实部和虚部信号进行重组
    re_x = rx_xi_tem + 1j * rx_xq_tem
    re_y = rx_yi_tem + 1j * rx_yq_tem

    """ 粗糙的频偏估计和补偿，先进行一个频偏的补偿，因为后面有一个帧同步，而帧同步之前需要先对频偏进行补偿，否则帧同步不正确 """

    # 利用FFT-FOE算法对信号的频偏进行估计与补偿
    re_x, re_y, fre_offset = phot.fre_offset_compensation_fft(re_x, re_y, sampling_rate)
    print('Estimated Coarse Frequency offset: {}'.format(fre_offset))

    """ 接收端相应的RRC脉冲整形，具体的参数代码与发射端的RRC滤波器是一致的 """

    re_x = lfilter(rrc_filter, 1, re_x, axis=0)
    re_y = lfilter(rrc_filter, 1, re_y, axis=0)

    """ 帧同步，寻找与发射端原始信号头部对应的符号 """

    # 对信号进行帧同步，找出接收信号与发射信号对准的开头
    start_index_x_1 = phot.fine_synchronize(re_x[0:10000 * up_sampling_factor:up_sampling_factor].T,
                                            tx_scm_matrix_x[0:2000].T)
    start_index_y_1 = phot.fine_synchronize(re_y[0:10000 * up_sampling_factor:up_sampling_factor].T,
                                            tx_scm_matrix_y[0:2000].T)

    print('两个偏振第一次对准的帧头')
    print('Start_Index_X_1: {} Start_Index_Y_1: {}'.format(start_index_x_1, start_index_y_1))

    # 将帧头位置前的信号去除，以便发射端信号与接收端信号的头部对准
    re_x = np.delete(re_x, np.arange(0, start_index_x_1 * up_sampling_factor))
    re_y = np.delete(re_y, np.arange(0, start_index_y_1 * up_sampling_factor))

    # 去掉接收信号的尾部部分信号
    re_x = re_x[:-1000]
    re_y = re_y[:-1000]

    # 调整发射端信号与接收端信号的长度
    tx_scm_matrix_x = tx_scm_matrix_x[0:int(np.floor(len(re_x) / up_sampling_factor))]
    tx_scm_matrix_y = tx_scm_matrix_y[0:int(np.floor(len(re_y) / up_sampling_factor))]

    """ 自适应均衡，此处采用恒模算法（CMA）对收敛系数进行预收敛，再拿收敛后的滤波器系数对正式的信号使用半径定向算法（RDE）进行均衡收敛，总的思想采用梯度下降法 """

    num_tap = 25  # 均衡器抽头数目，此处均衡器内部是采用FIR滤波器，具体可查阅百度或者论文，
    ref_power_cma = 2  # 设置CMA算法的模
    cma_convergence = 30000  # CMA预均衡收敛的信号长度
    step_size_cma = 1e-9  # CMA的更新步长，梯度下降法的步长
    step_size_rde = 1e-9  # RDE的更新步长，梯度下降法的步长  ， %% CMA和RDE主要就是损失函数不同

    input_x_i = np.real(re_x)  # 求出接收端X偏振的实部信号
    input_x_q = np.imag(re_x)  # 求出接收端X偏振的虚部信号
    input_y_i = np.real(re_y)  # 求出接收端Y偏振的实部信号
    input_y_q = np.imag(re_y)  # 求出接收端Y偏振的实部信号

    # 对信号采用CMA-RDE进行自适应均衡
    equalization_matrix_x, equalization_matrix_y = phot.cma_rde(input_x_i, input_x_q, input_y_i, input_y_q, num_tap,
                                                                cma_convergence, ref_power_cma, step_size_cma,
                                                                step_size_rde, up_sampling_factor, bits_per_symbol)

    phot.utils.plot_scatter(equalization_matrix_x, pt_size=1)

    # 此处均衡器内部存在一个下采样，因此均衡器出来后信号回到一个符号一个样本的采样率，也就是现在的采样率等于符号率

    """ 均衡后进行精确的频偏估计和补偿 采用FFT-FOE算法，与前面的粗估计一样，防止前面粗估计没补偿完全，此处做一个补充 """

    # 利用FFT-FOE算法对信号的频偏进行估计与补偿
    equalization_matrix_x, equalization_matrix_y, fre_offset = phot.fre_offset_compensation_fft(equalization_matrix_x,
                                                                                                equalization_matrix_y,
                                                                                                total_baud)
    print('Estimated Accurate Frequency offset: {}'.format(fre_offset))

    """ 相位恢复  采用盲相位搜索算法（BPS）进行相位估计和补偿 """

    num_test_angle = 64  # BPS算法的测试角数目，具体算法原理可以参考函数内部给的参考文献
    block_size = 100  # BPS算法的块长设置

    # BPS算法
    equalization_matrix_x, phase_x = phot.bps_hybrid_qam(np.real(equalization_matrix_x), np.imag(equalization_matrix_x),
                                                         num_test_angle, block_size, bits_per_symbol)
    equalization_matrix_y, phase_y = phot.bps_hybrid_qam(np.real(equalization_matrix_y), np.imag(equalization_matrix_y),
                                                         num_test_angle, block_size, bits_per_symbol)
    """ normalization 对信号进行归一化，代码过程中存在一个信号的缩放问题，此处将信号恢复回去 """
    if bits_per_symbol == 2:
        equalization_matrix_x = equalization_matrix_x * np.sqrt(2)  # 不同调制格式对应的归一化因子不同
        equalization_matrix_y = equalization_matrix_y * np.sqrt(2)
    elif bits_per_symbol == 4:
        equalization_matrix_x = equalization_matrix_x * np.sqrt(10)
        equalization_matrix_y = equalization_matrix_y * np.sqrt(10)
    elif bits_per_symbol == 5:
        equalization_matrix_x = equalization_matrix_x * np.sqrt(20)
        equalization_matrix_y = equalization_matrix_y * np.sqrt(20)

    phot.utils.plot_scatter(equalization_matrix_x)

    """ 此处开始计算误码率 """

    """ 再进行一个帧同步，因为经过均衡器会存在符号的一些舍弃，因此在计算误码率（BER）之前需要再一次帧同步 """

    # 对发射端信号跟均衡后信号进行同步
    start_index_x_1 = phot.fine_synchronize(tx_scm_matrix_x[:, 0].T, equalization_matrix_x[0:10000, 0].reshape((1, -1)))

    # 对发射端信号利用自带函数进行移动
    tx_scm_matrix_x = np.roll(tx_scm_matrix_x, -start_index_x_1)
    tx_scm_matrix_y = np.roll(tx_scm_matrix_y, -start_index_x_1)

    # 变为16的倍数
    equalization_matrix_x = equalization_matrix_x[:-1]
    equalization_matrix_y = equalization_matrix_y[:-1]

    # 使得均衡后信号与发射端信号一样长度
    tx_scm_matrix_x = tx_scm_matrix_x[0:len(equalization_matrix_x), :]
    tx_scm_matrix_y = tx_scm_matrix_y[0:len(equalization_matrix_y), :]

    """ BER COUNT  对信号进行误码率计算，将接收信号与发射信号转化为格雷编码，比较各个码元的正确率 """

    tx_bits = modem.demodulate(
        np.concatenate((np.reshape(tx_scm_matrix_x, (-1, 1)), np.reshape(tx_scm_matrix_y, (-1, 1))), axis=0).ravel(),
        demod_type='hard')
    rx_bits = modem.demodulate(
        np.concatenate((np.reshape(equalization_matrix_x, (-1, 1)), np.reshape(equalization_matrix_y, (-1, 1))),
                       axis=0).ravel(), demod_type='hard')
    ber, q_db = phot.ber_count(rx_bits, tx_bits)  # 比较码元的正确率
    print('Calculated overall bits error is {:.5f}'.format(ber))
