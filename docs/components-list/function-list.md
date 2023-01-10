# 函数清单

| 函数名                          | 参数                | 类型  | 输入信号       | 类型       | 输出信号       | 类型       |
| ------------------------------- | ------------------- | ----- | -------------- | ---------- | -------------- | ---------- |
| qam_modulate                    | bits_per_symbol     | int   | data_x         | np.ndarray | symbols_x      | np.ndarray |
|                                 |                     |       | data_y         | np.ndarray | symbols_y      | np.ndarray |
| dac_noise                       | sampling_rate_awg   | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | sampling_rate       | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
| phase_noise                     | over_sampling_rate  | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | linewidth_tx        | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 | total_baud          | float |                |            |                |            |
| gaussian_noise                  | osnr_db             | int   | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | sampling_rate       | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
| optical_fiber_channel           | sampling_rate       | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | span                | int   | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 | num_steps           | int   |                |            | power_x        | np.ndarray |
|                                 | beta2               | float |                |            | power_y        | np.ndarray |
|                                 | delta_z             | int   |                |            |                |            |
|                                 | gamma               | float |                |            |                |            |
|                                 | alpha               | float |                |            |                |            |
|                                 | L                   | int   |                |            |                |            |
| add_freq_offset                 | frequency_offset    | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | sampling_rate       | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
| add_iq_imbalance                |                     |       | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 |                     |       | signal_y       | np.ndarray | signal_y       | np.ndarray |
| add_adc_noise                   | sampling_rate       | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | adc_sample_rate     | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 | adc_resolution_bits | int   |                |            |                |            |
| iq_freq_offset_and_compensation | sampling_rate       | float | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | beta2               | float | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 | span                | int   | power_x        | np.ndarray |                |            |
|                                 | L                   | int   | power_y        | np.ndarray |                |            |
| sync_frame                      | up_sampling_factor  | int   | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 |                     |       | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 |                     |       | prev_symbols_x | np.ndarray | prev_symbols_x | np.ndarray |
|                                 |                     |       | prev_symbols_y | np.ndarray | prev_symbols_y | np.ndarray |
| adaptive_equalize               | num_tap             | int   | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | cma_convergence     | int   | signal_y       | np.ndarray | signal_y       | np.ndarray |
|                                 | ref_power_cma       | int   |                |            |                |            |
|                                 | step_size_cma       | float |                |            |                |            |
|                                 | step_size_rde       | float |                |            |                |            |
|                                 | up_sampling_factor  | int   |                |            |                |            |
|                                 | bits_per_symbol     | int   |                |            |                |            |
|                                 | total_baud          | float |                |            |                |            |
| bps_restore                     | num_test_angle      | int   | signal_x       | np.ndarray | signal_x       | np.ndarray |
|                                 | block_size          | int   | signal_y       | np.ndarray | signal_x       | np.ndarray |
|                                 | bits_per_symbol     | int   |                |            |                |            |
| bits_error_count                | bits_per_symbol     | int   | signal_x       | np.ndarray |                |            |
|                                 |                     |       | signal_y       | np.ndarray |                |            |
|                                 |                     |       | prev_symbols_x | np.ndarray |                |            |
|                                 |                     |       | prev_symbols_y | np.ndarray |                |            |

