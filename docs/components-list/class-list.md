# 类清单

| 类构造方法/成员方法  | 参数               | 类型  | 输入信号  | 类型       | 输出信号 | 类型       |
| -------------------- | ------------------ | ----- | --------- | ---------- | -------- | ---------- |
| PulseShaper          | up_sampling_factor | int   |           |            |          |            |
|                      | len_filter         | int   |           |            |          |            |
|                      | alpha              | float |           |            |          |            |
|                      | ts                 | float |           |            |          |            |
|                      | fs                 | float |           |            |          |            |
| PulseShaper.tx_shape |                    |       | symbols_x | np.ndarray | signal_x | np.ndarray |
|                      |                    |       | symbols_y | np.ndarray | signal_y | np.ndarray |
| PulseShaper.rx_shape |                    |       | signal_x  | np.ndarray | signal_x | np.ndarray |
|                      |                    |       | signal_y  | np.ndarray | signal_y | np.ndarray |
