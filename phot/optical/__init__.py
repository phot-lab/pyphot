"""
Copyright (c) 2022 Beijing Jiaotong University
PhotLab is licensed under [Open Source License].
You can use this software according to the terms and conditions of the [Open Source License].
You may obtain a copy of [Open Source License] at: [https://open.source.license/]

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.

See the [Open Source License] for more details.

Author: Chunyu Li
Created: 2022/9/6
Supported by: National Key Research and Development Program of China
"""

from .resolution import dac_resolution, adc_resolution
from .linewidth_noise import linewidth_induced_noise
from .awgn import load_awgn
from .orthog import gram_schmidt_orthogonalize
from .offset_compens import fre_offset_compensation_fft
from .fine_sync import fine_synchronize
from .cma_rde import cma_rde
from .bps import bps_hybrid_qam
from .bit_error import ber_count, ber_estimate
from .fiber import optical_fiber_channel
from .dbp_compensate import dbp_compensate
from .cdc import cdc
