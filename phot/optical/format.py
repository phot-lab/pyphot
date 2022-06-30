class FormatInfo:
    def __init__(self, digit: int = None, family: str = None, sym_mean: int = None, sym_var: int = None):
        self.digit = digit
        self.family = family
        self.sym_mean = sym_mean
        self.sym_var = sym_var


def get_format_info(mod_format) -> FormatInfo:
    if mod_format == "qpsk" or mod_format == "dqpsk":
        format_info = FormatInfo(4, "psk", 0, 1)
    elif mod_format == "ook":
        format_info = FormatInfo(2, "ook", 1, 1)
    elif mod_format == "bpsk" or mod_format == "dpsk":
        format_info = FormatInfo(2, "psk", 0, 1)
    else:
        raise RuntimeError('Unknown modulation format')
    return format_info
