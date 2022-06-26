import phot
import math

if __name__ == '__main__':
    # Init global parameters
    phot.init_globals(num_sym=1024, num_pt=32, sym_rate=10)

    # Tx parameters (Tx refers to transmit)
    pow_dbm = 0  # power [dBm]
    mod_format = "qpsk"  # modulation format
    lam = 1550  # carrier wavelength [nm]
    tx_param = phot.TxParam(mod_format, "asin", 0.3)

    # Laser source parameter
    power_lin = math.pow(10, pow_dbm / 10)  # [mW]

    # Create a laser source object and generate lightwave
    laser_source = phot.LaserSource(power_lin, lam, 1)  # y-pol does not exist
    lightwave = laser_source.gen_light()

    # Generate random number sequence
    seq = phot.gen_seq("rand", mod_format, seed=1)

    # Init parameters of digit mod generate signal
    digit_mod = phot.DigitalModulator(mod_format, "rootrc", tx_param)

    # Return electric signal and normalization factor
    sig, norm_factor = digit_mod.modulate(seq)

    # Pass lightwave and signal to MZ modulator
    lightwave = phot.mz_modulator(lightwave, sig)

    # Create fiber and transmit lightwave
    fiber = phot.Fiber()
    lightwave, num_steps, first_dz = fiber.transmit(lightwave)

    # Rx parameters (Rx refers to receiver)
    rx_param = phot.RxParam(mod_format=mod_format, filter_type="gauss", bandwidth=math.inf)

    # Create frontend receiver and receive signal
    rx_frontend = phot.RxFrontend(lam=lam, rx_param=rx_param)
    elec_sig = rx_frontend.receive(lightwave)

    print(elec_sig)
