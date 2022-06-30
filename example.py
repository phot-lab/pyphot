import phot
import math

if __name__ == '__main__':
    # Init global parameters
    phot.init_globals(num_sym=1024, num_pt=32, sym_rate=10)

    # Tx parameters (Tx refers to transmit)
    power = 1.0  # power in linear scale [mW]
    mod_format = "qpsk"  # modulation format
    lam = 1550  # carrier wavelength [nm]
    tx_param = phot.TxParam(mod_format, "asin", 0.3)

    # Create a laser source object and generate lightwave
    laser_source = phot.LaserSource(power, lam, 1)  # y-pol does not exist
    lightwave = laser_source.gen_light()

    # Generate random number sequence
    seq = phot.gen_seq("rand", mod_format, seed=1)

    # Digital modulator
    digit_mod = phot.DigitalModulator(mod_format, "rootrc", tx_param)
    signal, norm_factor = digit_mod.modulate(seq)

    # Electric Amplifier at Tx side
    # signal = phot.elec_amp(signal, gain_ea=5, power=1, spec_density=10.0e-12)

    # Pass lightwave and signal to MZ modulator
    lightwave = phot.mz_modulator(lightwave, signal)

    # Create fiber and transmit lightwave
    fiber = phot.Fiber(length=10000, lam=lam, alpha_b=1, dispersion=17, n2=0)
    lightwave, num_steps, first_dz = fiber.transmit(lightwave)

    # Rx parameters (Rx refers to receiver)
    rx_param = phot.RxParam(mod_format=mod_format, filter_type="gauss", bandwidth=math.inf)

    # Create frontend receiver and receive signal
    rx_frontend = phot.RxFrontend(lam=lam, rx_param=rx_param)
    elec_sig = rx_frontend.receive(lightwave)

    # Electric Amplifier at Rx side
    # elec_sig = phot.elec_amp(elec_sig, gain_ea=5, power=1, spec_density=10.0e-12)

    # Create analyzer and evaluate eye opening value
    analyzer = phot.Analyzer(mod_format=mod_format)
    eye_opening = analyzer.eval_eye(seq, elec_sig, plot_eye=True)

    print("Eye opening: {:.4f} [mw]".format(eye_opening))
