import math
from deprecated import phot

if __name__ == '__main__':
    # Init global parameters
    phot.init_globals(num_sym=1024, num_pt=32, sym_rate=10)

    # Tx parameters (Tx refers to transmit)
    mod_format = 'qpsk'  # modulation format
    lam = 1550  # carrier wavelength [nm]
    rolloff = 0.3  # pulse roll-off

    # Create a laser source object and generate lightwave
    laser_source = phot.LaserSource(power=1, lam=lam, num_pol=1)  # y-pol does not exist
    lightwave = laser_source.gen_light()

    # Generate random number sequence
    seq = phot.gen_seq('rand', mod_format)

    # Digital modulator
    digit_mod = phot.DigitalModulator(mod_format=mod_format, pulse_type='rootrc', rolloff=rolloff)
    signal, norm_factor = digit_mod.modulate(seq)

    # Electric Amplifier at Tx side
    signal = phot.elec_amp(signal, gain_ea=5, power=1, spec_density=10.0e-12)

    # Pass lightwave and signal to MZ modulator
    lightwave = phot.mz_modulator(lightwave, signal)

    # Create fiber and transmit lightwave
    fiber = phot.Fiber(length=10000, lam=lam, alpha_b=1, dispersion=17, n2=0)
    lightwave, num_steps, first_dz = fiber.transmit(lightwave)

    # Create frontend receiver and receive signal
    rx_frontend = phot.RxFrontend(filter_type='gauss', lam=lam, mod_format=mod_format, bandwidth=math.inf)
    elec_sig = rx_frontend.receive(lightwave)

    # Electric Amplifier at Rx side
    elec_sig = phot.elec_amp(elec_sig, gain_ea=5, power=1, spec_density=10.0e-12)

    # Create analyzer and evaluate eye opening value
    analyzer = phot.Analyzer(mod_format=mod_format)
    eye_opening = analyzer.eval_eye(seq, elec_sig, plot_eye=True)

    print('Eye opening: {:.4f} [mw]'.format(eye_opening))

    rx_dsp = phot.RxDsp('gauss', 0.25, 'da', mod_format, seq, elec_param=rolloff)
    symbols, gain_factor = rx_dsp.process(elec_sig)

    phot.plot_constell(symbols)
