import numpy as np


class Lightwave:
    """
    The lightware is a form of carrier wave that is modulated to carry information.
    """

    def __init__(self,
                 lam: int = 1550,
                 field: np.ndarray = None
                 ) -> None:
        """
        :param lam: wavelengths [nm]
        :param field: time samples of the electric field along rows
        """
        self.lam = lam
        self.field = field

    def __repr__(self):
        return 'Wavelength: {} [nm]\nField ({} x {}):\n{}'.format(self.lam, self.field.shape[0], self.field.shape[1],
                                                                  self.field)
