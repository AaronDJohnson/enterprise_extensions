from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

class FeStat(object):
    
    def __init__(self, psrs, params=None):

        print('Initializing the model...')

        efac = parameter.Constant()
        equad = parameter.Constant()
        ef = white_signals.MeasurementNoise(efac=efac)
        eq = white_signals.EquadNoise(log10_equad=equad)

        tm = gp_signals.TimingModel(use_svd=True)

        s = eq + ef + tm

        model = []
        for p in psrs:
            model.append(s(p))
        self.pta = signal_base.PTA(model)

        # set white noise parameters
        if params is None:
            print('No noise dictionary provided!...')
        else:
            self.pta.set_default_params(params)

        self.psrs = psrs
        self.params = params

        self.Nmats = None
