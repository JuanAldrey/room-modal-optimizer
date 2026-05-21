import numpy as np

class Evaluator:
    
    @staticmethod
    def evaluate(f_modes, n):
        """
        Frequency Spacing Index
        Inputs:
            - f_modes: array type object. modes frequencies vector
            - n: int type object. Max mode to evaluate.
        """
        
        f = f_modes[:n]
        avg_frq_sp = (f[-1]-f[0])/(n-1)
        dif_nhb_modes = np.diff(f)
        sum_arg = (dif_nhb_modes/avg_frq_sp)**2
        frq_sp_idx = (1/(n-1))*np.sum(sum_arg, axis=0)
        return frq_sp_idx