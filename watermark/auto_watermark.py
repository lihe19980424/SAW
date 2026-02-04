# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated 
#              as one of the watermark classes of the library when created 
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import importlib

WATERMARK_MAPPING_NAMES={
    'KGW': 'watermark.kgw.KGW',
    'Unigram': 'watermark.unigram.Unigram',
    'SWEET': 'watermark.sweet.SWEET',
    'UPV': 'watermark.upv.UPV',
    'SIR': 'watermark.sir.SIR',
    'XSIR': 'watermark.xsir.XSIR',
    'EWD': 'watermark.ewd.EWD',
    'EXP': 'watermark.exp.EXP',
    'EXPEdit': 'watermark.exp_edit.EXPEdit',
    'DIP': 'watermark.dip.DIP',
    'TS': 'watermark.ts.TS',
    'SynthID': 'watermark.synthid.SynthID',
    
    # add more watermark algorithms here by wang 
    "KGW_plus_UNI":"watermark.kgw_plus_uni.KGW_plus_UNI",
    
    # add more watermark algorithms here by lihe
    "Black_Box":"watermark.black_box.Black_Box",
    # add more watermark algorithms here by lihe
    'Unbiased': 'watermark.unbiased.UnbiasedWatermark',
    # add more watermark algorithms here by lihe
    "White_Box":"watermark.white_box.White_Box",
    # add more watermark algorithms here by lihe
    "Rethink":"watermark.rethink.Rethink",
    # add more watermark algorithms here by lihe
    "Rethinking":"watermark.rethinking.Rethinking",
    "Rethinking_uni":"watermark.rethinking_uni.Rethinking",
    "Rethinking_gaosi":"watermark.rethinking_gaosi.Rethinking",
    "SAW":"watermark.saw.SAW",
    "SMOOTH":"watermark.smooth.SMOOTH",
    'ColorMark': 'watermark.colormark.ColorMark'
}

def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    for algorithm_name, watermark_name in WATERMARK_MAPPING_NAMES.items():
        if name == algorithm_name:
            return watermark_name
    return None

class AutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config,private_key,context_code_extractor,ignore_history)` method."
        )

    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name) #'watermark.kgw.black_box.Black_Box'
        module_name, class_name = watermark_name.rsplit('.', 1) #watermark.kgw.black_box Black_Box
        module = importlib.import_module(module_name) #watermark.kgw.black_box
        watermark_class = getattr(module, class_name) #watermark.kgw.black_box.Black_Box
        watermark_instance = watermark_class(algorithm_config, transformers_config)
        return watermark_instance

