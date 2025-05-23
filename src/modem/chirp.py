import numpy as np
import scipy.signal

from .constants import FS
from .wav import generate_wav

CHIRP_DURATION = 1.0
CHIRP_F0 = 500
CHIRP_F1 = 4000

CHIRP_TIMES = np.linspace(0, CHIRP_DURATION, int(CHIRP_DURATION * FS), endpoint=False)

# Use upward chirp for start
START_CHIRP = scipy.signal.chirp(CHIRP_TIMES, f0=CHIRP_F0, f1=CHIRP_F1, t1=CHIRP_DURATION, method='linear')

# Use downward chirp for end
END_CHIRP = scipy.signal.chirp(CHIRP_TIMES, f0=CHIRP_F1, f1=CHIRP_F0, t1=CHIRP_DURATION, method='linear')
