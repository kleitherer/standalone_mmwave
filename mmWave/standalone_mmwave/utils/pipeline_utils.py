import numpy as np

from utils.radar_capture_utils import DCA1000

# functions
def pw2db(x, scale=10):
    return scale*np.log10(np.abs(x)+1e-9)

# classes
class radarDataLoader:
    '''
        Load radar data from npz file
    '''

    def __init__(self, file_path, radar_params):
        self.file_path = file_path
        self.params = radar_params
    
    def load_data(self):
        data_npz = np.load(self.file_path)

        n_chirps = int(self.params['n_chirps'])
        n_samples = int(self.params['n_samples'])
        range_res = float(self.params['range_res'])
        velocity_res = float(self.params['velocity_res'])
        range_axis = np.arange(n_samples) * range_res
        doppler_axis = np.arange(-n_chirps // 2, n_chirps // 2) * velocity_res

        # UDP / .npy packed files (e.g. testing_gesture_osc_packed.npz) include a ready-made
        # 5D cube. Beckman / bigdata files only have radar_data (raw DCA wire) and need decode.
        if 'radar_cube' in data_npz.files:
            radar_frame = np.asarray(data_npz['radar_cube'])
            pcloud_list = []
            n_f = radar_frame.shape[0]
            num_frames = {
                'decoded_frames': n_f,
                'chirp_frames': n_f,
                'not_chirp_frames': 0,
                'varying_chirp_frames': [],
                'resulted_frames': n_f,
                'source': 'radar_cube',
            }
            info = {'frame_header': None, 'chirp header': None}
            return radar_frame, pcloud_list, info, num_frames, range_axis, doppler_axis

        # Beckman path: 1D int16 wire stream -> 5D complex cube via DCA1000.decode_data
        radar_data_uint8 = data_npz['radar_data']
        num_Rx = int(self.params['n_rx'])
        num_Tx = int(self.params['n_tx'])
        radar_frame, pcloud_list, info, num_frames = DCA1000.decode_data(
            radar_data_uint8, num_Tx=num_Tx, num_Rx=num_Rx
        )
        return radar_frame, pcloud_list, info, num_frames, range_axis, doppler_axis



def RD(radar_cube, declutter=True, window=True, r_axis=None):
    '''
        Perform range and doppler FFT on radar_cube data
        Input:
        - radar_cube: radar data in time domain
            - shape: (Nframes, Nchirps, Ntx, Nrx, Nsamples)
        - declutter: remove dc component or not
        - window: apply windowing or not
        Output:
        - RD: radar data after range and doppler FFT
            - shape: (Nframes, Nchirps, Ntx*Nrx, Nsamples)
    '''
    rda = radar_cube
    if declutter:
        rda -= np.expand_dims(rda.mean(axis=1), axis=1) # remove dc clutter
    if window:
        window_1d_range = np.hanning(rda.shape[-1])
        window_1d_doppler = np.hanning(rda.shape[1])
        rda = rda * window_1d_range[None, None, None, None, :]
        rda = rda * window_1d_doppler[None, :, None, None, None]

    RDa = np.fft.fftshift(np.fft.fft(rda, axis=1), axes=1)
    RDa = np.fft.fft(RDa, axis=4)

    # from (Nf, Nc, Ntx, Nrx, Nsamples) to (Nf, Nc, Ntx*Nrx, Nsamples)
    RDa = RDa.reshape(RDa.shape[0], RDa.shape[1], RDa.shape[2]*RDa.shape[3], RDa.shape[4])
    # RDa = RDa[..., :8, :] # only keep the first 8 pairs
    print(f"RDa shape: {RDa.shape}")
    noise_floor_db = 10*np.log10(np.mean(np.abs(RDa.mean(axis = 2))**2))
    noise_range_db = 10*np.log10(np.mean(np.abs(RDa.mean(axis = 2))**2, axis=(0,1)))

    return RDa, noise_floor_db, noise_range_db

