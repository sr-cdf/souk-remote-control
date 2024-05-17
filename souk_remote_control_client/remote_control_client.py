import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import numpy as np
import socket
import time
import os
import subprocess
from subprocess import Popen, PIPE
import pickle
import struct

import souk_mkid_readout


def get_interface_ip(target_address):
    """Connects to <target_address> and returns the ip address of the local socket"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(target_address)
    ip_addr = s.getsockname()[0]
    s.close()
    return ip_addr

def empty_socket_buffer(sock,printing=True):
    """Pulls all data out of a socket until none remains
    Should be called immediately prior to logging a stream """
    timeout=sock.gettimeout()
    sock.settimeout(0)
    n=0
    while True:
        try:
            n+=len(sock.recv(65536))
            if printing:
                print(n)
        except BlockingIOError:
            break
    sock.settimeout(timeout)
    if printing:
        if n==0:
            print('empty_socket_buffer: socket is already empty.')
        else:
            print(f'empty_socket_buffer: socket has been emptied of {n} bytes.')
    return

REMOTE_HOST     = 'krm4'
REMOTE_PORT     = 12345
REMOTE_USER     = 'casper'
# REMOTE_SCRIPT   = '~/src/souk-remote-control/souk_remote_control_server/remote_control_start'
REMOTE_SCRIPT   = '/home/casper/src/souk-remote-control/souk_remote_control_server/remote_control'
REMOTE_CONFIG   = '/home/sam/souk/souk-firmware/software/control_sw/config/souk-single-pipeline-krm-2G.yaml'
REMOTE_TESTPORT = 7147 #so we can temporarily connect to casper borph to find which local interface to bind the socket to.

#define the address of the receiver (destination of the packets)
DEST_IP         = get_interface_ip((REMOTE_HOST, REMOTE_TESTPORT))
DEST_PORT       = 10000
DEST_ADDR       = (DEST_IP,DEST_PORT)

# ADCCLK          = 2457600000 #2.4576 GHz
# ADCCLK          = 2048000000
PFBLEN          = 4096
NCHANS          = 2048 #number of tones in a single acquisition
ACCNUM          = 0 #which accumulation to read from

#define the packet datatype with a name and format, a single acquisition is transmitted over multiple packets

NAME_TIMESTAMP   = 'timestamp'
NAME_ERROR       = 'error'
NAME_INDEX       = 'index'
NAME_PAYLOAD     = 'payload'

FMT_TIMESTAMP    = '>u8'
FMT_ERROR        = '>u4'
FMT_INDEX        = '>u4'
FMT_PAYLOAD      = '<i4' # data was read directly from onboard memory and ARM is little-endian

PAYLOAD_BYTES    = 1024
PAYLOAD_ITEMS    = PAYLOAD_BYTES // (np.dtype(FMT_PAYLOAD).itemsize)
PAYLOAD_CHANS    = PAYLOAD_ITEMS // 2 # one for I and one for Q

PACKET_DTYPE     = np.dtype([('packet_header',[(NAME_TIMESTAMP, FMT_TIMESTAMP),
                                               (NAME_ERROR,     FMT_ERROR    ),
                                               (NAME_INDEX,     FMT_INDEX    )] ),
                             (NAME_PAYLOAD, FMT_PAYLOAD, PAYLOAD_ITEMS) ] )

#for convenience, define the offset of each field in the packet
OFFSET_TIMESTAMP = (PACKET_DTYPE['packet_header'].fields)[NAME_TIMESTAMP][1]
OFFSET_ERROR     = (PACKET_DTYPE['packet_header'].fields)[NAME_ERROR][1]
OFFSET_INDEX     = (PACKET_DTYPE['packet_header'].fields)[NAME_INDEX][1]
OFFSET_PAYLOAD   = (PACKET_DTYPE.fields)[NAME_PAYLOAD][1]
PACKET_BYTES     = PACKET_DTYPE.itemsize
PACKETS_PER_ACC  = 2*NCHANS // PAYLOAD_ITEMS

#setup socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(DEST_ADDR)

#create buffer to hold the incoming packets for a single accumulation
inbuf = bytearray(PACKET_BYTES)

#store all packets for a single accumulation here
PACKETS = np.zeros(PACKETS_PER_ACC, dtype=PACKET_DTYPE)


#define a data type for a single acquisition, to be filled in from individual packets
ACC_DTYPE = np.dtype([('timestamp',FMT_TIMESTAMP),
                      ('acc_error',FMT_ERROR),
                      ('i',FMT_PAYLOAD,NCHANS),
                      ('q',FMT_PAYLOAD,NCHANS)])



#functions to control the server on the krm board 


def remote_quit(host=REMOTE_HOST,port=REMOTE_PORT):
    """function to kill the remote control server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            s.send(b"exit 0")
            print('Successfully sent "x" to quit')
        except Exception as e:
            print('Failed to send quit')
            raise(e)
    time.sleep(1)
    return

def remote_start(host=REMOTE_HOST,port=REMOTE_PORT,waitstart=15):
    """
    function to start the remote control server
    """
    remote_command = 'ssh'
    remote_args = ['%s@%s'%(REMOTE_USER,REMOTE_HOST), '-tt', REMOTE_SCRIPT]
    process = Popen([remote_command, *remote_args],stdin=subprocess.DEVNULL)
    time.sleep(waitstart)
    return

def remote_program_and_initialise(host=REMOTE_HOST,port=REMOTE_PORT):
    """ function to program and initialise the remote control server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            cmd="prog 0"
            s.connect((host, port))
            s.send(cmd.encode())
            print(f'Successfully sent "{cmd}"')
        except Exception as e:
            print(f'Failed to send "{cmd}"')
            raise(e)
    return

def remote_stream(naccs,host=REMOTE_HOST,port=REMOTE_PORT):
    """function to tell the remote control server to start streaming"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            arg = pickle.dumps({'naccs':naccs})
            cmd = ("stream %d "%(len(arg))).encode()
            s.connect((host, port))
            s.send(cmd + arg)
            print(f'Successfully sent "{cmd}"')

        except Exception as e:
            print(f'Failed to send "{cmd}"')
            raise(e)
    return

def remote_sweep(freqs,spans,numpoints,samples_per_point,direction='up',amplitudes=None,phases=None,host=REMOTE_HOST,port=REMOTE_PORT):
    print('Not Implemented, try "sweep_v3" for now.')
    return

# def remote_set_freqs(freqs,host=REMOTE_HOST,port=REMOTE_PORT):
#     """function to tell the remote control server to do a fast mixer frequency setting"""
#     freqs = np.atleast_1d(freqs)
#     numfreqs=len(freqs)

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         try:
#             cmd="freqs %d"%naccs
#             s.connect((host, port))
#             s.send(cmd.encode())
#             print(f'Successfully sent "{cmd}"')
#         except Exception as e:
#             print(f'Failed to send "{cmd}"')
#             raise(e)
#     return

def sweep_v1(r,centerfreqs,span,numpoints,samples_per_point, direction='up',amplitudes=None,phases=None,ret_samples=False,sleeptime=0.0):
    """
    perform a sweep by repeated calling r.set_multi_tone(), and get_n_accs()

    r is an souk_mkid_readout object
    centerfreq is a numpy array of center frequencies in Hz
    span is the sweep span in Hz
    numpoints is an integer number of points to sweep
    samplesperpoint is an integer number of samples to collect and average at each point
    direction is a string, either 'up' or 'down'
    amplitude is an optional numpy array of amplitudes for each tone, defaults to all 1.0
    phases is an optional numpy array of phases for each tone, defaults to all 0.0

    returns a dictionary of the results
    """
    numtones = len(centerfreqs)
    freqs=centerfreqs
    sz = np.zeros((numpoints,numtones),dtype=complex)
    ez = np.zeros((numpoints,numtones),dtype=complex)              
    offsets = np.linspace(-span/2,span/2,numpoints)
    if direction =='up':
        pass
    elif direction == 'down':
        offsets = offsets[::-1]
    else:
        raise ValueError('direction should be "up" or "down", not "%s".'%direction)

    sweepfreqs = np.zeros((numpoints,numtones),dtype=float)
    
    #samples=None
    if ret_samples:
        assert(numpoints*len(centerfreqs)*samples_per_point<100e6)
        samples = np.zeros((numpoints,numtones,samples_per_point),dtype=complex)
    else:
        samples=None
    
    for j in range(len(offsets)):
        r.set_multi_tone(freqs+offsets[j],amplitudes=amplitudes,phase_offsets_rads=phases)
        time.sleep(0.1)
        r.sync.arm_sync(wait=False)
        time.sleep(0.1)
        r.sync.sw_sync()

        time.sleep(sleeptime)
        accs,accfreq = get_n_accs(r,samples_per_point)
        for k in range(len(freqs)):
            if ret_samples:
                samples[j,k] = accs['i'][:,k] + 1j*accs['q'][:,k]
            z=np.mean(accs['i'][:,k])+1j*np.mean(accs['q'][:,k])
            e=np.std(accs['i'][:,k])+1j*np.std(accs['q'][:,k])
            sz[j,k]=z
            ez[j,k]=e
            sweepfreqs[j,k] = freqs[k]+offsets[j]

    result = {'centerfreqs':centerfreqs,
              'span':span,
              'numpoints':numpoints,
              'samples_per_point':samples_per_point,
              'direction':direction,
              'amplitudes':amplitudes,
              'phases':phases,
              'numtones':numtones,
              'sweepfreqs':sweepfreqs, 
              'sweep':sz,
              'noise':ez,
              'accfreq':accfreq,
              'time_per_point':samples_per_point/accfreq,
              'fpga_status':r.fpga.get_status(),
              'samples':samples}
    return result


def sweep_v2(r,centerfreqs,spans,numpoints,samples_per_point,direction='up',amplitudes=None,phases=None):
    """
    Perform a frequency sweep using lower level calls than v1..
    
    centerfreqs is a numpy array of center frequencies in Hz
    spans is a numpy array of spans in Hz
    numpoints is an integer number of points to sweep
    samplesperpoint is an integer number of samples to collect and average at each point
    direction is a string, either 'up' or 'down'
    amplitudes is an optional numpy array of amplitudes for each tone, defaults to all 1.0
    phases is an optional numpy array of phases for each tone, defaults to all 0.0

    During the sweep, when a tone reaches the edge of a filterbank channel, it is remapped to the nearest channel,


    """

    #check inputs
    print('check inputs')

    centerfreqs = np.atleast_1d(centerfreqs)
    spans       = np.atleast_1d(spans)
    assert len(centerfreqs) == len(spans)
    numpoints=int(numpoints)
    assert samples_per_point>0 and samples_per_point%1==0
    assert direction in ('up','down')
    
    #explicitly define number of tones, and parallel/serial inputs
    print('define number of tones, and parallel/serial inputs')
    numtones = len(centerfreqs)
    chans    = np.arange(numtones,dtype=int)
    parallel = chans % r.mixer._n_parallel_chans
    serial   = chans // r.mixer._n_parallel_chans
    

    #define the frequency sweep points for each tone
    print('define the frequency sweep points for each tone')
    sweepfreqs = np.zeros((numpoints,numtones))
    for t in range(numtones):
        cf=centerfreqs[t]
        sp=spans[t]
        sweepfreqs[:,t] = np.linspace(cf-sp/2.,cf+sp/2.,numpoints)
        if direction=='down':
            sweepfreqs[:,t] = sweepfreqs[:,t][::-1]
    

    #define the channel maps and mixer offset frequencies for each sweep point
    print('define the channel maps and mixer offset frequencies for each sweep point')
    chanmap_in  = -1*np.ones((numpoints,r.chanselect.n_chans_out),dtype=np.int32)
    chanmap_out = -1*np.ones((numpoints,r.psb_chanselect.n_chans_out),dtype=np.int32)
    lo_freqs    = np.zeros((numpoints,numtones))

    # #N_RX_FFT = souk_mkid_readout.souk_mkid_readout.N_RX_FFT
    # N_RX_FFT=8192
    # rx_nearest_quick = (np.round(sweepfreqs/r.adc_clk_hz*N_RX_FFT+N_RX_FFT/2)).astype(int)%N_RX_FFT
    # rx_offset_quick =  sweepfreqs - ((rx_nearest_quick-N_RX_FFT/2)/N_RX_FFT*r.adc_clk_hz) % r.adc_clk_hz
    # chanmap_in[:,:len(rx_nearest_quick[0])] = rx_nearest_quick
    # lo_freqs = rx_offset_quick
    
    # #N_TX_FFT = souk_mkid_readout.souk_mkid_readout.N_TX_FFT
    # N_TX_FFT=4096
    # tx_nearest_quick = (np.round(sweepfreqs/r.adc_clk_hz*2*N_TX_FFT+(2*N_TX_FFT)/2)).astype(int)%(2*N_TX_FFT)
    # chanmap_out[:,tx_nearest_quick] = np.arange(numtones)
    
    for p in range(numpoints):
        for t in range(numtones):
            rx_nearest_bin, rx_offset = r._get_closest_pfb_bin(sweepfreqs[p,t])
            chanmap_in[p,t]           = rx_nearest_bin
            lo_freqs[p,t]             = rx_offset
            tx_nearest_bin            = r._get_closest_psb_bin(sweepfreqs[p,t])
            chanmap_out[p,tx_nearest_bin] = t
            
    #format the mixer offset frequencies
    print('format the mixer offset frequencies')
    fft_period_s = r.mixer._n_upstream_chans / r.mixer._upstream_oversample_factor / r.adc_clk_hz
    fft_rbw_hz   = 1./fft_period_s
    phase_steps  = lo_freqs / fft_rbw_hz * 2 * np.pi
    phase_steps  = ((((phase_steps/np.pi + 1) % 2) - 1)*2**r.mixer._phase_bp).astype('>u4')
         
    #format the amplitudes
    print('format the amplitudes')
    if amplitudes is None:
        amplitudes = np.ones(numtones,dtype=float)
    assert np.all(amplitudes>=0)
    scaling = np.round(amplitudes*(2**r.mixer._n_scale_bits-1)).astype('>u4')
    for i in range(min(r.mixer._n_parallel_chans,numtones)):
        r.mixer.write(f'rx_lo{i}_scale',scaling[i::r.mixer._n_parallel_chans].tobytes())
        r.mixer.write(f'tx_lo{i}_scale',scaling[i::r.mixer._n_parallel_chans].tobytes())
    
    #format the phases
    print('format the phases')
    if phases is None:
        phases = np.zeros(numtones,dtype=float)
    phase_offsets = ((((phases/np.pi + 1) % 2) -1 )*2**r.mixer._phase_offset_bp).astype('>u4')
    for i in range(min(r.mixer._n_parallel_chans,numtones)):
        r.mixer.write(f'rx_lo{i}_phase_offset',phase_offsets[i::r.mixer._n_parallel_chans].tobytes())
        r.mixer.write(f'tx_lo{i}_phase_offset',phase_offsets[i::r.mixer._n_parallel_chans].tobytes())
    

    #allocate arrays for the sweep result and errors
    print('allocate arrays for the sweep result and errors')
    sz = np.zeros((numpoints,numtones),dtype=complex)
    ez = np.zeros((numpoints,numtones),dtype=complex)


    #check we are using PSB
    print('check we are using PSB')
    if r.output.get_mode() != 'PSB':
        r.output.use_psb()


    #perform the sweep
    print('perform the sweep')
    for p in range(numpoints):
        #check if the chanmaps need updating
        if np.any(chanmap_in[p] != r.chanselect.get_channel_outmap()):
            print('updating the rx chanmap')
            r.chanselect.set_channel_outmap(np.copy(chanmap_in[p]))
        if np.any(chanmap_out[p] != r.psb_chanselect.get_channel_outmap()):
            print('updating the tx chanmap')
            r.psb_chanselect.set_channel_outmap(np.copy(chanmap_out[p]))
        #set the mixer frequencies
        print('set the mixer frequencies')
        for i in range(min(r.mixer._n_parallel_chans,numtones)):
            r.mixer.write(f'rx_lo{i}_phase_inc',phase_steps[p,i::r.mixer._n_parallel_chans].tobytes())
            r.mixer.write(f'tx_lo{i}_phase_inc',phase_steps[p,i::r.mixer._n_parallel_chans].tobytes())
        #start the stream
        print('start the stream')
        time.sleep(0.001)
        accs,accfreq=get_n_accs(r,samples_per_point)
        sz[p] = np.mean(accs['i'][:,chans],axis=0) + 1j * np.mean(accs['q'][:,chans],axis=0)
        ez[p] = np.std(accs['i'][:,chans],axis=0) + 1j * np.std(accs['q'][:,chans],axis=0)
        

    result = {'centerfreqs':centerfreqs,
              'spans':spans,
              'numpoints':numpoints,
              'samples_per_point':samples_per_point,
              'direction':direction,
              'amplitudes':amplitudes,
              'phases':phases,
              'chans':chans,
              'chanmap_in':chanmap_in,
              'chanmap_out':chanmap_out,
              'lo_freqs':lo_freqs,
              'scaling':scaling,
              'phase_offsets':phase_offsets,
              'phase_steps':phase_steps,
              'sweepfreqs':sweepfreqs,
              'sweep':sz,
              'noise':ez,
              'accfreq':accfreq,
              'time_per_point':samples_per_point/accfreq,
              'fpga_status':r.fpga.get_status()}
    
    return result


def sweep_v3(r,centerfreqs,spans,numpoints,samples_per_point,direction='up',amplitudes=None,phases=None,host=REMOTE_HOST,port=REMOTE_PORT,ret_samples=False):
    """
    Perform a frequency sweep using the method in sweep_v2, but operating remotely on the server.
    
    r is an souk_mkid_readout object
    centerfreqs is a numpy array of center frequencies in Hz
    spans is a numpy array of spans in Hz
    numpoints is an integer number of points to sweep
    samplesperpoint is an integer number of samples to collect and average at each point
    direction is a string, either 'up' or 'down'
    amplitudes is an optional numpy array of amplitudes for each tone, defaults to all 1.0
    phases is an optional numpy array of phases for each tone, defaults to all 0.0
    host is a string with the ip-addr or hostname of the server that will perform the sweep
    port is an integer with the port number of the server that will perform the sweep
    

    During the sweep, when a tone reaches the edge of a filterbank channel, it is remapped to the nearest channel,


    """

    #check inputs
    print('check inputs')

    centerfreqs = np.atleast_1d(centerfreqs)
    spans       = np.atleast_1d(spans)
    assert len(centerfreqs) == len(spans)
    numpoints=int(numpoints)
    assert samples_per_point>0 and samples_per_point%1==0
    assert direction in ('up','down')
    if ret_samples:
        assert(numpoints*len(centerfreqs)*samples_per_point<100e6)

    #explicitly define number of tones, and parallel/serial inputs
    print('define number of tones, and parallel/serial inputs')
    numtones = len(centerfreqs)
    chans    = np.arange(numtones,dtype=int)
    parallel = chans % r.mixer._n_parallel_chans
    serial   = chans // r.mixer._n_parallel_chans
    accfreq = r.adc_clk_hz/(r.accumulators[ACCNUM]).get_acc_len()/PFBLEN

    #define the frequency sweep points for each tone
    print('define the frequency sweep points for each tone')
    sweepfreqs = np.zeros((numpoints,numtones))
    for t in range(numtones):
        cf=centerfreqs[t]
        sp=spans[t]
        sweepfreqs[:,t] = np.linspace(cf-sp/2.,cf+sp/2.,numpoints)
        if direction=='down':
            sweepfreqs[:,t] = sweepfreqs[:,t][::-1]
    

    #define the channel maps and mixer offset frequencies for each sweep point
    print('define the channel maps and mixer offset frequencies for each sweep point')
    chanmap_in  = -1*np.ones((numpoints,r.chanselect.n_chans_out),dtype=np.int32)
    chanmap_out = -1*np.ones((numpoints,r.psb_chanselect.n_chans_out),dtype=np.int32)
    lo_freqs    = np.zeros((numpoints,numtones))

    # #N_RX_FFT = souk_mkid_readout.souk_mkid_readout.N_RX_FFT
    # N_RX_FFT=8192
    # rx_nearest_quick = (np.round(sweepfreqs/r.adc_clk_hz*N_RX_FFT+N_RX_FFT/2)).astype(int)%N_RX_FFT
    # rx_offset_quick =  sweepfreqs - ((rx_nearest_quick-N_RX_FFT/2)/N_RX_FFT*r.adc_clk_hz) % r.adc_clk_hz
    # chanmap_in[:,:len(rx_nearest_quick[0])] = rx_nearest_quick
    # lo_freqs = rx_offset_quick
    
    # #N_TX_FFT = souk_mkid_readout.souk_mkid_readout.N_TX_FFT
    # N_TX_FFT=4096
    # tx_nearest_quick = (np.round(sweepfreqs/r.adc_clk_hz*2*N_TX_FFT+(2*N_TX_FFT)/2)).astype(int)%(2*N_TX_FFT)
    # chanmap_out[:,tx_nearest_quick] = np.arange(numtones)
    
    for p in range(numpoints):
        for t in range(numtones):
            rx_nearest_bin, rx_offset = r._get_closest_pfb_bin(sweepfreqs[p,t])
            chanmap_in[p,t]           = rx_nearest_bin
            lo_freqs[p,t]             = rx_offset
            tx_nearest_bin            = r._get_closest_psb_bin(sweepfreqs[p,t])
            chanmap_out[p,tx_nearest_bin] = t
            
    #format the mixer offset frequencies
    print('format the mixer offset frequencies')
    fft_period_s = r.mixer._n_upstream_chans / r.mixer._upstream_oversample_factor / r.adc_clk_hz
    fft_rbw_hz   = 1./fft_period_s
    phase_steps  = lo_freqs / fft_rbw_hz * 2 * np.pi
    phase_steps  = ((((phase_steps/np.pi + 1) % 2) - 1)*2**r.mixer._phase_bp).astype('>u4')
    phase_steps_raw = np.zeros((numpoints,r.mixer.n_chans),dtype='<i4')
    phase_steps_raw[:,:numtones] = phase_steps
    
    #format the amplitudes
    print('format the amplitudes')
    if amplitudes is None:
        amplitudes = np.ones(numtones,dtype=float)
    assert np.all(amplitudes>=0)
    scaling = np.round(amplitudes*(2**r.mixer._n_scale_bits-1)).astype('>u4')
    for i in range(min(r.mixer._n_parallel_chans,numtones)):
        r.mixer.write(f'rx_lo{i}_scale',scaling[i::r.mixer._n_parallel_chans].tobytes())
        r.mixer.write(f'tx_lo{i}_scale',scaling[i::r.mixer._n_parallel_chans].tobytes())
    
    #format the phases
    print('format the phases')
    if phases is None:
        phases = np.zeros(numtones,dtype=float)
    phase_offsets = ((((phases/np.pi + 1) % 2) -1 )*2**r.mixer._phase_offset_bp).astype('>u4')
    for i in range(min(r.mixer._n_parallel_chans,numtones)):
        r.mixer.write(f'rx_lo{i}_phase_offset',phase_offsets[i::r.mixer._n_parallel_chans].tobytes())
        r.mixer.write(f'tx_lo{i}_phase_offset',phase_offsets[i::r.mixer._n_parallel_chans].tobytes())
    

    #allocate arrays for the sweep result and errors
    print('allocate arrays for the sweep result and errors')
    sz = np.zeros((numpoints,numtones),dtype=complex)
    ez = np.zeros((numpoints,numtones),dtype=complex)


    #check we are using PSB
    print('check we are using PSB')
    if r.output.get_mode() != 'PSB':
        r.output.use_psb()

    #request the sweep from the server
    print('request the sweep from the server')
    request = pickle.dumps({'numtones':numtones,
                          'numpoints':numpoints,
                          'samples_per_point':samples_per_point,
                          'chanmap_in':chanmap_in,
                          'chanmap_out':chanmap_out,
                          'phase_steps':phase_steps_raw,
                          'scaling':scaling,
                          'phase_offsets':phase_offsets,
                          'accfreq':accfreq
                          })
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            #send request
            if ret_samples:
                cmd=("sweepsamples %d "%(len(request))).encode()
            else:
                cmd=("sweep %d "%(len(request))).encode()
            print(len(request))
            s.connect((host, port))
            s.sendall(cmd + request)
            print(f'Successfully requested "{cmd}"')
            
            #recv response
            bs = s.recv(8)
            (length,) = struct.unpack('>Q', bs)
            data = b''
            while len(data) < length:
                data += s.recv(65536)
            response = pickle.loads(data)
            print(f'Successfully received response') #"{response}"')

        except Exception as e:
            print(f'Failed to request sweep"')
            raise(e)

    sz=response['sweepi']+1j*response['sweepq']
    ez=response['noisei']+1j*response['noiseq']
        
    acclen=(r.accumulators[ACCNUM]).get_acc_len()
    accfreq = r.adc_clk_hz/acclen/PFBLEN
    
    result = {'centerfreqs':centerfreqs,
              'spans':spans,
              'numpoints':numpoints,
              'samples_per_point':samples_per_point,
              'direction':direction,
              'amplitudes':amplitudes,
              'phases':phases,
              'chans':chans,
              'chanmap_in':chanmap_in,
              'chanmap_out':chanmap_out,
              'lo_freqs':lo_freqs,
              'scaling':scaling,
              'phase_offsets':phase_offsets,
              'phase_steps':phase_steps,
              'sweepfreqs':sweepfreqs,
              'sweep':sz,
              'noise':ez,
              'accfreq':accfreq,
              'time_per_point':samples_per_point/accfreq,
              'fpga_status':r.fpga.get_status()}
    if ret_samples:
        samplesz = response['samplesi']+1j*response['samplesq']
        result['samples']=samplesz

    return result
    


    


    

    

    

#function to collect <n> samples
def get_n_accs(r, n,print_summary=False,plot_accs=False,remote_cmd=True):
    #OBSTIME = 30 #default number of seconds to record
    #ACCLEN  = 32768 #default
    #ACCLEN  = 3000 #400hz dropped 2 packets in 2.5 seconds
    #ACCLEN  = 4000 #300hz miss one very few seconds
    #ACCLEN  = 5000 #240hz
    #ACCLEN  = 600 #1000hz
    #ACCFREQ = ADCCLK/ACCLEN/2048
    #NACCS   = int(OBSTIME*ACCFREQ)
    #NACCS   = int( 2**np.floor(np.log2(OBSTIME*ACCFREQ)))

    NACCS=n
    #print('Streamer: Get ACCLEN')
    ACCLEN=(r.accumulators[ACCNUM]).get_acc_len()
    ACCFREQ = r.adc_clk_hz/ACCLEN/PFBLEN

    #store samples in this array:
    accs      = np.zeros(NACCS,dtype=ACC_DTYPE)

    #initialise variables to track the header values and any errors
    previous_timestamp = -1
    previous_index     = -1
    timestamp_error    = False
    error_error        = False
    index_error        = False
    count_accs         = 0

    #disable any current stream
    #print('Streamer: Start: 0.0.0.0')
    #(r.accumulators[ACCNUM]).set_dest_ip(ACC_OFF_IP)


    #clear the socket buffer in case it is has overflowed
    empty_socket_buffer(s, printing=False)

    #start transmitting
    print('Streamer: Start: ',DEST_ADDR[0])
    (r.accumulators[ACCNUM]).set_dest_ip(DEST_ADDR[0])

    if remote_cmd:
        remote_stream(NACCS)
        print('process done')


    #start receiving
    while count_accs<NACCS:
        #print(count_accs)
        #get a packet
        nbytes,addr = s.recvfrom_into(inbuf,PACKET_BYTES,0)

        #start timer
        if (previous_timestamp<0) and (previous_index<0):
            t0=time.time()

        #note the header values
        current_timestamp = int(np.frombuffer(inbuf,FMT_TIMESTAMP,1,OFFSET_TIMESTAMP))
        current_error     = int(np.frombuffer(inbuf,FMT_ERROR,1,OFFSET_ERROR))
        current_index     = int(np.frombuffer(inbuf,FMT_INDEX,1,OFFSET_INDEX))
        #print(current_timestamp,current_error,current_index)

        #log header errors
        if current_error:
            acc_error=True

        #Check if we start with non-zero packet index and skip if so.
        if previous_index==-1:
            if current_index != 0:
                print(f'Current packet index ({current_index}) not at 0, skipping.')
                continue

        #Check the packet index is increasing by 1
        if (current_index - previous_index) % PACKETS_PER_ACC != 1:
            print(f'Packet indexing error: previous={previous_index} current={current_index}')
            index_error = True

        #Check the accumulation timestamp is increasing by 1
        if current_index==0:
            if (current_timestamp - previous_timestamp) != 1:
                if previous_timestamp == -1:
                    pass
                else:
                    print(f'Timestamp error: previous={previous_timestamp} current={current_timestamp}')
                    timestamp_error = True

        #if all is good, add the packet to the data store
        PACKETS[current_index] = np.frombuffer(inbuf,dtype=PACKET_DTYPE)


        #Now pull the data from the packet store and put it into the acquisition
        accs['timestamp'][count_accs] = PACKETS['packet_header']['timestamp'][0]

        accs['acc_error'][count_accs] = np.bitwise_or.reduce(PACKETS['packet_header']['error'])

        accs['i'][count_accs][current_index*PAYLOAD_CHANS:(current_index+1)*PAYLOAD_CHANS] = PACKETS[current_index]['payload'][::2]

        accs['q'][count_accs][current_index*PAYLOAD_CHANS:(current_index+1)*PAYLOAD_CHANS] = PACKETS[current_index]['payload'][1::2]

        #update previous timestamp after getting the final packet of this (r.accumulators[ACCNUM]).
        if current_index == PACKETS_PER_ACC-1:
            #print(current_timestamp)
            previous_timestamp = current_timestamp
            count_accs+=1

        previous_index = current_index

    #stop acc
    #print('Streamer: Stop, IP:0.0.0.0')
    #(r.accumulators[ACCNUM]).set_dest_ip(ACC_OFF_IP)
    print('\nStreamer Done')
    empty_socket_buffer(s, printing=False)


    tn=time.time()

    if print_summary:
        print(f'Received {count_accs} whole accumulations.')
        print(f'timestamp_error = {timestamp_error}')
        print(f'error_error = {error_error}')
        print(f'index_error = {index_error}')
        print(f'total duration = {tn-t0} seconds')
        print(f'averge packet interval = {(tn-t0)/(PACKETS_PER_ACC*NACCS)} seconds')
        print(f'average acc interval = {(tn-t0)/(NACCS)} seconds')
        print(f'packet rate = {(PACKETS_PER_ACC*NACCS)/(tn-t0)} Hz')
        print(f'acc rate = {(NACCS)/(tn-t0)} Hz')

    if plot_accs:
        plot_acc(accs,ACCFREQ,0,logmag=False,unwrapphase=False,nfft=None)


    return accs, ACCFREQ





def plot_acc(accs,accfreq,ch,logmag=False,unwrapphase=False,nfft=None):
    f,((s1,s2,s3,s4),(s5,s6,s7,s8)) = plt.subplots(2,4,
                                                sharex='row',
                                                figsize=[9.6, 4.5])

    i=accs['i'][:,ch]
    q=accs['q'][:,ch]
    a=np.absolute(i+1j*q)
    p=np.angle(i+1j*q)

    nfft= len(i) if nfft==None else nfft

    pxi,pfi = mlab.psd(i,Fs=accfreq,NFFT=nfft,window=mlab.window_none)
    pxq,pfq = mlab.psd(q,Fs=accfreq,NFFT=nfft,window=mlab.window_none)
    pxa,pfa = mlab.psd(a,Fs=accfreq,NFFT=nfft,window=mlab.window_none)
    pxp,pfp = mlab.psd(p,Fs=accfreq,NFFT=nfft,window=mlab.window_none)

    if logmag:
        a = 20*np.log10(a)
    if unwrapphase:
        p = np.unwrap(p)

    s1.plot(i)
    s2.plot(q)
    s3.plot(a)
    s4.plot(p)

    s5.loglog(pfi,pxi)
    s6.loglog(pfq,pxq)
    s7.loglog(pfa,pxa)
    s8.loglog(pfp,pxp)

    s1.title.set_text('I')
    s2.title.set_text('Q')
    s3.title.set_text('Magnitude')
    s4.title.set_text('Phase')

    plt.tight_layout()
    return f



def plot_sweep(freqs_hz,samples,sample_errors=None,offset_center=True,logmag=False,unwrapphase=False,group_delay_sec=25.3e-6,correct_boundary_phase=True,figtitle='',adc_clk_hz=2048000000):
    f           = freqs_hz
    z           = samples.real +1j*samples.imag
    
    if sample_errors is not None:
        ez = sample_errors

    phase_shift = -group_delay_sec*2*np.pi*f
    z           *= np.exp(-1j*phase_shift)
    

    if correct_boundary_phase:
        #when sweeping across the boundary between filterbank channels, a phase offset is introduced.
        #the value of the offset was recorded to be pi + 2*pi*k/1024, where k is the bin number of the leftmost bin
        #the cumulative phase offset for each bin is then given by sum(i=0, i<=k, pi + 2*pi*i/1024) = pi*(k+1)*(k+1024)/1024
        N_RX_FFT     = 8192
        bins         = (np.round(f/adc_clk_hz*N_RX_FFT+N_RX_FFT/2)).astype(int)%N_RX_FFT
        #phase_offset = np.pi*(bins+1)*(bins+1024)/1024
        phase_offset = np.pi*(bins%2-1)
        z            = z*np.exp(-1j*phase_offset)


    i           = z.real
    erri        = ez.real
    q           = z.imag
    errq        = ez.imag
    mag         = np.absolute(z)
    errmag      = np.absolute(ez)
    phase       = np.angle(z)
    errphase    = np.angle(ez) 

    #The noise in the phase is biased by any integrated linear phase shift imparted from a group delay.
    #A linear phase shift with frequency (constant group delay) is modelled as a uniform distribution,
    # so the std = sqrt( 1/12 * (b-a)**2 ). This is then divided out of the measured noise:
    err_from_gd = np.sqrt(1/12*(phase_shift[0]-phase_shift[-1])**2)
    if abs(group_delay_sec)>0:
        errphase    /= err_from_gd

    funit       = 'Hz'
    iunit       = '[ADU]'
    qunit       = '[ADU]'
    magunit     = '[ADU]'
    phaseunit   = '[rad]'



    ic = len(f) / 2
    if (ic % 1):
        cf = f[int(ic)]    #odd num points
    else:
        cf = 0.5*(f[int(ic)-1] + f[int(ic)]) # even num points

    if offset_center:
        f= (f - cf) /1e3
        funit='offset [kHz]'
    else:
        f=f / 1e6
        funit='[MHz]'

    if logmag:
        errmag = 20*np.log10((mag+errmag)/mag)
        mag = 20*np.log10(mag)
        magunit='[dB]'
    if unwrapphase:
        errphase = errphase
        phase = np.unwrap(phase)
        phaseunit = 'unwrapped [rad]'

    flabel     = 'Frequency '+funit
    ilabel     = 'I '+iunit
    qlabel     = 'Q '+qunit
    maglabel   = 'Magnitude '+magunit
    phaselabel = 'Phase '+phaseunit


    fig=plt.figure(figsize=(9.6, 4.5))
    fig.suptitle(figtitle)
    s0=plt.subplot(131,aspect='equal',adjustable='datalim')
    s1=plt.subplot(232)
    s2=plt.subplot(233,sharex=s1)
    s3=plt.subplot(235,sharex=s1)
    s4=plt.subplot(236,sharex=s1)
    s0.set_xlabel(ilabel)
    s0.set_ylabel(qlabel)

    #s1.set_xlabel('')
    #s2.set_xlabel('I')
    s3.set_xlabel(flabel)
    s4.set_xlabel(flabel)
    s1.set_ylabel(ilabel)
    s2.set_ylabel(maglabel)
    s3.set_ylabel(qlabel)
    s4.set_ylabel(phaselabel)

    if sample_errors is None:
        s0.plot(i, q, '.')
        s1.plot(f, i)
        s2.plot(f, mag)
        s3.plot(f, q)
        s4.plot(f, phase)
    else:
        s0.errorbar(i, q,xerr=erri,yerr=errq,fmt='o',ecolor='k',capsize=2)
        s1.errorbar(f, i,yerr=erri,ecolor='k',capsize=2)
        s2.errorbar(f, mag,yerr=errmag,ecolor='k',capsize=2)
        s3.errorbar(f, q,yerr=errq,ecolor='k',capsize=2)
        s4.errorbar(f, phase,yerr=errphase,ecolor='k',capsize=2)
    plt.suptitle('Center frequency = %.6f MHz'%(cf/1e6))
    plt.tight_layout()
    return fig





def get_started(restart_server=False):
    
    r=souk_mkid_readout.SoukMkidReadout(REMOTE_HOST,configfile=REMOTE_CONFIG)

    if not r.fpga.is_programmed():
        r.program()
        r.initialize()
    try:
        acc = r.accumulators[ACCNUM]
    except AttributeError:
        r.initialize()
        acc = r.accumulators[ACCNUM]
    


    if not r.adc_clk_hz:
        print('client:get_started: hardcoding r.adc_clk_hz')
        # r.adc_clk_hz=2457600000
        r.adc_clk_hz = r.rfdc.core.get_pll_config()['SampleRate']*1e9 / float(r.rfdc.core.device_info['t224_dt_adc0_dec_mode'].split('x')[0])

    if restart_server:
        #kill an aleady running server that, if there is one.
        try:
            remote_quit()
        except ConnectionRefusedError:
            pass

        #start the server now
        remote_start()
    
    #continue running in interactive shell and get accs as required
    #a1=get_n_accs(r,1)
    
    return r,acc


if __name__ == '__main__':
    
    r,acc = get_started(restart_server=False)

    gd=0.00005048176
    ch=0
    r.output.use_psb()
    freqs=np.array([500e6,1000e6,1500e6,2000e6]); amps=np.ones_like(freqs); phases=np.random.uniform(0,2*np.pi,len(freqs))

    


