#!/usr/bin/env python3

import socket
import time
import numpy as np
import struct
import pickle

# FPGFILE = '/home/casper/src/souk-firmware/firmware/src/souk_single_pipeline_krm/outputs/souk_single_pipeline_krm.fpg'
CONFIGFILE = '/home/casper/src/souk-firmware/software/control_sw/config/souk-single-pipeline-krm-2G.yaml'
LISTENER_IP = '0.0.0.0'
LISTENER_PORT = 12345
DESTPORT = 10000
ACCNUM = 0

from souk_mkid_readout import SoukMkidReadout
from souk_poll_acc import get_bram_addresses, get_bram_addresses_mixer_tx,get_bram_addresses_mixer_rx,fast_write_mixer,fast_read_bram, wait_non_zero_ip, format_packets



#define the available remote control actions:

def rc_program_and_initialise(r):
    print('Remote Control: Program and initialise...')
    r.program()
    r.initialize()
    print(r.fpga.print_status())
    return

def read_n_accs(r,naccs):
    acc=r.accumulators[ACCNUM]
    addrs, nbytes = get_bram_addresses(acc)
    goodaccs=0
    errcount=0
    data = np.zeros((naccs,2*acc.n_chans),dtype='<i4')
    while goodaccs<naccs:

        acc._wait_for_acc(0.00005)
        start_acc_cnt, dout, err_acc_cnt = fast_read_bram(acc, addrs, nbytes)

        if err_acc_cnt:
            print('Error reading acc')
            errcount+=1
            if errcount==naccs:
                'too many errs, breaking out of loop'
                break
            continue
        else:
            data[goodaccs] = dout
            goodaccs+=1
            # print(f'Accumulation {goodaccs} of {naccs} read')
    return data
        
def write_mixer_phase_steps(r,phases):
    phase_steps = np.zeros(r.mixer.n_chans)
    phase_steps[:len(phases)] = phases
    mixer_tx_addrs, mixer_tx_nbytes = get_bram_addresses_mixer_tx(r.mixer)
    mixer_rx_addrs, mixer_rx_nbytes = get_bram_addresses_mixer_rx(r.mixer)
    fast_write_mixer(r.mixer, phase_steps, mixer_tx_addrs, mixer_tx_nbytes)
    fast_write_mixer(r.mixer, phase_steps, mixer_rx_addrs, mixer_rx_nbytes)
    return


def rc_stream(r,sock,arg):
    args = pickle.loads(arg)
    naccs = args['naccs']
    print(f'Remote Control: Stream {naccs} samples')
    acc = r.accumulators[ACCNUM]
    acc_len = acc.get_acc_len()
    n_chans = acc.n_chans
    #fpga_clk = r.fpga.get_fpga_clock()
    #r.adc_clk_hz = fpga_clk * 8 # HACK
    #acc_time_ms = 1000* acc_len * acc._n_serial_chans / acc._n_parallel_samples / r.fpga.get_fpga_clock()
    # r.adc_clk_hz = r.rfdc.core.get_pll_config(0,'adc')['SampleRate']*1e9 / float(r.rfdc.core.device_info['t224_dt_adc0_dec_mode'].split('x')[0])

    fpga_clk = r.adc_clk_hz / 8
    acc_time_ms = 1000* acc_len * acc._n_serial_chans / acc._n_parallel_samples / fpga_clk
    print(f'Accumulation time is approximately {acc_time_ms:.1f} milliseconds')
    addrs, nbytes = get_bram_addresses(acc)
    acc._wait_for_acc(0.00005)
    t0 = time.time()
    ip = None
    err_cnt = 0
    loop_cnt = 0
    times = []
    tlast = None
    print('Entering loop')
    while True:
        ip = wait_non_zero_ip(acc)
        acc._wait_for_acc(0.00005)
        tt0 = time.time()
        t, d, err = fast_read_bram(acc, addrs, nbytes)
        if ip is not None:
            for p in format_packets(t, d, error=err):
                sock.sendto(p, (ip, DESTPORT))
        if err or (tlast is not None and tlast != t-1):
            err_cnt += 1
        tt1 = time.time()
        times += [tt1 - tt0]
        loop_cnt += 1
        if loop_cnt == naccs:
            break
        tlast = t
    t1 = time.time()
    avg_read_ms = np.mean(times)*1000
    max_read_ms = np.max(times)*1000
    avg_loop_ms = (t1-t0)/loop_cnt * 1000
    print(f'Average read time: {avg_read_ms:.2f} ms')
    print(f'Max read time: {max_read_ms:.2f} ms')
    print(f'Average loop time: {avg_loop_ms:.2f} ms')
    print(f'Number of reads: {loop_cnt}')
    print(f'Number of too slow reads: {err_cnt}')

    return


def rc_sweep(r,r_local,arg,ret_samples=False):
    request=pickle.loads(arg)
    print(f'Remote Control: Sweep request')
    numpoints = request['numpoints']
    numtones = request['numtones']
    samples_per_point = request['samples_per_point']
    if numpoints*numtones*samples_per_point >=100e6:
        ret_samples=False
    accfreq=request['accfreq']

    chanmap_in = request['chanmap_in'].astype(np.int32)
    chanmap_out = request['chanmap_out'].astype(np.int32)
    current_chan_outmap_in = r.chanselect.get_channel_outmap()
    current_chan_outmap_out = r.psb_chanselect.get_channel_outmap()
    
    mixer_nchans = r.mixer.n_chans
    szi = np.zeros((numpoints,numtones),dtype='<i4')
    szq = np.zeros((numpoints,numtones),dtype='<i4')
    ezi = np.zeros((numpoints,numtones),dtype='<i4')
    ezq = np.zeros((numpoints,numtones),dtype='<i4')

    if ret_samples:
        samplesi = np.zeros((numpoints,numtones,samples_per_point),dtype='<i4')
        samplesq = np.zeros((numpoints,numtones,samples_per_point),dtype='<i4')
        
    phase_steps = request['phase_steps'].astype('<i4')
    mixer_tx_addrs, mixer_tx_nbytes = get_bram_addresses_mixer_tx(r_local.mixer)
    mixer_rx_addrs, mixer_rx_nbytes = get_bram_addresses_mixer_rx(r_local.mixer)

    for p in range(numpoints):
        print(f'sweep point {p+1} of {numpoints}')

        #check if the chanmaps need updating
        print('compare in')
        chanmap_update = False
        comp=chanmap_in[p] != current_chan_outmap_in
        # print('check any')
        if np.any(comp):
            print('updating the rx chanmap')
            chanmap_update=True
            r.chanselect.set_channel_outmap(np.copy(chanmap_in[p]))
            current_chan_outmap_in = chanmap_in[p]

        print('compare out')
        comp = chanmap_out[p] != current_chan_outmap_out
        # print('check any')
        if np.any(comp):
            print('updating the tx chanmap')
            chanmap_update=True
            r.psb_chanselect.set_channel_outmap(np.copy(chanmap_out[p]))
            current_chan_outmap_out = chanmap_out[p]
        if chanmap_update:
            time.sleep(5*1/accfreq)
        #set the mixer frequencies
        print('set the mixer frequencies')

        # write_mixer_phase_steps(r_local,request['phase_steps'][p])
        phases = phase_steps[p]
        #fast_write_mixer(r.mixer, phase_steps, mixer_addrs, mixer_nbytes)
        phases = phases.reshape(r.mixer._n_parallel_chans, r.mixer._n_serial_chans)
        n_write_tx = (mixer_tx_nbytes // 512)
        n_write_rx = (mixer_rx_nbytes // 512)
        for i, addr in enumerate(mixer_tx_addrs):
            raw = phases[i].tobytes()
            for j in range(n_write_tx):
                r_local.mixer.host.transport.axil_mm[addr+j*512:addr +(j+1)*512] = raw[j*512:(j+1)*512]
        for i, addr in enumerate(mixer_rx_addrs):
            raw = phases[i].tobytes()
            for j in range(n_write_rx):
                r_local.mixer.host.transport.axil_mm[addr+j*512:addr +(j+1)*512] = raw[j*512:(j+1)*512]

        # for i in range(min(r.mixer._n_parallel_chans,request['numtones'))):
        #     write_mixer_phase_steps(r_local,request['phase_steps'])
        #     r.mixer.write(f'lo{i}_phase_inc',request['phase_steps'][p,i::r.mixer._n_parallel_chans].tobytes())
        
        
        time.sleep(50*1/accfreq)

        #start the stream
        print('start the stream')

        data = read_n_accs(r_local,samples_per_point)
        # print('slice iq data for mean/std')
        i,q=data[:,0:2*numtones:2],data[:,1:2*numtones:2]
        # print('calc mean/std')
        szi[p] = np.mean(i,axis=0)[:numtones]
        szq[p] = np.mean(q,axis=0)[:numtones]
        ezi[p] = np.std(i,axis=0)[:numtones]
        ezq[p] = np.std(q,axis=0)[:numtones]
        if ret_samples:
            samplesi[p] = i[:,:numtones].swapaxes(0,1)
            samplesq[p] = q[:,:numtones].swapaxes(0,1)    
        
    request['sweepi']=szi
    request['sweepq']=szq
    request['noisei']=ezi
    request['noiseq']=ezq
    if ret_samples:
        request['samplesi']=samplesi
        request['samplesq']=samplesq
    return request


#establish casper connections to the firmware
# r       = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=False) # for general fw register wirting
# r_local = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=True) # for fast accumulator streaming
r       = SoukMkidReadout('localhost', configfile=CONFIGFILE, local=False) # for general fw register wirting
r_local = SoukMkidReadout('localhost', configfile=CONFIGFILE, local=True) # for fast accumulator streaming

#check the firmware is intitialised
try:
    acc       = r.accumulators[ACCNUM]
except AttributeError as e:
    print('Remote Control Error: accumulator not found, trying to program the board...')
    r.program()
    r.initialize()
    # r       = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=False) # for general fw register wirting
    # r_local = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=True) # for fast accumulator streaming
    r       = SoukMkidReadout('localhost', configfile=CONFFILE, local=False) # for general fw register wirting
    r_local = SoukMkidReadout('localhost', configfile=CONFFILE, local=True) # for fast accumulator streaming


sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # for general fw register wirting
sock_local = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # for fast accumulator streaming 


#open the remote control listener socket
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((LISTENER_IP, LISTENER_PORT))
listener.listen()
print('Remote Control: listening on', (LISTENER_IP, LISTENER_PORT))

#listen for remote control commands, break out of loop if exit command is received
while True:
    lconn, laddr = listener.accept()
    with lconn:
        print(f'Remote Control: Connection from: {laddr}')
        
        #receive message
        msg = lconn.recv(65536)
        
        #seperate the command and the argument
        cmd = (msg[:msg.find(b' ')]).decode()
        print(f'Remote Control: Received message: "{cmd}"')

        args = msg[msg.find(b' ')+1:]
        if args.find(b' ')<0:
            arglen = int(args.decode())
            arg=b''
        else:
            arglen = int((args[:args.find(b' ')]).decode())
            arg = args[args.find(b' ')+1:] 
        while len(arg)<arglen:
            arg+=lconn.recv(65535)

        #parse the command and perform the requested action
        if cmd == 'h':
            print('Remote Control: Help: Available commands are: exit, prog, stream, sweep')
            continue

        elif cmd == 'exit':
            print('Remote Control: Quitting.')
            break
        
        elif cmd == 'prog':
            rc_program_and_initialise(r)
            continue
        
        elif cmd == 'stream':
            rc_stream(r_local, sock_local, arg)
            print('Remote Control: Stream: Done')
            continue
        
        elif cmd == 'sweep':
            print('rc_sweep')
            print(len(arg))
            result   = rc_sweep(r, r_local, arg)
            print('pickle dump')
            response = pickle.dumps(result)
            print('sendall')
            length   = struct.pack('>Q', len(response))
            lconn.sendall(length)
            lconn.sendall(response)
            print('Remote Control: Sweep: Sent response')
            continue

        elif cmd == 'sweepsamples':
            print('rc_sweep')
            print(len(arg))
            result   = rc_sweep(r, r_local, arg,ret_samples=True)
            print('pickle dump')
            response = pickle.dumps(result)
            print('sendall')
            length   = struct.pack('>Q', len(response))
            lconn.sendall(length)
            lconn.sendall(response)
            print('Remote Control: Sweep: Sent response')
            continue
        else:
            print(f'Remote Control: Invalid command: "{cmd} {arg}" ')
            continue
        
#close the listener
try:
    sock.shutdown(socket.SHUT_RDWR)
    sock_local.shutdown(socket.SHUT_RDWR)
except:
    pass
sock.close()
