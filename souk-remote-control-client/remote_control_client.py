import matplotlib.pyplot as plt
import numpy as np
import socket
import time
import os
import subprocess
from subprocess import Popen, PIPE


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
            n+=len(sock.recv(1024))
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
# REMOTE_SCRIPT   = '~/src/souk-remote-control/souk-remote-control-server/remote_control_start'
REMOTE_SCRIPT   = '/home/casper/src/souk-remote-control/souk-remote-control-server/remote_control'
REMOTE_CONFIG   = '/home/sam/souk/souk-firmware/software/control_sw/config/souk-single-pipeline-krm.yaml'
REMOTE_TESTPORT = 7147 #so we can temporarily connect to casper borph to find which local interface to bind the socket to.

#define the address of the receiver (destination of the packets)
DEST_IP         = get_interface_ip((REMOTE_HOST, REMOTE_TESTPORT))
DEST_PORT       = 10000
DEST_ADDR       = (DEST_IP,DEST_PORT)

ADCCLK          = 2457600000 #2.4576 GHz
PFBLEN          = 4096
NCHANS          = 2048 #number of tones in a single acquisition

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
OFFSET_TIMESTAMP = PACKET_DTYPE['packet_header'].fields[NAME_TIMESTAMP][1]
OFFSET_ERROR     = PACKET_DTYPE['packet_header'].fields[NAME_ERROR][1]
OFFSET_INDEX     = PACKET_DTYPE['packet_header'].fields[NAME_INDEX][1]
OFFSET_PAYLOAD   = PACKET_DTYPE.fields[NAME_PAYLOAD][1]
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

# xxx

#get access to the on board accumulator so we can later find out the accumulator length to calculate the sample rate and get the right number of packets
import souk_mkid_readout

r=souk_mkid_readout.SoukMkidReadout(REMOTE_HOST,configfile=REMOTE_CONFIG)

if not r.fpga.is_programmed():
    r.program()
    r.initialize()

#ACC_OFF_IP     = "0.0.0.0" #address to use to disable stream

acc = r.accumulators[0]






#functions to control a script on the board that sends out the packets

#the following scripts need to be in place on the krm board:

#/home/casper/remote_control_start -> to call the remote control script as root via ssh@localhost

#/home/casper/remote_control -> to run the python script

#souk-firmware/software/control_sw/test_scripts/souk_poll_acc_remote_control.py -> to start the control loop

#for first time, need to run ssh-keygen as casper on krm and copy the public key to /root/.ssh/authorized_keys

def remote_quit(host=REMOTE_HOST,port=REMOTE_PORT):
    """function to kill the remote control script"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            s.send(b"x")
            print('Successfully sent "x" to quit')
        except Exception as e:
            print('Failed to send quit')
            raise(e)
    time.sleep(1)
    return

def remote_start(host=REMOTE_HOST,port=REMOTE_PORT):
    """function to start the remote control script"""
    remote_command = 'ssh'
    remote_args = ['%s@%s'%(REMOTE_USER,REMOTE_HOST), '-tt', REMOTE_SCRIPT]
    process = Popen([remote_command, *remote_args],stdin=subprocess.DEVNULL)
    time.sleep(10)
    return

def remote_program_and_initialise(host=REMOTE_HOST,port=REMOTE_PORT):
    """ function to program and initialise the remote control script"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            cmd="p"
            s.connect((host, port))
            s.send(cmd.encode())
            print(f'Successfully sent "{cmd}"')
        except Exception as e:
            print(f'Failed to send "{cmd}"')
            raise(e)
    return

def remote_stream(naccs,host=REMOTE_HOST,port=REMOTE_PORT):
    """function to tell the remote control script to start streaming"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            cmd="stream %d"%naccs
            s.connect((host, port))
            s.send(cmd.encode())
            print(f'Successfully sent "{cmd}"')
        except Exception as e:
            print(f'Failed to send "{cmd}"')
            raise(e)
    return


#function to collect <n> samples
def get_n_accs(n,print_summary=False,plot_accs=False,remote_cmd=True):
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
    ACCLEN=acc.get_acc_len()
    ACCFREQ = ADCCLK/ACCLEN/PFBLEN

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
    #acc.set_dest_ip(ACC_OFF_IP)


    #clear the socket buffer in case it is has overflowed
    empty_socket_buffer(s, printing=False)

    #start transmitting
    print('Streamer: Start: ',DEST_ADDR[0])
    acc.set_dest_ip(DEST_ADDR[0])

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

        #update previous timestamp after getting the final packet of this acc.
        if current_index == PACKETS_PER_ACC-1:
            #print(current_timestamp)
            previous_timestamp = current_timestamp
            count_accs+=1

        previous_index = current_index

    #stop acc
    #print('Streamer: Stop, IP:0.0.0.0')
    #acc.set_dest_ip(ACC_OFF_IP)
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
        plot_acc(accs,ACCFREQ,ch,logmag=False,unwrapphase=False,nfft=None)


    return accs, ACCFREQ


#kill any scripts that are already running now, if there are any.
try:
    remote_quit()
except ConnectionRefusedError:
    pass

#start the script now
remote_start()


#continue running in interactive shell and get accs as required
#a1=get_n_accs(1)


def plot_acc(accs,accfreq,ch,logmag=False,unwrapphase=False,nfft=None):
    f,((s1,s2,s3,s4),(s5,s6,s7,s8)) = plt.subplots(2,4,
                                                sharex='row',
                                                figsize=[9.6, 4.5])

    i=accs['i'][:,ch]
    q=accs['q'][:,ch]
    a=np.absolute(i+1j*q)
    p=np.angle(i+1j*q)

    nfft= len(i) if nfft==None else nfft

    pxi,pfi = plt.mlab.psd(i,Fs=accfreq,NFFT=nfft,window=plt.mlab.window_none)
    pxq,pfq = plt.mlab.psd(q,Fs=accfreq,NFFT=nfft,window=plt.mlab.window_none)
    pxa,pfa = plt.mlab.psd(a,Fs=accfreq,NFFT=nfft,window=plt.mlab.window_none)
    pxp,pfp = plt.mlab.psd(p,Fs=accfreq,NFFT=nfft,window=plt.mlab.window_none)

    if logmag:
        a = 20*log10(a)
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


def plot_sweep(freqs_hz,samples,offset_center=True,logmag=False,unwrapphase=False,group_delay_sec=0.0):
    f           = freqs_hz
    z           = samples.real +1j*samples.imag
    phase_shift = -group_delay_sec*2*np.pi*f
    z           *= np.exp(-1j*phase_shift)
    i           = z.real
    q           = z.imag
    mag         = np.absolute(z)
    phase       = np.angle(z)
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
        mag = 20*log10(mag)
        magunit='[dB]'
    if unwrapphase:
        phase = np.unwrap(phase)
        phaseunit = 'unwrapped [rad]'

    flabel     = 'Frequency '+funit
    ilabel     = 'I '+iunit
    qlabel     = 'Q '+qunit
    maglabel   = 'Magnitude '+magunit
    phaselabel = 'Phase '+phaseunit


    fig=plt.figure(figsize=[9.6, 4.5])
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

    s0.plot(i, q, '.')
    s1.plot(f, i)
    s2.plot(f, mag)
    s3.plot(f, q)
    s4.plot(f, phase)
    plt.suptitle('Center frequency = %.6f MHz'%(cf/1e6))
    plt.tight_layout()
    return fig

