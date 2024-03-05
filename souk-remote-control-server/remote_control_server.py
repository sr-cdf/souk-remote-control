#!/usr/bin/env python3

import socket
import time
import numpy as np
import struct


FPGFILE = '/home/casper/src/souk-firmware/firmware/src/souk_single_pipeline_krm/outputs/souk_single_pipeline_krm.fpg'
LISTENER_IP = '0.0.0.0'
LISTENER_PORT = 12345
DESTPORT = 10000
ACCNUM = 0

from souk_mkid_readout import SoukMkidReadout
from souk_poll_acc import get_bram_addresses, fast_read_bram, wait_non_zero_ip, format_packets

def rc_quit(r,*args):
    print('Remote Control: Quitting.')
    return

def rc_program_and_initialise(r,*args):
    print('Remote Control: Program and initialise...')
    r.program()
    r.initialize()
    print(r.fpga.print_status())
    return

def rc_stream(r,sock,naccs,*args):

    print(f'Remote Control: Stream {naccs} samples')

    acc = r.accumulators[ACCNUM]
    acc_len = acc.get_acc_len()
    n_chans = acc.n_chans
    fpga_clk = r.fpga.get_fpga_clock()
    r.adc_clk_hz = fpga_clk * 8 # HACK
    acc_time_ms = 1000* acc_len * acc._n_serial_chans / acc._n_parallel_samples / r.fpga.get_fpga_clock()
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







r       = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=False)
r_local = SoukMkidReadout('localhost', fpgfile=FPGFILE, local=True)

sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_local = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    acc       = r.accumulators[ACCNUM]
except AttributeError as e:
    print('Remote Control Error: accumulator not found, is the firmware intialised')
    raise e


listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((LISTENER_IP, LISTENER_PORT))
listener.listen()

print('Remote Control: listening on', (LISTENER_IP, LISTENER_PORT))

while True:
    lconn, laddr = listener.accept()
    with lconn:
        print(f'Remote Control: Connection from {laddr}')
        
        data = lconn.recv(65535).decode()
        print(f'Remote Control: Received \'{data}\'')

        cmd,*args = data.strip().split(' ')

        if cmd == 'x':
            rc_quit(r, args)
            break
        
        elif cmd == 'p':
            rc_program_and_initialise(r, args)
            continue
        
        elif cmd == 'stream':
            if not args:
                print('Remote Control: Stream: Invalid args')
                continue
            naccs = int(args[0])
            rc_stream(r_local, sock_local, naccs, args)
            continue
        
sock.close()
sock_local.close()