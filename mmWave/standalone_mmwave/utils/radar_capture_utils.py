# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:21:10 2021

@author: Soheil
"""
# DCA code is based on mmwave package (openradar)

# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



import serial
import threading
import time
import json
import socket
import struct
import codecs
import os
from enum import Enum
from collections import namedtuple

# import tensorflow as tf
import numpy as np
import pandas as pd

def fix_byte_order(data_buffer):
    data_buffer=np.frombuffer(data_buffer,dtype=np.dtype('<H'))

    tmp_buffer=np.empty_like(data_buffer)
    tmp_buffer[0::4]=data_buffer[0::4]
    tmp_buffer[1:-1:4]=data_buffer[2::4]
    tmp_buffer[2::4]=data_buffer[1::4]
    tmp_buffer[3::4]=data_buffer[3::4]

    data_buffer=tmp_buffer
    return data_buffer


def read_stream(dca):
    packet_num=0
    byte_cnt=0
    packet_dat=[]
    while(1):
        try:
            packet_num,byte_cnt,packet_dat=dca._read_data_packet()
        except:
            break
    try:
        packet_num,byte_cnt,packet_dat=dca._read_data_packet()
    except:
        pass
    if len(packet_dat)==0:
        print('Timeout!')
    return packet_num, byte_cnt, packet_dat


class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'

    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    # CONFIG_FPGA_GEN_CMD_CODE
    # 5a a5 03 00 06 00 01 02 01 02 03 1e aa ee
    # header: 5aa5
    # cmd: 0300
    # size: 0600
    # logging mode: 01 (rawmode: 01 multimode: 02)
    # lvds mode: 02 (1243: 01 1642: 02)
    # data transfer mode: 01 (lvds capture: 01 DMM playback: 02)
    # data capture mode: 02 (sd card store: 01 ethernet stream: 02)
    # data format mode: 03 (12bit: 01 14bit: 02 16bit:03)
    # timer: 1e (timer infor in seconds: 1e=30s)        

    CONFIG_EEPROM_CMD_CODE = '0400'
    # CONFIG_EEPROM_CMD_CODE 
    # 5a a5 04 00 12 00 c0 05 35 0c 00 00 aa ee
    # header: 5aa5
    # cmd: 0400
    # size: 0x12: 18 bytes
    # system ip address: 211ec0a8 (30:1e 33:21 192:c0 168:a8) 
    # fpga ip address: 21b4c0a8 (30:1e 33:21 192:c0 168:a8) 
    # fpga mac address: 6 bytes of mac:  0th byte- 12 1st byte – 90 2nd byte – 78 3rd byte – 56 4th byte – 34 5th byte - 12
    # config port: 0010 : 4096
    # data port: 0210 : 4098
    
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'

    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'

    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'    
    # CONFIG_PACKET_DATA_CMD_CODE 
    # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
    # header: 5aa5
    # cmd: 0b00
    # size: 0600
    # packetsize: c005 (0x05c0=1472) in packets
    # delay: 350c (0x0c35=3125) in uc*1000/8
 
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'
    # READ_FPGA_VERSION_CMD_CODE (this is the response)
    # 5a a5 0e 00 04 03 02 01 aa ee
    # header: 5aa5
    # status: 04030201 (fpga verson: 1.2.3.4)

    def __str__(self):
        return str(self.value)
        


# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'
#ADC_PARAMS = {'chirps': 128,  # 32
#              'rx': 4,
#              'tx': 3,
#              'samples': 128,
#              'IQ': 2,
#              'bytes': 2}

ADC_PARAMS = {'chirps': 4,  # 32
              'rx': 1,
              'tx': 1,
              'samples': 13*13,
              'IQ': 2,
              'bytes': 2}

# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456
# DYNAMIC
BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET
UINT16_IN_PACKET = BYTES_IN_PACKET // 2
UINT16_IN_FRAME = BYTES_IN_FRAME // 2


class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """

    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096,verbose=False):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.data = []
        self.packet_count = []
        self.byte_count = []

        self.frame_buff = []

        self.curr_buff = None
        self.last_frame = None

        self.lost_packets = None
        
        self.active_flag=True
        
        self.verbose=verbose
        
        self.data_thread=threading.Thread(target=self.worker_data_read);
#        self.file_thread=threading.Thread(target=self.worker_file_write);


    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        # SYSTEM_CONNECT_CMD_CODE
        # 5a a5 09 00 00 00 aa ee
        (self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE))

        # READ_FPGA_VERSION_CMD_CODE
        # 5a a5 0e 00 00 00 aa ee
        (self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE))

        # CONFIG_FPGA_GEN_CMD_CODE
        # 5a a5 03 00 06 00 01 02 01 02 03 1e aa ee
        # header: 5aa5
        # cmd: 0300
        # size: 0600
        # logging mode: 01 (rawmode: 01 multimode: 02)
        # lvds mode: 02 (1243: 01 1642: 02)
        # data transfer mode: 01 (lvds capture: 01 DMM playback: 02)
        # data capture mode: 02 (sd card store: 01 ethernet stream: 02)
        # data format mode: 03 (12bit: 01 14bit: 02 16bit:03)
        # timer: 1e (timer infor in seconds: 1e=30s)        
#        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', 'c005350c0000'))
        (self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', '01020102031e'))
        
        # CONFIG_PACKET_DATA_CMD_CODE 
        # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
        # header: 5aa5
        # cmd: 0b00
        # size: 0600
        # packetsize: c005 (0x05c0=1472) in packets
        # delay: 350c (0x0c35=3125) in uc*1000/8
        # future use: 0x0000
#        M=struct.pack('HHH',1472,3125,0x0000)

#        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))
        (self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()
        self.active_flag=False
    def reset_DCA(self):
        self._send_command(CMD.RESET_FPGA_CMD_CODE.value)
        
    def reset_AR(self):
        self._send_command(CMD.RECORD_START_CMD_CODE.value)
        self._send_command(CMD.RESET_AR_DEV_CMD_CODE.value)
        self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE.value)


    def read(self, timeout=1):
        """ Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        ret_frame = np.zeros(UINT16_IN_FRAME, dtype=np.uint16)
        # Wait for start of next frame
        while True:
            packet_num, byte_count, packet_data = self._read_data_packet()
            if byte_count % BYTES_IN_FRAME_CLIPPED == 0:
#                print(packet_num)
                packets_read = 1
                print('Packet read !')
                ret_frame[0:UINT16_IN_PACKET] = packet_data
                break

        # Read in the rest of the frame            
        while True:
            packet_num, byte_count, packet_data = self._read_data_packet()
            packets_read += 1
            print('Packet read !')

            if byte_count % BYTES_IN_FRAME_CLIPPED == 0:
                self.lost_packets = PACKETS_IN_FRAME_CLIPPED - packets_read
                return ret_frame

            curr_idx = ((packet_num - 1) % PACKETS_IN_FRAME_CLIPPED)
            try:
                ret_frame[curr_idx * UINT16_IN_PACKET:(curr_idx + 1) * UINT16_IN_PACKET] = packet_data
            except:
                pass

            if packets_read > PACKETS_IN_FRAME_CLIPPED:
                packets_read = 0

    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print('Timeout Error!:'+ str(e))
        try:
            decoded_response=struct.unpack(b'4H',resp)
        except:
            print('Unknown Error!\n Hint:') 
            print(resp) 
            return resp
        if decoded_response[1]==0x000e:
            print('FPGA version:'+str(decoded_response[2]))
        else:
            success_flag=decoded_response[2]==0
            if success_flag:
                print('Success!')
            else:
                print('Error!')            
        return resp

    def _read_data_packet(self,verbose=False):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        self.data.append(packet_data)
        if verbose:
            print('Packet_num:'+str(packet_num)+ ' Byte count'+str(byte_count)+' Data len:'+ str(len(packet_data)))
        return packet_num, byte_count, packet_data

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    def _stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            str: Response Message

        """
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)

        # Separate IQ data
        ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        return ret.reshape((num_chirps, num_rx, num_samples))

    def worker_data_read(self):
        """thread reader function"""
        while self.active_flag:
            try:
                packet_num,byte_cnt,packet_dat=self._read_data_packet(self.verbose)
            except:
                pass
        return;

    def start_record(self,timeout=1):
        self.data_socket.settimeout(timeout)
        #self.file_thread.start();
        self.data_thread.start();   
        return 0

    def decode_data(data_buffer,num_Tx,num_Rx=1): 
        # returns pointcloud and headers in a more concise way

        header_size=56
        HSI_HEADER_ID1=b'\xdc\x0a\xda\x0c\xdc\x0a\xda\x0c' ## for chirps
        HSI_HEADER_ID2=b'\xc9\x0c\xcc\x09\xc9\x0c\xcc\x09' ## for user data (tlv)
    #    paddingBuffer=b'\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f\x0f' ## end of each header


        data_buffer=np.frombuffer(data_buffer,dtype=np.dtype('<H'))

        data_buffer=fix_byte_order(data_buffer) ## byte order is interleaved for some reason
        # this is because 2 lanes in lvds, order of bytes: lane1 real(sample 1), lane1 real(sample 2), lane2 imag(sample 1), lane2 imag(sample 2)

        frames=bytearray(data_buffer).split((HSI_HEADER_ID2))
        blocks=[frame.split(HSI_HEADER_ID1) for frame in frames]

        # first packet in each block is the tlv packet of previous frame, rest are chirps
        num_chirps=np.int32(np.median([len(block)-1 for block in blocks])) 
        # shu: see on average how many chirps

        # just process blocks that have data (in addition to header)
        #    tlv_frames=[block[0] for block in blocks[1:]]
        tlv_frames=[block[0] for block in blocks[1:] if len(block)>1]


        ####################### interpret per-frame headers
        ####################### TODO: interpret sdk_header

        Header = namedtuple('Header_info','Total_len reserved version header_size sdk_header padding_buffer')
        header_info2=pd.DataFrame([Header._make(struct.unpack('2I2H32s12s', frame[:56])) for frame in tlv_frames])
        # shu: '2I2H32s12s' specifies the layout of the binary data: 2 unsigned integers, 2 unsigned shorts, 32 bytes, 12 bytes
        Header = namedtuple('Header_info','frame_no sub_frame_no num_obj')    
        tmp_header=[struct.unpack('I2H',tlv_frames[i][header_info2['header_size'][i]*2-8:header_info2['header_size'][i]*2]) for i in range(len(tlv_frames))]
        header_info3=pd.DataFrame([Header._make(header) for header in tmp_header])

        frame_header=pd.concat([header_info2,header_info3],axis=1)

        ###################### get tlv data

        tlv_data=[tlv_frames[i][header_info2['header_size'][i]*2:] for i in range(len(tlv_frames))]
        # shu: TLV (Type-Length-Value) is a common data encoding scheme

        p_list=[]
        pinfo_list=[]
        # frame_header.to_csv(f'/bigdata/shuboy/mmw_capture_uncontrolled/test_waveform_2024_Sep_14/2024Sep17-0527_18fps_3s_test5mins/frame_header_file_mvdopclark0.csv', index=False)
        for i in range(len(frame_header)):
            obj_num=frame_header['num_obj'][i]
            if obj_num>0:
                p=np.array(struct.unpack(str(4*obj_num)+'f',tlv_data[i][:4*4*obj_num])).reshape((-1,4))
                ## x,y,z,v
                info=np.array(struct.unpack(str(2*obj_num)+'h',tlv_data[i][4*4*obj_num:])).reshape((-1,2))*0.1 # steps are 0.1db
                ## snr, noise in db
                pc=np.concatenate([p,info],-1)
                pc=pd.DataFrame(columns=['x','y','z','v','snr_db','noise_db'],data=pc)
                pc['frame']=i
            else:
                pc=pd.DataFrame()
            p_list.append(pc)

        ######################## get adc data

        block_size=np.median([len(block[1:][-1]) for block in blocks[1:-1]],0) 

        samples_per_chirp=np.int32((block_size-header_size)/2/2/num_Rx) # 2(IQ) x 2(bytes)
        # print('samples_per_chirp', samples_per_chirp) # shu: 128

        # first block is shorter (for calibration?), and last block doesnt have chirp data
        # if length is not num_chirps+header, skip the entire frame
    #    chirp_block=np.concatenate([block[1:] for block in blocks if len(block)==num_chirps+1],0) 
        num_frames = { 'decoded_frames': len(blocks) }
        # for i, block in enumerate(blocks):
        #     if len(block) != num_chirps + 1:
        #         print(f"-----Block {i} does not have chirps+1 length")
        chirp_block=([block[1:] for block in blocks if len(block)==num_chirps+1]) # shu: +1 means + header?

        # shu: block[1:] means skip the first element - header

        frames=[]
        frame_idx = len(blocks) - len(chirp_block)
        num_frames['chirp_frames'] = len(chirp_block)
        num_frames['not_chirp_frames'] = frame_idx
        varying_chirp_frames = []
        # print(f"Block {frame_idx} does not have chirps+1 length")
        for frame in chirp_block: # shu: chirp_block: frames that have chirps
            frame_idx += 1
            if (np.var([len(block) for block in frame])==0): # shu: each block in frame is a chirp+header
                # if this frame has varying chirp number, skip
                frames.append(frame)
            else:
                varying_chirp_frames.append(frame_idx)
                # print('Varying chirp number in this frame', frame_idx)
        num_frames['varying_chirp_frames'] = varying_chirp_frames

        chirp_block=np.concatenate(frames)    
        chirp_data=chirp_block[...,header_size:]

        Header = namedtuple('Header_info','Total_len reserved version header_size sdk_header padding_buffer')
        chirp_header=pd.DataFrame([Header._make(struct.unpack('2I2H32s12s', chirp_block[i,:56])) for i in range(chirp_block.shape[0]) ])

        chirp_data=np.frombuffer(bytearray(chirp_data),dtype=np.dtype('<h'))

        # chirp_data=chirp_data[...,::2]+chirp_data[...,1::2]*1j
        # JIDA change real and imag part
        chirp_data=chirp_data[...,::2]*1j+chirp_data[...,1::2]

        dat=chirp_data.reshape((-1,num_chirps//num_Tx,num_Tx,num_Rx,samples_per_chirp))
        headers={'frame_header':frame_header,
                 'chirp header':chirp_header
        }
        num_frames['resulted_frames'] = len(dat)


        ######################### 
        return dat, p_list, headers, num_frames
        
    def startup(self,timeout=0.3):
        self.reset_DCA()
        self.configure()
        self.start_record(timeout=timeout)
        self.data=[]
        self.reset_AR()

    def restart(self):
        self.reset_DCA()
        self.configure()
        self.data=[]
        self.reset_AR()



class AWR1843:
    def __init__(self,radar_name='Radar1843',cmd_serial_port='COM3',dat_serial_port='COM4',config_file_name=''):
        
        self.radar_name=radar_name;
        self.recieved_data=[];
        self.loaded_config=[];
        self.config_file_name=config_file_name;
        
        self.serial_cmd=serial.Serial(cmd_serial_port, 115200, timeout=5);
        self.serial_data=serial.Serial(dat_serial_port, 921600, timeout=5);

        self.data_recieved_flag=threading.Event();
        self.active_flag=threading.Event();
        

        self.serial_thread=threading.Thread(target=self.worker_serial_read);
        self.file_thread=threading.Thread(target=self.worker_file_write);
#       t_s = threading.Thread(target=worker_serial_read,args=(0,))

        str_time=time.ctime();
        str_time=str_time.replace(' ','_');
        self.str_time=str_time.replace(':','_');    
        self.data=[]
        
        self.setup()

    def open_serial(self):
        
        self.serial_cmd.close();
        self.serial_data.close();
        
        self.serial_cmd.open();
        self.serial_data.open();
            
        return;

    def close_serial(self):
        self.serial_cmd.close()
        self.serial_data.close()
        return;

    def load_config(self,file_name,print_outputs=True,start=True):
        self.reset_cmd_port()        
        file_tmp  = open(file_name, 'r');
        str_tmp=file_tmp.read()+'\n';
        self.loaded_config=str_tmp;
        tmp_idx=str_tmp.find('sensorStop\n');    
#        tmp_idx=str_tmp.find('flushCfg\n');    

        if print_outputs:
            print(self.radar_name+' config:\n'+str_tmp[0:tmp_idx])
        file_tmp.close()    
        list_tmp=str_tmp[tmp_idx:].splitlines(True)
        for cnt in range(len(list_tmp)):        
            self.serial_cmd.write(list_tmp[cnt].encode('ASCII'));
            time.sleep(0.01)
            reply=str(self.serial_cmd.read_all())
            if print_outputs:
#                print(str(self.serial_cmd.read_until(b'Done\n')))        
                print(reply)
            if len(reply)<4:
                self.reset_cmd_port()
                cnt=0
            else:
#                print(len(reply))
                pass
                
        if start:
            self.serial_cmd.write(b"sensorStart\n");
        return;

    def set_config(self):
        self.serial_cmd.write(b"sensorStart\n");
        time.sleep(0.1)
        self.serial_cmd.write(b"sensorStop\n");


    def send_command(self,command,print_outputs=True):

#        str_tmp=command.encode()+'\n';

        self.serial_cmd.write(command.encode('ASCII'));
        if print_outputs:
            print(str(self.serial_cmd.read_until(b'Done\n')))        
#                print(str(self.serial_cmd.read_all()))        
        return;

    def start_radar(self,flag=0):
        if flag:
            self.serial_cmd.write(b"sensorStart\n");
        else:
            self.serial_cmd.write(b"sensorStart 0\n");
#        strtmp=self.serial_cmd.read_until(b'Done\n');
        self.serial_cmd.read_all();        
        # if strtmp!=b'':
        #     print(self.radar_name+' Start successful!\n')            
        # else:
        #     print(self.radar_name+' Start failed!\nReason:\n')
        #     print(strtmp)
        return;

    def stop_radar(self):
        self.serial_cmd.write(b"sensorStop\n");
        strtmp=self.serial_cmd.read_until(b'Done\n');
        # if strtmp!=b'':
        #     print(self.radar_name+' Stop successful!\n')
        # else:
        #     print(self.radar_name+' Stop failed!\nReason:\n')
        #     print(strtmp)
        self.serial_cmd.read_all();        
        return;

    def reset_cmd_port(self):
        self.serial_cmd.flush();        
        self.serial_cmd.close();
#        time.sleep(0.5)
        self.serial_cmd.open();        
        
    def worker_serial_read(self):
        """thread reader function"""
        while self.active_flag.is_set():            
            if not self.data_recieved_flag.is_set() and self.serial_data.in_waiting>0:
                strtmp=self.serial_data.read_until(b'\x02\x01\x04\x03\x06\x05\x08\x07');
                #strtmp=self.serial_data.read_all();
                if(strtmp!=b''):
                    self.recieved_data=strtmp;
                    self.data_recieved_flag.set();
#            else:
#                time.sleep(0.01)
                
        return;

    def get_data(self):
        if True:#self.serial_data.in_waiting>0:
#            strtmp=self.serial_data.read_until(b'\x02\x01\x04\x03\x06\x05\x08\x07');
            strtmp=self.serial_data.read_all();
            self.data.append(strtmp)
            if(strtmp!=b''):
                self.recieved_data=strtmp;
                file_dat  = open('test_'+self.radar_name+'_'+self.str_time+'.dat', 'ab+') 
                file_dat.write(strtmp)
                file_dat.close()
            else:
                self.serial_data.close();        
                self.serial_data.open();        
        else:
            strtmp=''
                
        return strtmp;

    def worker_file_write(self):
        """thread worker function"""
        str_time=time.ctime();
        str_time=str_time.replace(' ','_');
        str_time=str_time.replace(':','_');    
        while self.active_flag.is_set():
            self.data_recieved_flag.wait();
            file_dat  = open('test_'+self.radar_name+'_'+str_time+'.dat', 'ab+') 
            file_dat.write(self.recieved_data)
            file_dat.close()
            self.recieved_data=b'';
            self.data_recieved_flag.clear();        
        return;
            
    def setup(self):
         self.open_serial();
#         self.load_config(self.config_file_name);
         self.stop_radar();
         self.serial_data.flush();
         self.active_flag.set();
#         self.file_thread.start();
#         self.serial_thread.start();                                    
         return;
         
    def kill(self):
        self.stop_radar();
        self.active_flag.clear();
        self.close_serial();
        return;
        
class async_radar_saver:
    def __init__(self,radar_data=[],camera_frame=[],path_out='./'):
        self.file_thread=threading.Thread(target=self.worker_file_write);
        self.radar_data=radar_data.copy()
        self.camera_frame=camera_frame.copy()
        self.path_out=path_out
        self.file_thread.start()   

    def worker_file_write(self):
        """thread reader function"""
        np.savez(self.path_out,radar_data=self.radar_data,camera_data=self.camera_frame)
        self.radar_data=[]
        return;

        