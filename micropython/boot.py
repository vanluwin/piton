import gc
import sys
import network as n
import gc
import time
import micropython

micropython.alloc_emergency_exception_buf(100)

b = n.Bluetooth().ble_settings(
    int_min = 1280, 
    int_max = 1280,
    adv_type = bluetooth.ADV_TYPE_IND,
    own_addr_type = bluetooth.BLE_ADDR_TYPE_PUBLIC,
    peer_addr = bytes([0] * 6),
    peer_addr_type = bluetooth.BLE_ADDR_TYPE_PUBLIC,
    channel_map = bluetooth.ADV_CHNL_ALL,
    filter_policy = blueooth.ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    adv_is_scan_rsp = False,
    adv_dev_name = None,
    adv_man_name = None,
    adv_inc_tx_power = False,
    adv_int_min = 1280,
    adv_int_max = 1280,
    adv_appearance = 0,
    adv_uuid = None,
    adv_flags = 0
)

found = {}
complete = True

def bcb(b,e,d,u):
    global complete
    global found
    if e == b.CONNECT:
	print("CONNECT")
    elif e == b.DISCONNECT:
	print("DISCONNECT")
    elif e == b.SCAN_RES:
        if complete:
            complete = False
            found = {}

        adx, name, rssi = d
        if adx not in found:
            found[adx] = name

    elif e == b.SCAN_CMPL:
	print("Scan Complete")
        complete = True
        print ('\nFinal List:')
        for adx, name in found.items():
            print ('Found:' + ':'.join(['%02X' % i for i in adx]), name)
    else:
        print ('Unknown event', e,d)

def cb (cb, event, value, userdata):
    print('charcb ', cb, userdata, ' ', end='')
    if event == b.READ:
        print('Read')
        return 'ABCDEFG'
    elif event == b.WRITE:
        print ('Write', value)
    elif event == b.NOTIFY:
        print ('Notify', value)
        period = None
        flags = value[0]
        hr = value[1]
        if flags & 0x10:
            period = (value[3] << 8) + value[2]
        print ('HR:', hr, 'Period:', period, 'ms')

def hr(bda):
    ''' Will connect to a BLE heartrate monitor, and enable HR notifications '''

    conn = b.connect(bda)
    while not conn.is_connected():
        time.sleep(.1)

    print ('Connected')

    time.sleep(2) # Wait for services

    service = ([s for s in conn.services() if s.uuid()[0:4] == b'\x00\x00\x18\x0d'] + [None])[0]
    if service:
        char = ([c for c in service.chars() if c.uuid()[0:4] == b'\x00\x00\x2a\x37'] + [None])[0]
        if char:
            descr = ([d for d in char.descrs() if d.uuid()[0:4] == b'\x00\x00\x29\x02'] + [None])[0]
            if descr:
                char.callback(cb)
                descr.write(b'\x01\x00') # Turn on notify

    return conn


def gatts():
    s1 = b.Service(0xaabb)
    s2 = b.Service(0xDEAD)

    c1 = s1.Char(0xccdd)
    c2 = s2.Char(0xccdd)

    c1.callback(cb, 'c1 data')
    c2.callback(cb, 'c2 data')

    s1.start()
    s2.start()

    b.ble_settings(adv_man_name = "mangocorp", adv_dev_name="mangoprod")
    b.ble_adv_enable(True)


b.callback(bcb)
gatts()
while(True):
  pass