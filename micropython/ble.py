import gc
import sys
import network as n
import gc
import time
 
b = n.Bluetooth()
 
found = {}
complete = True
 
def bcb(b,e,d,u):
  global complete
  global found
  if e == b.CONNECT:
    print("CONNECT")
    b.ble_settings(adv_man_name = "firebeetle-esp32", adv_dev_name="firebeetle-esp32")
    b.ble_adv_enable(True)
  elif e == b.DISCONNECT:
          print("DISCONNECT")
  else:
    print ('Unknown event', e,d)
 
def cb (cb, event, value, userdata):
  print('charcb ', cb, userdata, ' ', end='')
  if event == b.READ:
    print('Read')
    return 'ABCDEFG'
  elif event == b.WRITE:
    print ('Write', value)
 
def gatts():
  s1 = b.Service(0xaabb)
  s2 = b.Service(0xDEAD)
 
  c1 = s1.Char(0xccdd)
  c2 = s2.Char(0xccdd)
 
  c1.callback(cb, 'c1 data')
  c2.callback(cb, 'c2 data')
 
  s1.start()
  s2.start()
 
  b.ble_settings(adv_man_name = "firebeetle-esp32", adv_dev_name="firebeetle-esp32")
  b.ble_adv_enable(True)
 
b.callback(bcb)
gatts()
while(True):
  pass