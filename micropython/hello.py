import machine

print("Hello World!")


led = machine.Pin(2, machine.Pin.OUT)
led.value(1)
