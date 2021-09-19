import pyaudio

p = pyaudio.PyAudio()
for device_index in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(device_index)
    if device_info["maxInputChannels"] > 0:
        print(device_info)
