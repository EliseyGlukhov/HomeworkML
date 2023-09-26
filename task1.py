import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def generate_white_noise(num_samples):
    white_noise_simple = np.random.random(num_samples)
    return white_noise_simple

def generate_pink_noise2(white_noise):
    N = len(white_noise)
    N2 = N//2

    print(N)
    print(N2)

    pink_noise = []

    for el in white_noise:  
        s = 0
        for k in range(1,N2):
            s += 1/np.sqrt(k) * np.cos(2*np.pi*k*(el-1)/N)
        s *=2
        s = s + 1 + np.cos(np.pi*(el-1))/np.sqrt(N2)

        pink_noise.append(s/N)
        #print(len(pink_noise))    
    return pink_noise

def generate_pink_noise(white_noise):
    fourier = np.fft.rfft(white_noise)
    print(len(fourier), len(white_noise))
    return fourier / np.real(np.sqrt(fourier))

def generate_black_noise(num_samples):
    return 0

def generate_noised_function(num_samples):
    return 0


white_noise = generate_white_noise(10000)

pink_noise = generate_pink_noise(white_noise)

time_axis = np.linspace(0, 5, len(pink_noise)) 

plt.figure(figsize=(10, 6))  

plt.plot(time_axis[:2000], pink_noise[:2000])

plt.title('Белый шум (простой)')
plt.xlabel('Время (секунды)')
plt.ylabel('Амплитуда')

plt.show()