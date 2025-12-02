import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests


asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

class ASR:
    def __init__(self):
        self._recognizer = OfflineRecognizer()
        raise NotImplementedError

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate=16000) -> str:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sample_rate)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text


class Whisper(ASR):
    def __init__(self, encoder_path: str, decoder_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )


class Paraformer(ASR):
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )

print('正在加载模型...')
asr = Paraformer(
    model_path=f'{asr_path}/model.int8.onnx',
    tokens_path=f'{asr_path}/tokens.txt',
    # provider='cuda',
)
print('模型加载完成')

sample_rate = 16000

from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
config = VadModelConfig(
    SileroVadModelConfig(
        model=f'{vad_path}/silero_vad.onnx',
        min_silence_duration=0.25,
    ),
    sample_rate=sample_rate
)
window_size = config.silero_vad.window_size
vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
samples_per_read = int(0.1 * sample_rate)

control_url = "http://172.20.107.7:5000/control"  # 改成树莓派的ip地址

# 速度控制参数 (0 ~ 100)
current_speed = 50
min_speed = 0
max_speed = 100
speed_step = 10
# 当前运动状态: 'STOP', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT'
movement_command = 'STOP'

def send_command(text):
    try:
        global current_speed, movement_command
        # 当识别到前进/后退/左右转时，更新 movement_command 并在 payload 中携带当前速度
        if '前进' == text:
            movement_command = 'FORWARD'
            response = requests.post(control_url, json={'command': "FORWARD", 'speed': current_speed})
        elif '后退' == text:
            movement_command = 'BACKWARD'
            response = requests.post(control_url, json={'command': "BACKWARD", 'speed': current_speed})
        elif '左转' == text:
            movement_command = 'LEFT'
            response = requests.post(control_url, json={'command': "LEFT", 'speed': current_speed})
        elif '右转' in text:
            movement_command = 'RIGHT'
            response = requests.post(control_url, json={'command': "RIGHT", 'speed': current_speed})
        # 加速 / 减速 语音命令
        elif '加速' in text or '快一点' in text or '快点' in text or '提速' in text:
            # only increase if below max
            prev_speed = current_speed
            current_speed = min(max_speed, current_speed + speed_step)
            print(f'速度提升: {prev_speed} -> {current_speed}')
            # 如果正在运动，则同时发送新的速度
            if movement_command in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']:
                response = requests.post(control_url, json={'command': movement_command, 'speed': current_speed})
            else:
                # 发送 SET_SPEED 以兼容服务端实现
                response = requests.post(control_url, json={'command': "SET_SPEED", 'speed': current_speed})
        elif '减速' in text or '慢一点' in text or '慢点' in text or '放慢' in text:
            prev_speed = current_speed
            current_speed = max(min_speed, current_speed - speed_step)
            print(f'速度下降: {prev_speed} -> {current_speed}')
            if movement_command in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']:
                response = requests.post(control_url, json={'command': movement_command, 'speed': current_speed})
            else:
                response = requests.post(control_url, json={'command': "SET_SPEED", 'speed': current_speed})
        else:
            response = requests.post(control_url, json={'command': "STOP"})

        if response.status_code != 200:
            print('小车指令请求失败：', response)
    except Exception as e:
        print('小车指令请求异常：', e)

print('\n正在识别语音指令...')
idx = 1
buffer = []
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                text = asr.transcribe(vad.front.samples, sample_rate=sample_rate)

                vad.pop()
                if len(text):
                    print()
                    print(f'第{idx}句：{text}')
                    if '前进' in text:
                        print('小车指令：前进')
                        send_command('前进')
                    elif '后退' in text:
                        print('小车指令：后退')
                        send_command('后退')
                    elif '左转' in text:
                        print('小车指令：左转')
                        send_command('左转')
                    elif '右转' in text:
                        print('小车指令：右转')
                        send_command('右转')
                    elif '加速' in text or '快一点' in text or '快点' in text or '提速' in text:
                        print('小车指令：加速')
                        send_command('加速')
                    elif '减速' in text or '慢一点' in text or '慢点' in text or '放慢' in text:
                        print('小车指令：减速')
                        send_command('减速')
                    else:
                        print('小车指令：无')
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')
