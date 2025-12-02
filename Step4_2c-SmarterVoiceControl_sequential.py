import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests
import time
import json


asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

def get_command(text: str) -> list:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:d4mpmc9719ns73co0s30:multi_instruct:csufrakfiutgvfhbakjt",
        "messages": [
            {
                "role": "user",
                "content": f"""你是小车控制模块。请根据语音识别结果"{text}"，**严格按照语音中提到的顺序**提取动作，并以**JSON数组**格式返回。

**关键要求：**
1. **严格按照顺序**：必须完全按照语音识别结果中提到的顺序来提取动作，不要改变顺序，不要添加语音中没有提到的动作。
2. **必须返回纯 JSON 数组格式**，不要返回对象格式。
   - 正确格式：["后退", "左转"]
   - 错误格式：{{"command": "RIGHT"}} 或 {{"commands": ["前进"]}}

**动作列表：**
数组元素只能是这些动作：前进、后退、左转、右转、停止。

**匹配规则：**
- 前进：前进、往前走、向前、直走、往前等
- 后退：后退、往后退、倒车、向后、退后等
- 左转：左转、向左转、左拐、往左、向左等
- 右转：右转、向右转、右拐、往右、向右等
- 停止：停止、停下、停等
- 无操作：如果语音中没有明确的控制指令，返回空数组 []

**顺序示例（非常重要）：**
- 语音："先后退再左转" → 返回：["后退", "左转"]
- 语音："先前进再右转" → 返回：["前进", "右转"]
- 语音："先左转再前进" → 返回：["左转", "前进"]
- 语音："后退然后左转" → 返回：["后退", "左转"]
- 语音："前进" → 返回：["前进"]
- 语音："你好" → 返回：[]

**重要提醒：**
- 严格按照语音中的顺序，不要改变顺序
- 不要添加语音中没有提到的动作
- 只输出动作名称的 JSON 数组，不要添加任何其他文字或解释
- 如果语音中没有明确的控制指令，返回空数组 []

回答："""
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "enable_thinking": False,
        "thinking_budget": 512,
        "min_p": 0.05,
        "stop": None,
        "temperature": 0.2,  # 降低温度以提高输出稳定性，适合分类任务
        "top_p": 0.3,  # 降低top_p以提高确定性
        "top_k": 10,  # 降低top_k，只考虑最可能的选项
        "frequency_penalty": 0.0,  # 移除频率惩罚，避免影响固定输出
        "n": 1,
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": "Bearer sk-diashhelzuktcatjjjvayunkrueyvchoapfhlnvdrwtsnocp",
        "Content-Type": "application/json"
    }
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # 检查HTTP状态码
        
        response_data = json.loads(response.text)
        
        # 检查响应中是否有 'choices' 字段
        if 'choices' not in response_data:
            print('大模型请求失败：响应中缺少 choices 字段')
            print('响应内容：', response.text[:500])
            return []
        
        if not response_data['choices'] or len(response_data['choices']) == 0:
            print('大模型请求失败：choices 数组为空')
            print('响应内容：', response.text[:500])
            return []
        
        r_content = response_data['choices'][0]['message']['content'].strip()
        
        # 只有当 enable_thinking 为 True 时才尝试获取推理过程
        r_reason = None
        if 'reasoning_content' in response_data['choices'][0]['message']:
            r_reason = response_data['choices'][0]['message']['reasoning_content'].strip()

        print('大模型回复：', r_content)
        if r_reason:
            print('大模型推理过程：', r_reason)
        
        # 英文命令到中文的映射
        en_to_cn = {
            'FORWARD': '前进', 'forward': '前进', 'Forward': '前进',
            'BACKWARD': '后退', 'backward': '后退', 'Backward': '后退',
            'LEFT': '左转', 'left': '左转', 'Left': '左转',
            'RIGHT': '右转', 'right': '右转', 'Right': '右转',
            'STOP': '停止', 'stop': '停止', 'Stop': '停止'
        }
        
        # 尝试直接解析模型返回的 JSON
        try:
            parsed = json.loads(r_content)
            
            # 如果是数组格式（期望的格式）
            if isinstance(parsed, list):
                # 仅保留合法命令
                legal_commands = []
                for c in parsed:
                    if isinstance(c, str):
                        cmd = c.strip()
                        # 如果是英文命令，转换为中文
                        if cmd in en_to_cn:
                            cmd = en_to_cn[cmd]
                        if cmd in ['前进', '后退', '左转', '右转', '停止']:
                            legal_commands.append(cmd)
                if len(legal_commands) > 0:
                    return legal_commands
            
            # 如果是对象格式（如 {"command": "RIGHT", "speed": 50}）
            elif isinstance(parsed, dict):
                if 'command' in parsed:
                    cmd = str(parsed['command']).strip()
                    # 转换为中文
                    if cmd in en_to_cn:
                        cmd = en_to_cn[cmd]
                    if cmd in ['前进', '后退', '左转', '右转', '停止']:
                        return [cmd]
                # 如果对象中有 commands 字段（数组）
                if 'commands' in parsed and isinstance(parsed['commands'], list):
                    legal_commands = []
                    for c in parsed['commands']:
                        if isinstance(c, str):
                            cmd = c.strip()
                            if cmd in en_to_cn:
                                cmd = en_to_cn[cmd]
                            if cmd in ['前进', '后退', '左转', '右转', '停止']:
                                legal_commands.append(cmd)
                    if len(legal_commands) > 0:
                        return legal_commands
        except json.JSONDecodeError:
            pass  # 如果不是 JSON，继续下面的文本匹配
        
        # 如果不是严格 JSON，则尝试从文本中抽取命令关键词顺序
        commands = []
        order_keywords = ['前进', '后退', '左转', '右转', '停止']
        for kw in order_keywords:
            # 按出现顺序查找，允许重复
            start = 0
            while True:
                idx = r_content.find(kw, start)
                if idx == -1:
                    break
                commands.append(kw)
                start = idx + len(kw)
        
        # 如果没找到中文关键词，尝试查找英文命令
        if len(commands) == 0:
            for en_cmd, cn_cmd in en_to_cn.items():
                if en_cmd in r_content:
                    commands.append(cn_cmd)
        
        if len(commands) > 0:
            return commands
    except requests.exceptions.RequestException as e:
        print('大模型请求失败（网络错误）：', e)
        if hasattr(e, 'response') and e.response is not None:
            print('响应状态码：', e.response.status_code)
            print('响应内容：', e.response.text[:500])
    except KeyError as e:
        print('大模型请求失败（响应格式错误）：', e)
        print('响应内容：', response.text[:500] if 'response' in locals() else '无响应')
    except json.JSONDecodeError as e:
        print('大模型请求失败（JSON解析错误）：', e)
        print('响应内容：', response.text[:500] if 'response' in locals() else '无响应')
    except Exception as e:
        print('大模型请求失败（未知错误）：', e)
        print('错误类型：', type(e).__name__)
        if 'response' in locals():
            print('响应内容：', response.text[:500])
    return []

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

print('正在加载ASR模型...')
asr = Paraformer(
    model_path=f'{asr_path}/model.int8.onnx',
    tokens_path=f'{asr_path}/tokens.txt',
    # provider='cuda',
)

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

control_url = "http://172.20.10.7:5000/control"  

def send_command(text):
    try:
        if '前进' == text:
            response = requests.post(control_url, json={'command': "FORWARD"})
        elif '后退' == text:
            response = requests.post(control_url, json={'command': "BACKWARD"})
        elif '左转' == text:
            response = requests.post(control_url, json={'command': "LEFT"})
        elif '右转' in text:
            response = requests.post(control_url, json={'command': "RIGHT"})
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
                    commands = get_command(text)
                    if len(commands) == 0:
                        print('未识别到小车指令')
                    else:
                        print('识别到小车指令序列：', commands)
                        # 依次执行每个命令，每个命令执行 3s
                        for cmd in commands:
                            print('执行指令：', cmd)
                            send_command(cmd)
                            # 保持 3 秒
                            time.sleep(3)
                        # 执行序列完毕后发送停止命令
                        send_command('停止')
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')
