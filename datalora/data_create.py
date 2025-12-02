import json, random, itertools, os

actions = ['前进','后退','左转','右转']
speeds  = ['加速','减速','快一点','慢一点','提速','放慢']
templates_single = [
    "{verb}","小车{verb}","请{verb}","现在{verb}吧","马上{verb}"
]
templates_seq_by_len = {
    2: [
        "先{0}再{1}",
        "请{0}，接着{1}",
        "{0}，随后{1}",
        "{0}，然后{1}"
    ],
    3: [
        "先{0}再{1}然后{2}",
        "请{0}，接着{1}，最后{2}",
        "小车{0}，{1}，再{2}",
        "{0}，随后{1}，接着{2}",
        "{0}，{1}，再{2}，最后停"
    ],
    4: [
        "先{0}再{1}随后{2}然后{3}",
        "请{0}，接着{1}，再{2}，最后{3}",
        "小车{0}，{1}，{2}，然后{3}",
    ],
    5: [
        "先{0}再{1}随后{2}然后{3}最后{4}",
        "请{0}，接着{1}，接着{2}，然后{3}，最后{4}",
    ]
}

def mk_single():
    a = random.choice(actions)
    t = random.choice(templates_single)
    if random.random() < 0.4:  # 40% include speed change
        s = random.choice(speeds)
        sent = t.format(verb=a + s)
        spd = 70 if s in ['加速', '快一点', '提速'] else 30
        cmd = 'FORWARD' if a == '前进' else ('BACKWARD' if a == '后退' else ('LEFT' if a == '左转' else 'RIGHT'))
        assistant_out = json.dumps({'command': cmd, 'speed': spd}, ensure_ascii=False)
    else:
        sent = t.format(verb=a)
        cmd = 'FORWARD' if a == '前进' else ('BACKWARD' if a == '后退' else ('LEFT' if a == '左转' else 'RIGHT'))
        assistant_out = json.dumps({'command': cmd, 'speed': 50}, ensure_ascii=False)

    # Optionally add a system instruction (10% chance)
    messages = []
    if random.random() < 0.1:
        messages.append({
            'role': 'system',
            'content': '你是小车控制的语义解析模块。请仅按要求输出控制指令，不输出额外文字。'
        })
    messages.append({'role': 'user', 'content': sent})
    messages.append({'role': 'assistant', 'content': assistant_out})
    return {'messages': messages}

def mk_seq(n=3):
    # If you need more unique actions than we have, allow repeats (sample or choices)
    if n <= len(actions):
        acts = random.sample(actions, n)
    else:
        acts = random.choices(actions, k=n)
    # Choose a template that matches the number of placeholders
    tmpl_list = templates_seq_by_len.get(n)
    if tmpl_list is None:
        # Fallback to a simple comma separated sentence
        sent = '，'.join(acts)
    else:
        tmpl = random.choice(tmpl_list)
        sent = tmpl.format(*acts)
    outs = []
    for a in acts:
        outs.append({'command': ('FORWARD' if a == '前进' else ('BACKWARD' if a == '后退' else ('LEFT' if a == '左转' else 'RIGHT'))), 'speed': 50})

    assistant_out = json.dumps(outs, ensure_ascii=False)
    messages = []
    if random.random() < 0.1:
        messages.append({
            'role': 'system',
            'content': '你是小车控制指令的解析模块。请输出 JSON 数组，元素为动作对象，数组顺序代表执行顺序。'
        })
    messages.append({'role': 'user', 'content': sent})
    messages.append({'role': 'assistant', 'content': assistant_out})
    return {'messages': messages}

# 生成 400 单条
single = [mk_single() for _ in range(400)]
# 生成 1000 条序列
seq2 = [mk_seq(2) for _ in range(200)]
seq3 = [mk_seq(3) for _ in range(500)]
seq4 = [mk_seq(4) for _ in range(200)]
seq5 = [mk_seq(5) for _ in range(100)]
# 兜底
stop = []
for w in ["停", "停下", "停止", "急停", "别动了"]:
    assistant_out = json.dumps({'command': 'STOP', 'speed': 0}, ensure_ascii=False)
    messages = [{'role': 'user', 'content': w}, {'role': 'assistant', 'content': assistant_out}]
    stop.append({'messages': messages})

all_data = single + seq2 + seq3 + seq4 + seq5 + stop
random.shuffle(all_data)
out_path = os.path.join(os.path.dirname(__file__), 'raspberry_car_1500.jsonl')
with open(out_path, 'w', encoding='utf-8') as f:
    for d in all_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')