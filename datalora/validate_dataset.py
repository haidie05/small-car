import json
from pathlib import Path


def validate_line(obj):
    # Rule 1: Each line is an independent JSON object: outer code will handle this
    if not isinstance(obj, dict):
        return False, 'Not a JSON object at top level'
    # Rule 2: must contain messages array and not empty
    if 'messages' not in obj:
        return False, 'Missing messages key'
    messages = obj['messages']
    if not isinstance(messages, list) or len(messages) == 0:
        return False, 'messages must be a non-empty list'
    # Rule 3: each message must have role and content
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False, f'message[{i}] is not an object'
        if 'role' not in m or 'content' not in m:
            return False, f'message[{i}] missing role/content'
    # Rule 4: role must be system/user/assistant
    valid_roles = {'system', 'user', 'assistant'}
    for i, m in enumerate(messages):
        if m['role'] not in valid_roles:
            return False, f'message[{i}] invalid role: {m["role"]}'
    # Rule 5: if system message exists, must be first
    roles = [m['role'] for m in messages]
    if 'system' in roles and roles[0] != 'system':
        return False, 'system role must be at messages[0]'
    # Rule 6: first non-system must be user
    first_non_sys = None
    for r in roles:
        if r != 'system':
            first_non_sys = r
            break
    if first_non_sys != 'user':
        return False, 'First non-system message must be user'
    # Rule 7: user and assistant must alternate, appear in pairs, at least one pair
    filtered = [r for r in roles if r in ('user','assistant')]
    if len(filtered) < 2:
        return False, 'Need at least one user+assistant pair'
    # They should alternate starting with user
    for i in range(0, len(filtered), 2):
        if i+1 >= len(filtered):
            return False, 'Unpaired user message at end of messages array'
        if not (filtered[i] == 'user' and filtered[i+1] == 'assistant'):
            return False, f'user and assistant must alternate in pairs, found pair at index {i}: {filtered[i]}/{filtered[i+1]}'
    # All good
    return True, 'OK'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='path to jsonl file')
    parser.add_argument('--max-lines', type=int, default=5000, help='maximum allowed lines')
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print('File not found:', args.file)
        raise SystemExit(1)

    errors = []
    total = 0
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append((idx, f'JSON parse error: {e}'))
                continue
            ok, msg = validate_line(obj)
            if not ok:
                errors.append((idx, msg))
            total += 1
            if total > args.max_lines:
                errors.append((idx, f'Exceeded max lines ({args.max_lines})'))
                break
    print(f'Total lines checked: {total}')
    if errors:
        print('Found errors:')
        for e in errors[:50]:
            print(f'Line {e[0]}: {e[1]}')
        print(f'...and {len(errors)-50} more' if len(errors) > 50 else '')
        raise SystemExit(2)
    else:
        print('Validation passed ✅')
