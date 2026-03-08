import json
data = json.load(open("experiments/results/fusion_e2e_test.json"))
for r in data:
    act = r["last_action"]["action_type"]
    acc = r["accuracy"]
    lat = r["latency"]["total_ms"]
    sp = r["last_action"].get("speech_text", "")[:60]
    print(f'{r["scenario_id"]}: acc={acc}%  lat={lat}ms  action={act}')
    print(f'   speech: {sp}...')
