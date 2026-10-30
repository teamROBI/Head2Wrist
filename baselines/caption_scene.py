from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Scene prompt for the background video model, written from the head frame instead of by hand (prompts.py has hand-written
ones for the first 4 scenes only). Cosmos-Reason1-7B looks at <task_inputs>/first_frame.png (hands and the grasped object
removed: what the background shows) and fills a structured description - table surface, backdrop / walls, lighting, the
objects with colour and material - which goes into the same template as the hand-written prompts.
Writes <task_inputs>/proxy/prompt.json: {fields..., prompt}.
"""
import argparse, glob, json, os, re
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration





def main():
    global ASK, T, TEMPLATE, a, ap, bg, f, img, inp, lc, m, model, msgs, name, o, objs, olist, out, proc, txt
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--model', default=glob.glob(f'{H2W_DATA_ROOT}/checkpoints/hf_hub/models--nvidia--Cosmos-Reason1-7B/snapshots/*/')[0])
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    os.makedirs(f'{T}/proxy', exist_ok=True)
    ASK = ('Describe this tabletop scene for a video generator. Look carefully at the actual colours in the image. Answer '
           'with JSON only, no other text, with these keys:\n'
           '"table": at most 10 words: the table top as "a <colour> <material> table" or "a table covered with a <colour> '
           '<material>",\n'
           '"background": at most 15 words: the backdrop and walls behind the table with their colours; no positions, no '
           'interpretation,\n'
           '"lighting": at most 4 words,\n'
           '"objects": a list with one entry per object lying on the table, each at most 6 words: "a <colour> <material> '
           '<object name>"; no positions.\n'
           'Do not mention hands or people.')
    TEMPLATE = ('A slow, steady close-up video from a small camera on a robot wrist moving over {table}, {lighting}. Behind '
                'the table: {background}. On the table {are} {objects}. The camera looks down at the tabletop at close range. '
                'Realistic, sharp, natural colours, soft shadows, no people, no hands, no robot visible.')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cuda').eval()
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=256*28*28, max_pixels=1280*28*28)
    img = Image.open(f'{T}/first_frame.png').convert('RGB')
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': 'You are a careful visual describer. Answer directly.'}]},
            {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': ASK}]}]
    inp = proc(text=[proc.apply_chat_template(msgs, add_generation_prompt=True)], images=[img], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=600, do_sample=False)
    txt = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r'\{.*\}', re.sub(r'<think>.*?</think>', '', txt, flags=re.S), re.S)
    assert m, f'no JSON in the answer:\n{txt}'
    f = json.loads(m.group(0))
    lc = lambda x: x[:1].lower() + x[1:] if x[1:2].islower() else x
    bg = re.sub(r'^(the )?(background|backdrop)( behind the table)?( is| consists of| features| includes)?:?\s*', '',
                f['background'].strip().split('. ')[0], flags=re.I).rstrip('.')
    objs = [lc(o.strip().rstrip('.')) for o in f.get('objects', []) if o.strip()]
    olist = 'nothing' if not objs else objs[0] if len(objs) == 1 else ', '.join(objs[:-1]) + ' and ' + objs[-1]
    f['prompt'] = TEMPLATE.format(table=lc(f['table'].strip().rstrip('.')), background=lc(bg),
                                  lighting=lc(f['lighting'].strip().rstrip('.')), are='is' if len(objs) == 1 else 'are', objects=olist)
    f['raw'] = txt
    json.dump(f, open(f'{T}/proxy/prompt.json', 'w'), indent=1)
    print(f['prompt'])


if __name__ == '__main__':
    main()
