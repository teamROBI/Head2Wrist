from head2wrist.config import H2W_OUTPUT_ROOT
"""Scene prompts for the background video model (Cosmos-Transfer2.5, Cosmos 3): the scene as the head camera shows it,
as seen by a wrist camera; background only. Hand-written for the first 4 scenes; every other scene uses the one
caption_scene.py wrote from the head frame (<task_inputs>/proxy/prompt.json). get_prompt(ep, src): src 'hand', 'auto', or
'any' (hand-written if there is one, else auto)."""
import json, os
PROMPTS = {
    'basic_pick_place/74': (
        'A slow, steady close-up video from a small camera on a robot wrist moving over a table covered with a smooth '
        'peach-pink cloth, in a room with a white fabric backdrop and warm indoor lighting. On the table are a dark teal '
        'ceramic bowl, a small yellow plastic bottle, a grey plastic tool with a light blue handle and a yellow '
        'clothespin. The camera looks down at the tabletop at close range. Realistic, sharp, natural colours, soft '
        'shadows, no people, no hands, no robot visible.'),
    'stack/18': (
        'A slow, steady close-up video from a small camera on a robot wrist moving over a table covered with a deep red '
        'cloth with soft folds, in a room with a white fabric backdrop, cream walls and warm indoor lighting. On the table '
        'is a small closed kraft-brown cardboard box. The camera looks down at the tabletop at close range. Realistic, '
        'sharp, natural colours, soft shadows, no people, no hands, no robot visible.'),
    'basic_pick_place/25': (
        'A slow, steady close-up video from a small camera on a robot wrist moving over a light beige table in front of a '
        'dark navy fabric backdrop and a beige wall, warm indoor lighting. On the table are a yellow foam die with black '
        'dots, a black jar lid lying flat, a small orange and blue battery, a black plastic clip and a small white ceramic '
        'cup. The camera looks down at the tabletop at close range. Realistic, sharp, natural colours, soft shadows, no '
        'people, no hands, no robot visible.'),
    'basic_pick_place/1': (
        'A slow, steady close-up video from a small camera on a robot wrist moving over a light grey table against a grey '
        'wall with a row of power outlets, soft indoor lighting. On the table are a white cardboard box lid used as a tray, '
        'a rainbow silicone pop-it toy, a pair of pliers with red and yellow handles and a colourful puzzle cube. The '
        'camera looks down at the tabletop at close range. Realistic, sharp, natural colours, soft shadows, no people, no '
        'hands, no robot visible.'),
}

# blockout (skeleton proxy) runs: the scene-intent part says what each primitive stands for, as in the Blender -> AI
# video workflows (fal H3 Max "scene intent", Higgsfield Scene Builder)
BLOCKOUT_INTENT = {
    'basic_pick_place/74': (
        ' The input is a simple 3D blockout of this exact scene: the dark teal open cylinder is a dark teal ceramic bowl, '
        'the yellow box is a small yellow plastic bottle, the grey box is a grey plastic tool with a light blue handle, '
        'the small yellow cylinder is a yellow clothespin, the orange slab is the table covered with a smooth peach-pink '
        'cloth, the flat panels are a white fabric backdrop and a cream wall. Keep the scene exactly as in the reference '
        'image: same objects, same colours, same cloth.'),
}


def get_prompt(ep, src='any'):
    auto = f"{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{ep.replace('/', '_')}/proxy/prompt.json"
    if src == 'hand' or (src == 'any' and ep in PROMPTS):
        return PROMPTS[ep]
    assert os.path.exists(auto), f'no scene prompt for {ep}: run caption_scene.py'
    return json.load(open(auto))['prompt']
