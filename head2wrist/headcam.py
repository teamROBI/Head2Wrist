from head2wrist.config import H2W_OUTPUT_ROOT
"""
The head camera's focal length for an episode. EgoDex (Vision Pro): 736.6 px at 1920x1080 with the principal point at
the image centre - the value the pipeline was written for. Other capture devices (capture/egokit_to_egodex.py) are
converted to the same 1920x1080 frame with the principal point at the centre, so only the focal length differs: the
converter marks the hdf5 (attrs capture_device, camera/intrinsic), the prep stage (make_hover_tour.py) writes it to the
inputs' meta.json as 'head_focal', make_task_inputs.py carries it on. Episodes without it are EgoDex: 736.6.
"""
import json, os

F_EGODEX, W0, H0 = 736.6, 1920, 1080


def focal(inputs_dir):
    """focal length (px at 1920x1080) of the episode whose explore / task inputs folder this is"""
    m = os.path.join(inputs_dir, 'meta.json')
    if os.path.exists(m):
        f = json.load(open(m)).get('head_focal')
        if f: return float(f)
    return F_EGODEX


def focal_for_ep(ep):
    """the same from an episode name ('task/idx' or 'task_idx'), via its explore inputs"""
    return focal(f"{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{ep.replace('/', '_')}")
