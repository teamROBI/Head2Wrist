
def main():
    #!/usr/bin/env python3
    """
    LeRobot training (lerobot-train, same CLI) with a representation transform (env REPR, repr_lib.py): each sampled action
    chunk is re-expressed relative to the sample's current state (joint_rel: q* - q_now; eef_rel: inv(T_now) @ T*), and the
    action normalization statistics are recomputed on the transformed chunks (from the stored numbers, no video decoding)
    before the policy is built. joint_abs / eef_abs: no transform. Run in third_party/lerobot/.venv.
    """
    import os, sys
    import numpy as np, torch
    pass
    from head2wrist.policy.repr_lib import to_rel, T_of, mask_state
    import lerobot.scripts.train as LT
    REPR = os.environ.get('REPR', 'eef_hybrid')
    # AUG=strong: geometric + blur + noise + stronger colour on the wrist image (LeRobot 0.3.3 has colour / sharpness only) -
    # against the generated-vs-true view gap (small misregistration, shading, gripper texture, missing arm shadows)
    AUG = os.environ.get('AUG', 'strong')
    BLANK = os.environ.get('BLANK', '') == '1'   # BLANK=1: the wrist image is a constant grey (a state-only control: the same ACT, no visual input)
    if AUG == 'strong':
        from torchvision.transforms import v2
        _AUG = v2.Compose([v2.RandomApply([v2.RandomAffine(degrees=3, translate=(0.04, 0.04), scale=(0.95, 1.05))], p=0.8),
                           v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.06)], p=0.8),
                           v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.4)])


    class ReprDS(torch.utils.data.Dataset):
        def __init__(self, ds, r): self.ds, self.r = ds, r
        def __len__(self): return len(self.ds)
        def __getattr__(self, k):
            if k in ('ds', 'r'): raise AttributeError(k)
            return getattr(self.ds, k)
        def __getitem__(self, i):
            s = self.ds[i]
            if self.r in ('joint_rel', 'eef_rel', 'eef_rel_ns', 'eef_hybrid'):
                T_now = (T_of(s['aux.eef_abs'].numpy()[:9].astype(float)) if self.r in ('eef_rel', 'eef_rel_ns') else
                         T_of(s['observation.state'].numpy()[:9].astype(float)) if self.r == 'eef_hybrid' else None)   # hybrid: the state IS the current pose
                s['action'] = torch.from_numpy(to_rel(self.r, s['action'].numpy().astype(float), s['observation.state'].numpy(), T_now).astype(np.float32))
            if self.r == 'eef_rel_ns': s['observation.state'] = torch.from_numpy(mask_state(self.r, s['observation.state'].numpy()))
            if AUG == 'strong':
                im = _AUG(s['observation.images.wrist']); s['observation.images.wrist'] = (im + 0.02*torch.randn_like(im)).clamp(0, 1)
            if BLANK: s['observation.images.wrist'] = torch.full_like(s['observation.images.wrist'], 0.5)   
            return s


    _make = LT.make_dataset
    def make_dataset(cfg):
        ds = _make(cfg); w = ReprDS(ds, REPR)
        if REPR in ('joint_rel', 'eef_rel', 'eef_rel_ns', 'eef_hybrid'):
            hf = ds.hf_dataset
            A = torch.stack(list(hf['action'])).numpy().astype(float); S = torch.stack(list(hf['observation.state'])).numpy()
            E = np.asarray(hf['episode_index']).astype(int); Aux = torch.stack(list(hf['aux.eef_abs'])).numpy() if REPR in ('eef_rel', 'eef_rel_ns') else None
            H = cfg.policy.chunk_size if hasattr(cfg.policy, 'chunk_size') else cfg.policy.horizon
            rng = np.random.default_rng(0); out = []
            for t in rng.choice(len(A), min(6000, len(A)), replace=False):
                j = np.arange(t, min(t + H, len(A))); j = j[E[j] == E[t]]
                out.append(to_rel(REPR, A[j], S[t], T_of(Aux[t, :9]) if Aux is not None else T_of(S[t, :9].astype(float)) if REPR == 'eef_hybrid' else None))
            R = np.concatenate(out); st = ds.meta.stats['action']
            for k, v in (('mean', R.mean(0)), ('std', R.std(0) + 1e-6), ('min', R.min(0)), ('max', R.max(0))):
                st[k] = v.astype(np.float32) if not torch.is_tensor(st[k]) else torch.tensor(v, dtype=st[k].dtype)
            print(f'[train_repr] {REPR}: action stats from {len(R)} transformed steps, std {np.round(R.std(0), 4).tolist()}', flush=True)
        return w
    LT.make_dataset = make_dataset

    if __name__ == '__main__':
        LT.main()


if __name__ == '__main__':
    main()
