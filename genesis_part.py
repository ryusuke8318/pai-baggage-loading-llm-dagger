"""Genesis上で、荷物を1個つかんでスロットに置く動作をPPOで強化学習する。

MuJoCo版(DAgger)と同じ設計にそろえてある:
  - 手先 (ガントリー) をxyzに動かし、吸着で荷物を持ち、離して置く
  - 置き位置の許容誤差8mm、動作ノイズ0.6 cm
  - 荷物は剛体で、離したあとの落下・接触はGenesisが計算する
"""
import math, time, json
import numpy as np
import torch
import torch.nn as nn
import genesis as gs

TOL = 0.008            # 置き位置の許容誤差 [m]
ACT_NOISE = 0.006      # 動作ノイズの標準偏差 [m / ステップ]
STEP_MAX = 0.02        # 1ステップに手先が動ける最大量 [m]
EP_LEN = 150
CUBE = 0.05
OBS_DIM, ACT_DIM = 11, 4


class PickPlaceEnv:
    def __init__(self, n_envs, device="cuda", seed=0, render=False):
        self.n, self.dev = n_envs, torch.device(device)
        self.g = torch.Generator(device=self.dev); self.g.manual_seed(seed)
        self.scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.02, substeps=2))
        self.scene.add_entity(gs.morphs.Plane())
        self.render = render
        col = (lambda rgb: dict(surface=gs.surfaces.Default(color=rgb))) if render else (lambda rgb: {})
        self.cube = self.scene.add_entity(gs.morphs.Box(size=(CUBE, CUBE, CUBE), pos=(0.5, 0.0, CUBE / 2)), **col((0.2, 0.4, 0.9)))
        if render:
            self.ee_vis = self.scene.add_entity(gs.morphs.Sphere(radius=0.02, pos=(0.5, 0.0, 0.25), collision=False), **col((0.9, 0.2, 0.15)))
            self.tg_vis = self.scene.add_entity(gs.morphs.Box(size=(0.06, 0.06, 0.002), pos=(0.5, 0.2, 0.001), collision=False), **col((0.2, 0.8, 0.3)))
            self.cam = self.scene.add_camera(res=(640, 480), pos=(1.15, -0.95, 0.75), lookat=(0.5, 0.0, 0.05), fov=40, GUI=False)
        self.scene.build(n_envs=n_envs)
        self.lo = torch.tensor([0.30, -0.35, 0.03], device=self.dev)
        self.hi = torch.tensor([0.70, 0.35, 0.35], device=self.dev)
        self.ee = torch.zeros(n_envs, 3, device=self.dev)
        self.target = torch.zeros(n_envs, 3, device=self.dev)
        self.attached = torch.zeros(n_envs, dtype=torch.bool, device=self.dev)
        self.picked = torch.zeros(n_envs, dtype=torch.bool, device=self.dev)
        self.t = 0
        self.reset()

    def _u(self, n, a, b):
        return a + (b - a) * torch.rand(n, device=self.dev, generator=self.g)

    def reset(self):
        n = self.n
        cx, cy = self._u(n, 0.38, 0.62), self._u(n, -0.25, 0.25)
        tx, ty = self._u(n, 0.38, 0.62), self._u(n, -0.30, 0.30)
        far = torch.hypot(tx - cx, ty - cy) > 0.12                       # 荷物とスロットは離す
        tx = torch.where(far, tx, cx + 0.15 * torch.sign(0.5 - cx + 1e-6))
        ty = torch.where(far, ty, cy)
        self.target = torch.stack([tx, ty, torch.full_like(tx, CUBE / 2)], 1)
        pos = torch.stack([cx, cy, torch.full_like(cx, CUBE / 2)], 1)
        idx = torch.arange(n, device=self.dev)
        self.cube.set_pos(pos, envs_idx=idx, zero_velocity=True)
        self.cube.set_quat(torch.tensor([1.0, 0, 0, 0], device=self.dev).repeat(n, 1), envs_idx=idx, zero_velocity=True)
        self.ee = torch.stack([self._u(n, 0.45, 0.55), self._u(n, -0.05, 0.05), torch.full((n,), 0.25, device=self.dev)], 1)
        self.attached[:] = False; self.picked[:] = False
        self.t = 0
        self.scene.step()
        return self.obs()

    def draw(self):
        """動画用:1フレーム(480,640,3)の画像を返す。"""
        self.ee_vis.set_pos(self.ee, zero_velocity=True)
        self.tg_vis.set_pos(self.target * torch.tensor([1.0, 1.0, 0.0], device=self.dev) + torch.tensor([0.0, 0.0, 0.001], device=self.dev), zero_velocity=True)
        out = self.cam.render()
        rgb = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
        while rgb.ndim > 3:
            rgb = rgb[0]
        return rgb.astype(np.uint8)

    def cube_pos(self):
        return self.cube.get_pos().reshape(self.n, 3)

    def obs(self):
        c = self.cube_pos()
        #8mmの違いがMLPに見えるように、相対位置を20倍に拡大した特徴を使う
        return torch.cat([self.ee, 20.0 * (c - self.ee), 20.0 * (self.target - c), self.attached.float()[:, None],
                          torch.full((self.n, 1), self.t / EP_LEN, device=self.dev)], 1)

    def step(self, a, act_noise=ACT_NOISE):
        a = a.clamp(-1, 1)
        delta = a[:, :3] * STEP_MAX
        if act_noise > 0:
            delta = delta + act_noise * torch.randn(self.n, 3, device=self.dev, generator=self.g)
        self.ee = torch.max(torch.min(self.ee + delta, self.hi), self.lo)
        cmd = a[:, 3] > 0
        c = self.cube_pos()
        near = (torch.hypot(self.ee[:, 0] - c[:, 0], self.ee[:, 1] - c[:, 1]) < 0.02) & \
               ((self.ee[:, 2] - c[:, 2]) > 0.0) & ((self.ee[:, 2] - c[:, 2]) < 0.05)
        newly = cmd & near & ~self.attached
        self.attached = (self.attached | newly) & cmd
        self.picked |= self.attached
        idx = torch.nonzero(self.attached).reshape(-1)
        if idx.numel() > 0:                                               # 吸着中は、荷物を手先に追従させる(MuJoCo版と同じ簡易吸着)
            hold = torch.stack([self.ee[idx, 0], self.ee[idx, 1], self.ee[idx, 2] - 0.03], 1)
            self.cube.set_pos(hold, envs_idx=idx, zero_velocity=True)
        self.scene.step()
        self.t += 1
        c = self.cube_pos()
        d_xy = torch.hypot(c[:, 0] - self.target[:, 0], c[:, 1] - self.target[:, 1])
        d_reach = torch.linalg.norm(self.ee - (c + torch.tensor([0, 0, 0.04], device=self.dev)), dim=1)
        # 報酬: 持ち上げる前は「荷物に近いほど」、持ち上げたあとは「スロットに近いほど」大きい。
        # 持ち上げた状態の報酬は、持ち上げる前の最大値(1.0)より必ず大きくして、その場にとどまる局所解を避ける。
        reach = 1.0 - torch.tanh(5.0 * d_reach)
        carry = 1.0 + (1.0 - torch.tanh(5.0 * d_xy)) + 2.0 * (1.0 - torch.tanh(50.0 * d_xy))
        r = torch.where(self.picked, carry, reach) + 0.5 * newly.float()
        success = (~self.attached) & (d_xy < TOL) & (c[:, 2] < CUBE / 2 + 0.01) & self.picked
        r = r + 3.0 * success.float()
        done = self.t >= EP_LEN
        return self.obs(), r, done, success


class AC(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        mk = lambda o: nn.Sequential(nn.Linear(OBS_DIM, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))
        self.pi, self.v = mk(ACT_DIM), mk(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.0))

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())


@torch.no_grad()
def evaluate(env, ac, act_noise, episodes=1):
    ok = []
    for _ in range(episodes):
        o = env.reset()
        for _ in range(EP_LEN):
            o, r, done, s = env.step(ac.pi(o), act_noise)
        ok.append(s.float().mean().item())
    return float(np.mean(ok))


def train(n_envs=2048, iters=150, act_noise=ACT_NOISE, seed=0, log=print, env=None):
    torch.manual_seed(seed)
    env = env or PickPlaceEnv(n_envs, seed=seed)
    dev = env.dev
    ac = AC().to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    hist = []
    best = (-1.0, None, -1)                                          # (学習中の成功率, 重み, イテレーション)
    t0 = time.time()
    for it in range(iters):
        for gparam in opt.param_groups:                              # 学習率を3e-4から3e-5まで線形に下げる(後半の崩れを防ぐ)
            gparam["lr"] = 3e-4 * (1.0 - 0.9 * it / max(iters - 1, 1))
        O = torch.zeros(EP_LEN, env.n, OBS_DIM, device=dev); A = torch.zeros(EP_LEN, env.n, ACT_DIM, device=dev)
        LP = torch.zeros(EP_LEN, env.n, device=dev); R = torch.zeros(EP_LEN, env.n, device=dev); V = torch.zeros(EP_LEN + 1, env.n, device=dev)
        o = env.reset()
        for t in range(EP_LEN):
            with torch.no_grad():
                d = ac.dist(o); a = d.sample()
                O[t], A[t], LP[t], V[t] = o, a, d.log_prob(a).sum(-1), ac.v(o).squeeze(-1)
            o, r, done, succ = env.step(a, act_noise)
            R[t] = r
        train_succ = succ.float().mean().item()
        c_end = env.cube_pos()
        d_end = torch.hypot(c_end[:, 0] - env.target[:, 0], c_end[:, 1] - env.target[:, 1])
        picked_rate, d_med = env.picked.float().mean().item(), d_end.median().item()
        adv = torch.zeros_like(R); last = 0
        for t in reversed(range(EP_LEN)):
            nv = V[t + 1] if t + 1 < EP_LEN else 0.0
            delta = R[t] + 0.99 * nv - V[t]
            last = delta + 0.99 * 0.95 * last
            adv[t] = last
        ret = adv + V[:EP_LEN]
        Of, Af, LPf, advf, retf = O.reshape(-1, OBS_DIM), A.reshape(-1, ACT_DIM), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        N = Of.shape[0]
        for _ in range(5):
            perm = torch.randperm(N, device=dev)
            for mb in perm.chunk(8):
                d = ac.dist(Of[mb]); lp = d.log_prob(Af[mb]).sum(-1)
                ratio = (lp - LPf[mb]).exp()
                pl = -torch.min(ratio * advf[mb], ratio.clamp(0.8, 1.2) * advf[mb]).mean()
                vl = ((ac.v(Of[mb]).squeeze(-1) - retf[mb]) ** 2).mean()
                opt.zero_grad(); (pl + 0.5 * vl).backward(); nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt.step()
        if train_succ > best[0]:
            best = (train_succ, {k: v.detach().clone() for k, v in ac.state_dict().items()}, it)
        hist.append(dict(iter=it, train_success=train_succ, picked=picked_rate, d_med=d_med, reward=R.sum(0).mean().item(), sec=time.time() - t0))
        if it % 5 == 0 or it == iters - 1:
            log(f"iter {it:3d} | 成功率 {train_succ:.1%} | 持ち上げ {picked_rate:.0%} | 最終距離(中央値) {d_med*100:.1f}cm | 報酬 {hist[-1]['reward']:.2f} | {hist[-1]['sec']:.0f}s")
    if best[1] is not None:                                          # 学習中の成功率が最良だったときの重みを返す
        ac.load_state_dict(best[1]); log(f"最良のイテレーション: {best[2]} (学習中の成功率 {best[0]:.1%})")
    return ac, hist, env




# ---------------------------------------------------------------------------------------------
# 熟練者 (ルールを知っているプログラム)、BC、DAgger。PPOと同じ環境・同じノイズで比べる。
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def expert_action(env):
    """荷物の上まで移動 -> 下りて吸着 -> 持ち上げ -> スロットの上まで移動 -> 下りて、真上に来たら離す。"""
    c, ee, tg = env.cube_pos(), env.ee, env.target
    dev = env.dev
    xy_c = torch.hypot(ee[:, 0] - c[:, 0], ee[:, 1] - c[:, 1])
    xy_t = torch.hypot(c[:, 0] - tg[:, 0], c[:, 1] - tg[:, 1])
    cruise = 0.12
    # 吸着前
    des_pick = torch.stack([c[:, 0], c[:, 1], torch.where(xy_c > 0.010, torch.full_like(xy_c, cruise), c[:, 2] + 0.03)], 1)
    suck = (xy_c < 0.012) & ((ee[:, 2] - c[:, 2]) < 0.045)
    # 吸着中 (荷物は手先の3cm下)
    low = torch.where(xy_t > 0.010, torch.full_like(xy_t, cruise), torch.full_like(xy_t, 0.06))
    des_carry = torch.stack([tg[:, 0], tg[:, 1], low], 1)
    release = (xy_t < 0.005) & (ee[:, 2] < 0.07)
    carry_cmd = ~release
    # 離したあと
    des_done = ee.clone()
    des = torch.where(env.attached[:, None], des_carry, torch.where(env.picked[:, None], des_done, des_pick))
    cmd = torch.where(env.attached, carry_cmd, torch.where(env.picked, torch.zeros_like(suck), suck))
    a = ((des - ee) / STEP_MAX).clamp(-1, 1)
    return torch.cat([a, (cmd.float() * 2 - 1)[:, None]], 1)


class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, ACT_DIM))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def rollout(env, learner, beta, act_noise, collect):
    """learner:MLPかNone。betaの確率で熟練者の動作を実行(DAggerの混合)。collectなら(観測, 熟練者の動作)も返す。"""
    O, E = [], []
    o = env.reset()
    for _ in range(EP_LEN):
        ea = expert_action(env)
        if collect:
            O.append(o); E.append(ea)
        if learner is None:
            a = ea
        else:
            a = learner(o)
            if beta > 0:
                use = (torch.rand(env.n, 1, device=env.dev, generator=env.g) < beta)
                a = torch.where(use, ea, a)
        o, r, done, succ = env.step(a, act_noise)
    out = succ.float().mean().item()
    if collect:
        return out, torch.cat(O), torch.cat(E)
    return out


def fit(model, X, Y, epochs=30, lr=1e-3, bs=8192):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        for g_ in opt.param_groups:
            g_["lr"] = lr * (1 - 0.9 * ep / max(epochs - 1, 1))
        perm = torch.randperm(n, device=X.device)
        for mb in perm.chunk(max(n // bs, 1)):
            loss = ((model(X[mb]) - Y[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate_ctrl(env, learner, act_noise, seed=12345):
    env.g.manual_seed(seed)                                           # 同じ荷物・スロット・ノイズで比べる
    return rollout(env, learner, 0.0, act_noise, collect=False)


def run_imitation(env, rounds=4, log=print):
    res = {}
    res["expert_clean"] = evaluate_ctrl(env, None, 0.0)
    res["expert_noise"] = evaluate_ctrl(env, None, ACT_NOISE)
    log(f"熟練者: ノイズなし {res['expert_clean']:.1%} / ノイズ {res['expert_noise']:.1%}")
    _, X, Y = rollout(env, None, 1.0, 0.0, collect=True)               # 熟練者のデモ(ノイズなし)
    model = bc_model = fit(MLP().to(env.dev), X, Y)
    res["bc_clean"], res["bc_noise"] = evaluate_ctrl(env, model, 0.0), evaluate_ctrl(env, model, ACT_NOISE)
    log(f"BC (デモ {X.shape[0]}): ノイズなし {res['bc_clean']:.1%} / ノイズ {res['bc_noise']:.1%}")
    res["dagger"] = []
    for r in range(1, rounds + 1):
        _, X2, Y2 = rollout(env, model, max(0.4 - 0.2 * r, 0.0), ACT_NOISE, collect=True)   # 評価と同じノイズの下で、学習した方策を動かす
        X, Y = torch.cat([X, X2]), torch.cat([Y, Y2])
        model = fit(MLP().to(env.dev), X, Y)
        row = dict(round=r, samples=int(X.shape[0]), clean=evaluate_ctrl(env, model, 0.0), noise=evaluate_ctrl(env, model, ACT_NOISE))
        res["dagger"].append(row)
        log(f"DAgger ラウンド {r} (データ {row['samples']}): ノイズなし {row['clean']:.1%} / ノイズ {row['noise']:.1%}")
    return res, model, bc_model


def run_all(n_envs=2048, ppo_iters=250, log=print):
    env = PickPlaceEnv(n_envs, seed=0)
    out = {}
    t0 = time.time()
    out["imitation"], dag_model, bc_model = run_imitation(env, 4, log)
    log(f"(模倣学習 {time.time() - t0:.0f} 秒)")
    t0 = time.time()
    ac, hist, env = train(n_envs=n_envs, iters=ppo_iters, env=env, log=log)
    out["ppo"] = dict(clean=evaluate(env, ac, 0.0), noise=evaluate(env, ac, ACT_NOISE), iters=ppo_iters, sec=time.time() - t0,
                      curve=[(h["iter"], h["train_success"]) for h in hist])
    log(f"PPO ({ppo_iters}イテレーション, {out['ppo']['sec']:.0f}秒):ノイズなし{out['ppo']['clean']:.1%} / ノイズ {out['ppo']['noise']:.1%}")
    return out, ac, dag_model, bc_model, env


@torch.no_grad()
def run_episode(env, act_fn, seed, record_every=0):
    env.g.manual_seed(seed)                                           # 同じ荷物・スロット・ノイズ列で比べる
    o = env.reset()
    frames = [env.draw()] if record_every else []
    for t in range(EP_LEN):
        o, r, done, s_ = env.step(act_fn(o), ACT_NOISE)
        if record_every and (t + 1) % record_every == 0:
            frames.append(env.draw())
    return bool(s_[0]), frames


def make_clips(ac, dag_model, bc_model, path="genesis_clips.mp4", every=3, log=print):
    """BC/DAgger/PPOの3つを、同じ荷物・同じノイズで1エピソードずつ描画する。BCが失敗し、DAggerとPPOが成功するseedを探す。"""
    import mediapy
    env = PickPlaceEnv(1, seed=0, render=True)
    pol = [("BC only", lambda o: bc_model(o)), ("DAgger", lambda o: dag_model(o)), ("PPO (RL)", lambda o: ac.pi(o))]
    seed = 700
    for k in range(700, 730):
        ok = [run_episode(env, f, k)[0] for _, f in pol]
        if (not ok[0]) and ok[1] and ok[2]:
            seed = k; break
    log(f"動画用のseed: {seed}")
    allf, info = [], []
    for name, f in pol:
        ok, fr = run_episode(env, f, seed, record_every=every)
        allf += fr; info.append(dict(name=name, frames=len(fr), success=ok))
        log(f"  {name}: {'成功' if ok else '失敗'} ({len(fr)} フレーム)")
    mediapy.write_video(path, np.stack(allf), fps=20)
    return dict(seed=seed, items=info, every=every)


def save_chart(out, path="genesis_compare.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    im, pp = out["imitation"], out["ppo"]
    names = ["Expert (program)", "BC only"] + [f"DAgger round {d['round']}" for d in im["dagger"]] + ["PPO (RL)"]
    vals = [im["expert_noise"], im["bc_noise"]] + [d["noise"] for d in im["dagger"]] + [pp["noise"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    cols = ["#888888", "#d9822b"] + ["#3b6fb6"] * len(im["dagger"]) + ["#2e9e5b"]
    ax.barh(names[::-1], [v * 100 for v in vals][::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 100 + 1, i, f"{v * 100:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 112); ax.set_xlabel("success rate (%) with 0.6cm noise, 8mm tolerance")
    ax.set_title("Genesis: imitation vs reinforcement learning")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    import sys, traceback
    n_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    gs.init(backend=gs.gpu, logging_level="warning")
    out, ac, dag_model, bc_model, env = run_all(n_envs, iters, log=lambda m: print(m, flush=True))
    save_chart(out)
    res = dict(env=dict(n_envs=n_envs, noise_m=ACT_NOISE, tol_m=TOL, ep_len=EP_LEN), imitation=out["imitation"],
               ppo={k: v for k, v in out["ppo"].items() if k != "curve"}, ppo_curve_every10=out["ppo"]["curve"][::10],
               genesis=getattr(gs, "__version__", "?"), torch=torch.__version__, device=torch.cuda.get_device_name(0), clips=None)
    json.dump(res, open("genesis_results.json", "w"), ensure_ascii=False, indent=1)
    torch.save(ac.state_dict(), "genesis_ppo_policy.pt"); torch.save(dag_model.state_dict(), "genesis_dagger_policy.pt")
    try:                                                              # 描画に失敗しても、数値の結果は残す
        res["clips"] = make_clips(ac, dag_model, bc_model, log=lambda m: print(m, flush=True))
    except Exception:
        traceback.print_exc(); print("描画に失敗したため、動画用のクリップは作りません (数値の結果は保存済み)", flush=True)
    json.dump(res, open("genesis_results.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
