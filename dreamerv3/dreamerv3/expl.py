import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm
from .utils import concat, imag_loss, mean, prefix, sample, sg


class Exploration(nj.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        config,
        pol: embodied.jax.MLPHead,
        val: embodied.jax.MLPHead,
        rew: embodied.jax.MLPHead,
        con: embodied.jax.MLPHead,
        enc: rssm.Encoder,
        dyn: rssm.RSSM,
        dec: rssm.Decoder,
    ):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.pol = pol
        self.val = val
        self.rew = rew
        self.con = con
        self.enc = enc
        self.dyn = dyn
        self.dec = dec

        self.feat2tensor = lambda x: jnp.concatenate(
            [
                nn.cast(x["deter"]),
                nn.cast(x["stoch"].reshape((*x["stoch"].shape[:-2], -1))),
            ],
            -1,
        )
        self.modules = []

    def select_action(self, feat):
        raise NotImplementedError

    def act(self, carry, obs):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
        act = self.select_action(feat)
        out = {}
        out["finite"] = elements.tree.flatdict(
            jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
            )
        )
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            out.update(
                elements.tree.flatdict(
                    dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
                )
            )
        return carry, act, out

    def loss(self, carry, obs, prevact):
        return {}, {}


class Random(Exploration):
    def select_action(self, feat):
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        action = sample(policy)
        return action


class Greedy(Exploration):
    def select_action(self, feat):
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        action = mean(policy)
        return action


class Plan2Explore(Exploration):
    def __init__(
        self,
        obs_space,
        act_space,
        config,
        pol: embodied.jax.MLPHead,
        val: embodied.jax.MLPHead,
        rew: embodied.jax.MLPHead,
        con: embodied.jax.MLPHead,
        enc: rssm.Encoder,
        dyn: rssm.RSSM,
        dec: rssm.Decoder,
    ):
        super().__init__(
            obs_space, act_space, config, pol, val, rew, con, enc, dyn, dec
        )

        space = {
            "embed": {"output": elements.Space(np.float32, (enc.out_dim,))},
            "stoch": {
                "output": elements.Space(
                    np.float32,
                    (
                        config.dyn[config.dyn.typ].stoch
                        * config.dyn[config.dyn.typ].classes,
                    ),
                )
            },
            "deter": {
                "output": elements.Space(
                    np.float32, (config.dyn[config.dyn.typ].deter,)
                )
            },
            "feat": {
                "output": elements.Space(
                    np.float32,
                    (
                        config.dyn[config.dyn.typ].deter
                        + config.dyn[config.dyn.typ].stoch
                        * config.dyn[config.dyn.typ].classes,
                    ),
                )
            },
        }[config.exploration.plan2explore.disag_target]

        scalar = elements.Space(np.float32, ())

        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.expl_pol = embodied.jax.MLPHead(
            act_space, outs, **config.policy, name="expl_pol"
        )

        self.expl_val = embodied.jax.MLPHead(scalar, **config.value, name="expl_val")
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="expl_slowval"),
            source=self.val,
            **config.slowvalue,
        )

        self.retnorm = embodied.jax.Normalize(**config.retnorm, name="expl_retnorm")
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name="expl_valnorm")
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name="expl_advnorm")

        self.modules = [
            embodied.jax.MLPHead(
                space,
                "mse",
                **config.exploration.plan2explore.model,
                name=f"plan2explore_{i}",
            )
            for i in range(config.exploration.plan2explore.num_models)
        ] + [
            self.expl_pol,
            self.expl_val,
            self.slowval,
        ]

    def loss(self, carry, obs, prevact):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        B, T = reset.shape
        losses = {}
        metrics = {}

        training = True

        # World model
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, repfeat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, training
        )
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)

        # Imagination
        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)
        policyfn = lambda feat: sample(self.expl_pol(self.feat2tensor(feat), 1))
        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
        first = jax.tree.map(
            lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat
        )
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
        inp = self.feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            # self.config.exploration.plan2explore.extrinsic_scale
            1.0 * self.rew(inp, 2).pred()
            + self.config.exploration.plan2explore.intrinsic_scale
            * self.intrinsic_rew(inp),
            self.con(inp, 2).prob(1),
            self.expl_pol(inp, 2),
            self.expl_val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        losses.update(
            prefix({k: v.mean(1).reshape((B, K)) for k, v in los.items()}, "expl")
        )
        metrics.update(prefix(mets, "expl_imag"))

        # Replay
        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
            boot = imgloss_out["ret"][:, 0].reshape(B, K)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot)
            )
            inp = self.feat2tensor(feat)
            los, reploss_out, mets = repl_loss(
                last,
                term,
                rew,
                boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(prefix(los, "expl"))
            metrics.update(prefix(mets, "expl_reploss"))

        return losses, metrics

    def select_action(self, feat):
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        action = mean(policy)
        return action

    def intrinsic_rew(self, inp):
        return jnp.zeros((), np.float32)
