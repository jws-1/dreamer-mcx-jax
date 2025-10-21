import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm
from .utils import mean, sample


class Exploration(nj.Module):
    def __init__(
        self,
        config,
        policy: embodied.jax.MLPHead,
        val: embodied.jax.MLPHead,
        encoder: rssm.Encoder,
        dynamics: rssm.RSSM,
        decoder: rssm.Decoder,
    ):
        self.config = config
        self.policy = policy
        self.val = val
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder

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
        enc_carry, enc_entry, tokens = self.encoder(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dynamics.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, recons = self.decoder(dec_carry, feat, reset, **kw)
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

    def loss(self, carry, data):
        return {}, {}


class Random(Exploration):
    def select_action(self, feat):
        policy = self.policy(self.feat2tensor(feat), bdims=1)
        action = sample(policy)
        return action


class Greedy(Exploration):
    def select_action(self, feat):
        policy = self.policy(self.feat2tensor(feat), bdims=1)
        action = mean(policy)
        return action


class Plan2Explore(Exploration):
    def __init__(
        self,
        config,
        policy: embodied.jax.MLPHead,
        val: embodied.jax.MLPHead,
        encoder: rssm.Encoder,
        dynamics: rssm.RSSM,
        decoder: rssm.Decoder,
    ):
        super().__init__(config, policy, val, encoder, dynamics, decoder)

        space = {
            "embed": {"output": elements.Space(np.float32, (encoder.out_dim,))},
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
        print(space)

        scalar = elements.Space(
            np.float32,
        )
        self.modules = [
            embodied.jax.MLPHead(
                space,
                "mse",
                **config.exploration.plan2explore.model,
                name=f"plan2explore_{i}",
            )
            for i in range(config.exploration.plan2explore.num_models)
        ]

    def loss(self, carry, data):
        preds = []
        for net in self.modules:
            # Dummy input — same shape as expected space
            dummy_inp = jnp.zeros((1, 5))
            pred = net(dummy_inp, bdims=1)  # forward pass
            preds.append(pred)

        # Combine predictions into a single scalar
        dummy_loss = sum([jnp.mean(mean(p["output"]) ** 2) for p in preds]) * 0.0 + 0.0

        return {"expl": dummy_loss}, {"expl": dummy_loss}

    def select_action(self, feat):
        policy = self.policy(self.feat2tensor(feat), bdims=1)
        action = mean(policy)
        return action
