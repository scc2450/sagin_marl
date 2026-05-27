from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _copy_module_state(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init_actor", type=str, required=True)
    parser.add_argument("--init_critic", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    base_bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))
    fresh_bundle = build_structured_modules_from_config(cfg, hidden_dim=int(args.hidden_dim), embed_dim=int(args.embed_dim))

    load_checkpoint_forgiving(base_bundle.actor, args.init_actor, map_location=device, strict=True)
    load_checkpoint_forgiving(base_bundle.critic, args.init_critic, map_location=device, strict=True)

    # Reset the entire BW actor branch to fresh initialization while preserving accel/sat.
    _copy_module_state(base_bundle.actor.bw_policy, fresh_bundle.actor.bw_policy)

    # Reset only the BW-specific critic branch while preserving the shared critic trunk
    # and accel/sat heads from the base checkpoint.
    _copy_module_state(base_bundle.critic.stage_bw_adaptor, fresh_bundle.critic.stage_bw_adaptor)
    _copy_module_state(base_bundle.critic.team_fusion_bw, fresh_bundle.critic.team_fusion_bw)
    _copy_module_state(base_bundle.critic.value_bw_head, fresh_bundle.critic.value_bw_head)
    with torch.no_grad():
        base_bundle.critic.stage_embedding.weight[2].copy_(fresh_bundle.critic.stage_embedding.weight[2])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor_out = out_dir / "actor_bwreset.pt"
    critic_out = out_dir / "critic_bwreset.pt"
    torch.save(base_bundle.actor.state_dict(), actor_out)
    torch.save(base_bundle.critic.state_dict(), critic_out)
    print(f"Saved {actor_out}")
    print(f"Saved {critic_out}")


if __name__ == "__main__":
    main()
