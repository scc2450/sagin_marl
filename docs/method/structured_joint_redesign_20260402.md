# Structured Joint Redesign Blueprint

## 1. Purpose

This document proposes a full redesign of the current joint control system for `accel`, `bw`, and `sat`.
The redesign is derived from the problem itself rather than from existing code paths.

The proposal targets three confirmed problem classes:

1. **Structured-action mismatch**
   The current task is a strongly coupled joint control problem, but the actor models it as parallel factorized heads.

2. **Coarse cross-head credit**
   Shared scalar advantage can reward a locally bad action head when other heads happened to act well in the same step.

3. **A real `bw` head-internal optimization pathology**
   The current `bw` training geometry shows a confirmed `valid_count` / concentration-driven instability under PPO.

The redesign therefore treats the problem as a **structured joint-action control problem** and rebuilds:

- the actor factorization,
- the environment control cycle,
- the centralized critic input,
- the rollout representation,
- the PPO/GAE training target.


## 2. Design Principles

### 2.1 No backward compatibility as a design goal

No component is preserved merely because it already exists in code.
If the current environment step, state representation, policy factorization, or critic interface is part of the problem, it should be changed.

### 2.2 Preserve the task objective, not the legacy implementation

The end task remains:

- improve end-to-end delivered traffic / service,
- reduce backlog and drops,
- satisfy safety and feasibility constraints.

However, preserving this task objective does **not** imply preserving:

- the current `env.step` order,
- the current pseudo-global critic state,
- the current factorized actor structure,
- the current `bw` distribution,
- or the current reward formula in every detail.

### 2.3 Match model structure to control semantics

The three control decisions are not symmetric:

- `accel` changes geometry and future feasibility,
- `sat` selects relay/backhaul support,
- `bw` allocates access-side resources under the chosen geometry and relay context.

The policy and critic should therefore represent these decisions with **explicit conditional structure**, not as exchangeable parallel heads.

### 2.4 Use permutation-invariant processing without early information loss

The problem naturally involves sets:

- UAVs,
- GUs,
- satellites,
- pairwise candidate relations.

Permutation invariance is required, but simple early mean/max pooling is insufficient.
The model must keep **element-level and edge-level representations** until late readout.


## 3. Problem Diagnosis Summary

### 3.1 Why the current actor factorization is mismatched

The current actor is effectively:

```text
pi(a_accel, a_bw, a_sat | o) = pi_accel(a_accel|o) * pi_bw(a_bw|o) * pi_sat(a_sat|o)
```

This is mismatched for the current task because:

- `accel` changes the working geometry seen by both `bw` and `sat`,
- `sat` is not a simple one-of-K choice; it is an unordered `2-of-K` combinatorial choice,
- `bw` is a masked simplex allocation over a dynamic valid set,
- all three decisions interact through queues, coverage, and relay capacity.

### 3.2 Why the current critic/advantage pipeline is too coarse

The current pipeline uses shared scalar return / advantage logic.
This is often sufficient in many MAPPO settings, but the present system exhibits direct evidence that it is too coarse here:

- locally bad `sat` actions can receive positive joint advantage,
- `accel` strongly changes the regime in which the other heads operate,
- `bw` carries a separate optimization pathology that contaminates the joint update.

The redesign must therefore reduce cross-head credit mixing.

### 3.3 Why `bw` must be re-parameterized

The redesign does **not** assume that every Dirichlet-based policy is invalid in principle.
It does, however, conclude that the current `bw` parameterization and PPO geometry are not acceptable.

The new `bw` parameterization must avoid:

- direct domination by `valid_count`,
- concentration-driven geometry instability,
- opaque coupling between valid-set size and policy ratio.


## 4. Control Semantics Redesign

### 4.1 The environment control cycle should be redesigned

The current `env.step` ordering should not be treated as fixed.
If its order is semantically wrong for the task, it should be changed.

The redesigned control cycle for a single physical slot is:

1. **Motion update stage**
   The system first applies `accel` and obtains the post-motion geometry.

2. **Relay selection stage**
   Based on post-motion geometry and current queues/load, the system selects the satellite pair for each UAV.

3. **Access allocation stage**
   Based on post-motion geometry and selected satellite relay context, the system allocates `bw`.

4. **Physical execution stage**
   Access transmission, backhaul transmission, queue evolution, and reward computation are executed.

This cycle changes the current environment if necessary.
That is intentional.

### 4.2 Why the order is `accel -> sat_pair -> bw`

This ordering is chosen from system semantics, not from legacy code.

- `accel` must come first because it changes geometry and feasibility.
- `sat_pair` should precede `bw` because access allocation should be aware of the selected relay/backhaul context.
- `bw` is then allocated under the actual post-motion, post-relay decision context.

This order intentionally makes access-side allocation **relay-aware**, which is more aligned with the end-to-end control objective than the legacy step order.


## 5. Joint Action Factorization

The redesigned actor uses:

```text
pi(a | x)
= pi_accel(a_accel | z_accel)
 * pi_satpair(a_satpair | z_sat)
 * pi_bw(a_bw | z_bw)
```

where:

- `z_accel` is the pre-motion structured state,
- `z_sat` is the post-motion structured state,
- `z_bw` is the post-motion + post-relay structured state.

This is not a patch over the existing parallel-head policy.
It is a new structured policy class.


## 6. Stage State Definitions

The redesign uses three structured stage states per environment step.

### 6.1 `z_accel`: pre-motion structured state

This state contains the current physical and queue state before any decision in the slot:

- UAV node features,
- GU node features,
- SAT node features,
- UAV-GU edge features,
- UAV-SAT edge features,
- UAV-UAV edge features,
- global scalar context if needed.

### 6.2 `z_sat`: post-motion structured state

This state is built after applying the chosen `accel` through a deterministic stage transition.

It must recompute all geometry-sensitive quantities affected by motion:

- updated UAV positions and velocities,
- updated UAV-GU relative geometry,
- updated access candidate relations,
- updated UAV-SAT relative geometry,
- updated satellite visibility / validity,
- updated UAV-UAV safety relations.

This stage should not merely append `accel` as a token.
It should **rebuild the relevant relation features under the moved geometry**.

### 6.3 `z_bw`: post-motion + post-relay structured state

This state is built after selecting the satellite pair.

It must include the relay decision as a structural change rather than as a loose token:

- selected satellite pair per UAV,
- projected relay capacity / contention summaries,
- updated UAV-SAT edge state for selected links,
- updated per-UAV backhaul context derived from the chosen pair,
- post-motion access-side relation features.

This state is what the `bw` policy should condition on.


## 7. Actor Design

### 7.1 `AccelPolicy`

`AccelPolicy` operates on `z_accel`.

Its job is to determine motion commands that reshape geometry and load distribution.
It should use a UAV-centric relational encoder over:

- the UAV itself,
- nearby UAVs,
- relevant GU clusters / candidate users,
- visible satellites.

Output:

- continuous `accel` action per UAV.

### 7.2 `SatPairPolicy`

`SatPairPolicy` operates on `z_sat`.

It should **not** select satellites sequentially.
The actual action semantics is an **unordered pair**.

The redesigned policy should therefore score legal unordered satellite pairs directly:

```text
pi_satpair(pair | z_sat)
```

This is justified because:

- the semantics is pair selection rather than ordered selection,
- the maximum candidate set is currently small (`sats_obs_max = 6`),
- the number of legal pairs is therefore at most `C(6,2) = 15`,
- exact pair scoring is feasible and avoids introducing artificial order bias.

#### Pair representation

For each UAV and each legal pair, build a pair token from:

- per-satellite embeddings for both members,
- symmetric pair features such as sum/min/max/absolute difference,
- projected relay capacity features,
- queue/load features,
- diversity or redundancy features if relevant.

The pair scorer should be permutation-symmetric with respect to the two satellites.

### 7.3 `BwPolicy`

`BwPolicy` operates on `z_bw`.

It allocates access-side bandwidth over the valid-user simplex conditioned on:

- post-motion geometry,
- selected relay context,
- current user queues and access link quality,
- current valid candidate set.

#### Chosen distribution family

The proposed `bw` distribution family is:

> **masked logistic-normal on the valid-user simplex**

More concretely:

- a Gaussian is defined in log-ratio coordinates,
- a masked simplex transform maps it back to allocation fractions over the valid users,
- invalid users receive zero allocation.

#### Why masked logistic-normal

This choice is made from problem requirements and literature, not from legacy code convenience.

It is preferred because it:

1. avoids direct concentration-dominated geometry of the current `Dirichlet` path,
2. is a standard distribution family for compositional/simplex data,
3. preserves permutation symmetry across users,
4. avoids the artificial ordering bias of stick-breaking,
5. supports dynamic valid sets more naturally than a fixed-order factorization.

Relevant literature:

- Aitchison & Shen, *Logistic-Normal Distributions* (Biometrika, 1980)
- Aitchison, *The Statistical Analysis of Compositional Data* (JRSS B, 1985)

#### Why not stick-breaking

Stick-breaking is rejected here because the user set is inherently unordered.
A stick-breaking parameterization would impose a sequence on users that does not exist in the task semantics.


## 8. Centralized Critic Design

### 8.1 The critic input must be a superset of the actor-relevant information

The critic should not be fed the current flattened pseudo-global state from `env.get_global_state()`.
It should also not be fed a smaller information set than the actor.

The critic input must be rebuilt from simulator state and relation structure so that it includes:

- all actor-relevant local relational information,
- hidden global information unavailable to decentralized actors,
- stage-specific action prefix effects.

### 8.2 Critic input structure

The critic input is a typed relational world state with:

#### UAV nodes

- position
- velocity
- energy
- UAV queue
- association statistics
- stage prefix action fields when relevant

#### GU nodes

- position
- GU queue
- demand / arrival information if modeled
- previous association or continuity indicators

#### SAT nodes

- orbital / relative motion state
- SAT queue
- current load
- service capacity context

#### UAV-GU edges

- relative geometry
- access channel quality / spectral efficiency
- candidate / valid flags
- continuity and association-related fields

#### UAV-SAT edges

- relative position / velocity
- Doppler-related quantities
- relay quality / projected bandwidth
- queue / load / validity fields
- selected-pair indicators when relevant

#### UAV-UAV edges

- relative geometry
- relative velocity
- safety / collision / danger relation fields

### 8.3 Prefix actions should be written back into the structured state

The critic should not merely consume abstract action-prefix tokens.

Instead:

- `accel` should induce a deterministic post-motion state transition,
- `sat_pair` should induce a deterministic relay-context transition,
- the resulting structured state should be re-encoded at the next stage.

This prevents the critic from trying to infer changed geometry or changed relay context from a token alone.

### 8.4 Critic architecture

The critic should be a **relational centralized critic**, not a pooled global vector critic.

Recommended structure:

- typed node encoders,
- typed edge encoders,
- cross-attention or graph-style message passing,
- late readout for scalar value heads.

Simple early set pooling is insufficient because `bw` and `sat` require element-level and edge-level information throughout most of the computation.

Relevant literature:

- Deep Sets (permutation invariance)
- Set Transformer (attention-based set processing)
- relational inductive bias / graph network literature

### 8.5 Stage-specific value heads

The critic should output:

- `V_accel(z_accel)`
- `V_sat(z_sat)`
- `V_bw(z_bw)`

These are all value functions, not Q-functions.
The redesign still remains in the actor-critic / PPO family.


## 9. Expanded-MDP Training Formulation

### 9.1 One physical step becomes three internal decision transitions

Each physical environment step is expanded into:

1. `z_accel --a_accel--> z_sat`
2. `z_sat --a_satpair--> z_bw`
3. `z_bw --a_bw--> z_accel(next)`

### 9.2 Reward assignment

The default reward assignment is:

- `r_accel = 0`
- `r_sat = 0`
- `r_bw = r_env`

This is not an arbitrary heuristic.
It is the natural choice when the internal stages are only a decomposition of one physical control cycle and the task objective is defined at the physical step level.

Using zero intermediate rewards preserves the original task objective exactly while exposing the decision structure to PPO.

This follows the modified-MDP logic used in structured multidimensional action formulations.

### 9.3 Discounting

Use:

- `gamma_stage = 1` for intra-step stage transitions,
- `gamma_env = gamma` for the final transition into the next physical step.

Then GAE is run over the expanded trajectory.


## 10. PPO / GAE Training

### 10.1 Actor losses

The actor objective is the sum of three PPO losses:

- `L_accel`
- `L_satpair`
- `L_bw`

Each stage uses:

- its own old log-probability,
- its own current log-probability,
- its own stage-specific advantage,
- its own entropy regularization if desired.

### 10.2 Critic losses

The critic objective is the sum of stage-specific value losses:

- `MSE(V_accel, target_accel)`
- `MSE(V_sat, target_sat)`
- `MSE(V_bw, target_bw)`

### 10.3 Why this addresses the credit problem

The core problem to solve is:

> a locally bad action at one head can be positively reinforced because other heads made the joint return large.

The stage-conditioned values reduce this mixing because:

- `sat_pair` is evaluated from `z_sat`, where motion effects are already absorbed into the baseline,
- `bw` is evaluated from `z_bw`, where both motion and relay choice are already absorbed into the baseline.

This does not make cross-stage credit perfect by magic, but it removes a large part of the current cross-head confusion while staying within a PPO-style value-based advantage framework.


## 11. Environment Redesign Requirements

The redesign requires environment support beyond the current `step()` interface.

### 11.1 The environment must expose stage-transition builders

Required deterministic helpers:

- build `z_accel` from the simulator state,
- apply `accel` to obtain the post-motion state for `z_sat`,
- apply `sat_pair` to obtain the relay-context state for `z_bw`,
- execute the final physical step after `bw`.

### 11.2 The environment must expose structured world state, not only flattened pseudo-global state

The current `get_global_state()` should not be the critic input contract.
The redesign needs a structured export of:

- node states,
- edge states,
- masks,
- stage-dependent relation features.

### 11.3 The candidate and valid-set logic should be recomputed at the relevant stage

A redesigned stage pipeline should recompute geometry-sensitive and validity-sensitive quantities at the stage where they matter.
They should not remain cached from a semantically earlier point if that earlier point is no longer the correct decision state.


## 12. Module-Level Rewrite Plan

### 12.1 Replace

- current `HybridActionDist` main path
- current `MaskedDirichlet` main training path
- current pseudo-global critic input path
- current single-step rollout representation

### 12.2 Introduce

- `StructuredActor`
  - `AccelPolicy`
  - `SatPairPolicy`
  - `BwPolicy`

- `StructuredCritic`
  - relational encoder
  - `V_accel`, `V_sat`, `V_bw` heads

- `StructuredRolloutBuffer`
  - stage id
  - structured stage observation / graph inputs
  - action
  - old logprob
  - value
  - reward
  - done / next-stage metadata

- stage-transition helpers in the environment


## 13. What Should Not Be Reused as the New Core Design

The following should not be treated as the target redesign:

- legacy per-head value baselines under shared global reward,
- legacy headwise PPO surrogate path,
- auxiliary local-loss patches on top of the current parallel-head actor,
- sparse external counterfactual credit sampling attached to the old actor,
- simple early mean/max pooled set summaries as the critic core.


## 14. Validation Criteria

Success should not be defined by reward alone.
The redesign should be judged first by whether it fixes the diagnosed mechanisms.

### 14.1 `sat` mechanism checks

- selected satellite-pair ranking should become positively aligned with one-step and multi-step relay quality,
- locally bad relay choices should no longer systematically receive positive stage advantage,
- backhaul throughput / UAV queue explosion at the previous `u0100` failure region should be reduced.

### 14.2 `bw` mechanism checks

- `|log_ratio_bw|` should no longer scale pathologically with `valid_count`,
- training statistics should no longer be dominated by concentration-like geometry artifacts,
- execution quality should improve without relying on heuristic `bw`.

### 14.3 system-level checks

- learned `accel` should no longer push the system into the same bad regime as easily,
- cross-head / cross-stage interference should be reduced,
- the learned policy should outperform the legacy factorized architecture under the same evaluation protocol.


## 15. Final Position

The redesign proposed here is not:

- a small patch to the current MAPPO implementation,
- a reuse of previously failed per-head critic switches,
- a heuristic imitation-based fix,
- or a design constrained by the current `env.step` order.

It is a full restructuring based on the current diagnosis:

> the task is a structured, strongly coupled joint-action control problem,
> but the current system models it as parallel factorized heads with a coarse shared advantage,
> while `bw` additionally carries a real head-internal optimization pathology.

The proper response is therefore:

1. redesign the environment control cycle if necessary,
2. redesign the actor as a structured conditional joint policy,
3. redesign the critic as a relational stage-conditioned centralized critic,
4. redesign the rollout and PPO target around the expanded decision process,
5. replace the current `bw` parameterization with a better-matched simplex policy family.


## 16. References

- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*  
  https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf

- Metz et al., *Policy Gradient For Multidimensional Action Spaces*  
  https://openreview.net/forum?id=rk3b2qxCW

- Fan et al., *Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space*  
  https://www.ijcai.org/Proceedings/2019/316

- Aitchison and Shen, *Logistic-Normal Distributions: Some Properties and Uses*  
  https://academic.oup.com/biomet/article/67/2/261/314675

- Aitchison, *The Statistical Analysis of Compositional Data*  
  https://academic.oup.com/jrsssb/article/47/1/136/7028210

- Zaheer et al., *Deep Sets*  
  https://papers.neurips.cc/paper/6931-deep-sets

- Lee et al., *Set Transformer*  
  https://proceedings.mlr.press/v97/lee19d.html

- Battaglia et al., *Relational Inductive Biases, Deep Learning, and Graph Networks*  
  https://arxiv.org/abs/1806.01261
