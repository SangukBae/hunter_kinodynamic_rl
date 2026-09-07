# TRACTOR-TQC Model Specification

Status: **PARTIAL IMPLEMENTATION / formal contract blocked / untrained / performance unmeasured**
Family: `tractor_tqc`
Revision: `tractor-tqc-r1`
Core variant: `tractor_core_v1` (`Q_m=0`, `Q_r=1`)
Action contract: normalized 3-vector decoded as `[kappa,v_ref,L]`

Canonical name: **Trajectory-Risk via Action-Conditioned Tube-Occupancy Reasoning with Truncated
Quantile Critics (TRACTOR-TQC)**.

Executable source is under `hunter_kinodynamic_rl/rl/networks/tractor/`; A7/A8/A9 development
configurations are under `config/tractor/model*.yaml`. Unit/property tests establish tensor,
causality, mask and gradient behavior only. Sections explicitly labelled **target** specify remaining
research implementation and may not be cited as landed code. Current gaps are exported by
`formal_research_implementation_readiness()` and block every formal campaign evidence stage. No
trained checkpoint or comparative result exists as of 2026-09-07.

Current implementation includes the belief/forecast/tube/product/value/risk forward path, candidate
selector, Bellman/risk-head/actor development updates and config fingerprint. It does not yet include
the target rollout-module/source-hash parity, residual validity/supervision, Stage 3/4 objectives,
Stage-5 atomic RNG transaction, complete semantic checkpoint or full hypothesis evaluator.

## 1. Scope and contribution

TRACTOR-TQC replaces the current flat Local representation and scalar risk critic. It does not
replace localization, Global planning, Pure-Pursuit, the fixed ActionGuard, or the physical action
contract.

The research contribution to test is the **Causal Tube-Occupancy Interaction Operator**:

1. build a factorized, ego-warped scene belief;
2. roll each candidate into an Ackermann-feasible probabilistic swept tube;
3. predict one action-independent scene future;
4. gather an isolated sparse feature copy along each tube;
5. preserve explicit tube×occupancy products by time and cause;
6. estimate return quantiles and cause-time adverse-event hazards.

Recurrence, attention, residual dynamics, ensembles and calibration are supporting techniques, not
independent novelty claims.

## 2. Coordinate and default shape contract

| Item | Proposed default | Contract |
|---|---:|---|
| LiDAR history | `T_obs=4`, `N_scan=80` | current frame first, front `pi` rad |
| BEV | `64×64`, `0.25 m/cell` | `[-8,8) m` in `x,y`; `+x` forward, `+y` left |
| candidates | `K=8` | ordering must be permutation-equivariant |
| future | `H=15`, `dt_out=0.2 s` | index `q=0..14` means time `(q+1)dt_out`, maximum 3.0 s |
| sparse samples | `M=32` | per candidate and future step |
| return critic | `2×25` quantiles | same TQC count as current baseline |
| causes | `C=3` | static, dynamic, boundary-or-unknown |

Tensor axes are `[batch,...,row_y,col_x]`. Grid indices use half-open bounds and
`floor((coord+8)/0.25)`; `(0,0)` maps to `(row=32,col=32)`. Out-of-bounds samples have no valid grid
index but retain their probability as explicit boundary mass; they are never dropped, renormalized or
converted to synthetic free space. These rules are fingerprinted.

## 3. Inputs

The legacy vector is split without reordering:

```text
X_t = observation[:320].reshape(B,4,80)
e_t = observation[320:328]

e_t order:
  goal_distance, relative_goal_heading,
  previous_normalized_kappa, previous_normalized_v_ref, previous_normalized_L,
  measured_speed, measured_yaw_rate, measured_centre_steering
```

Required sequence/runtime metadata:

| Tensor | Shape | Meaning |
|---|---:|---|
| `scan_valid` | `(B,4,80)` bool | max-range return와 invalid fill을 구분 |
| `motion_delta` | `(B,3,4)` | older→adjacent-newer `[dx,dy,dyaw,dt]` |
| `motion_valid` | `(B,3)` bool | invalid edge 뒤의 older warp chain 전체 mask |
| `previous_command_published` | `(B,2)` | confirmed `[speed,centre_steering]` |
| intent/command/response validity | explicit masks | numeric zero와 valid zero를 구분 |
| localization confidence/covariance | scalar, `(B,3,3)` + masks | current pose uncertainty |
| sensor freshness | `(B,1)` + mask | seconds since coherent snapshot |
| `scene_reset`, `response_reset` | two `(B,1)` bool masks | episode reset sets both; stream faults may reset one |
| `previous_scene_hidden`, validity | `(B,32,64,64)`, `(B,1)` bool | caller-owned ConvGRU state |
| `previous_response_hidden`, validity | `(B,32)`, `(B,1)` bool | caller-owned vehicle-response GRU state |

`motion_delta[:,0]` maps `t-1→t`; older transforms are cumulatively composed. Missing pose is not
zero motion. Repeated/non-monotonic timestamps invalidate the sample rather than becoming 0.1 s.

The encoder also emits an immutable `DecisionContext`: metric goal vector, current measured vehicle
response, localization covariance and decision timestamp. Rollout/progress code uses this typed
context, never decodes physical truth from a latent.

## 4. Belief model

Each scan frame is differentiably lifted to hit/free/observed/age channels, ego-warped into the
decision-time frame and concatenated to `(B,16,64,64)`. A 32-channel convolutional backbone and
ConvGRU produce:

| Output | Shape | Semantics |
|---|---:|---|
| `F_t` | `(B,32,64,64)` | shared scene feature |
| `O_t^(F,S,D,U)` | four `(B,1,64,64)` maps | mutually exclusive softmax classes |
| `V_t^D` | `(B,2,64,64)` | ground-relative dynamic-object velocity in `E_t^ego` |
| `z_t^E` | `(B,64)` | goal, ego motion, localization and health |
| `z_t^P` | `(B,32)` | previous intent/command to current response state |

The observed mask remains evidence validity; it is not a fifth occupancy class. Static world points
must have zero object flow under pure ego translation/rotation. Scene usability requires a current
valid/fresh scan and localization; response usability requires valid prior intent, confirmed published
command, current response and positive `dt`. Each state applies this same ordered truth table
independently:

| reset | prior valid | current usable | next state / validity |
|---:|---:|---:|---|
| true | either | true | update once from exact zero / true |
| true | either | false | exact zero / false |
| false | true | true | update once from prior / true |
| false | true | false | hold prior byte-for-byte / true |
| false | false | true | update once from exact zero / true |
| false | false | false | exact zero / false |

Reset is applied before usability/update. Invalid data cannot advance a state, and scene validity
cannot substitute for response validity or vice versa.

The scene predictor produces one action-independent future:

```text
O_base : (B,H,4,64,64)
V_base : (B,H,2,64,64)
F_base : (B,H,32,64,64)
```

Candidates cannot mutate current posterior or dense base probabilities. Reactive-agent branching is
a separate experiment family, not an optional core behavior.

## 5. Normalized action and stop contract

Schema ID: `trajectory_kappa_vref_L_v1`. The authoritative current implementation is
`hunter_kinodynamic_rl/trajectory/action_space.py`; TRACTOR stores its source hash plus every resolved
bound in the architecture fingerprint.

For finite `a=(a_0,a_1,a_2)`, first clip each coordinate to `[-1,1]`, then use
`lerp(a;l,u)=0.5(a+1)(u-l)+l`:

```text
kappa_max = tan(steering_limit_rad) / wheelbase_m * kappa_scale
kappa      = lerp(a_0; -kappa_max, +kappa_max)     [m^-1]
v_ref      = lerp(a_1; v_min_mps, v_max_mps)       [m/s]
L          = lerp(a_2; L_min_m, L_max_m)           [m]
```

`v_max_mps` is the action override when non-null, otherwise the robot maximum. The inverse is
`a=clip(2(x-l)/(u-l)-1,-1,1)`. Non-finite input, wrong shape, nonpositive wheelbase, invalid ordered
bounds or a resolved limit outside the robot attestation rejects the candidate before decode. Clipping
finite actor samples is part of the current parity contract; external overrides also record whether
clipping occurred.

The initial improved-profile values are `kappa_scale=1`, `v_ref∈[0,1.3333333333] m/s` and
`L∈[0.5,3.0] m`, with `wheelbase=0.550 m` and centre-steering limit
`18.8883201008 deg`; `kappa_max` remains derived, never copied as an independent constant. These are
simulation-profile values until real attestation.

Candidate index 0 is the base actor action. In **physical** coordinates, the stop construction is
`[base.kappa,0,base.L]`; its normalized form is the canonical inverse decode (`a_v=-1` when
`v_min=0`). If the base is already the same physical stop after tolerance-based dedup, index 0 receives
the sole `candidate_is_stop=true` mark and no duplicate index 1 is appended. Otherwise the marked stop
is reserved at index 1. Stop identity follows every later permutation and may never be inferred merely
from near-zero speed. It is a braking intent, not an assertion that current speed is instantly zero.
The selector/fallback distinguishes it from a guard-generated emergency stop.

## 6. Candidate, nominal dynamics and tube

The minimum-paper actor remains a single tanh-Gaussian. Inference forms a deterministic structured
set containing the squashed mean, curvature/speed/length perturbations and one validated stop, then
deduplicates it while preserving masks and stop identity. External candidate override requires
content, source, action-contract and decoder hashes.

### Current differentiable rollout

The current `NominalRolloutAdapter` decodes the same physical action bounds and now fail-fast matches
the collection profile's wheelbase, steering/speed bounds, accel/brake, steering rate, lag, horizon,
footprint and LiDAR range. It uses a fixed global
`substeps=ceil(dt_out/dt_dyn)` and `dt=dt_out/substeps`, samples after every output bin, propagates an
analytic pose Jacobian with diagonal configured process noise, and uses architecture-revision plus
resolved dataclass config fingerprinting. It is a differentiable reimplementation and does not import
the package's existing non-neural dynamics modules.

### Target rollout parity — not implemented

Every normalized candidate is canonically decoded to `[kappa,v_ref,L]`. The target normative rollout reuses
`dynamics/ackermann_rollout.py`, `dynamics/actuator_model.py` and `dynamics/bicycle_model.py` after
parity tests. With current defaults, dynamics integrates at nominal `dt_dyn=0.1 s`; TRACTOR risk
outputs are sampled at `t_h=0.2h` up to `H=15`. Integration always uses substeps no larger than the
fingerprinted `dt_dyn`; output sampling/interpolation may not change the actuator update order.

For each candidate:

```text
delta_target = atan(wheelbase * kappa)
h_commit = horizon_max                              if v_ref <= 0.05
           clip(L/v_ref, horizon_min, horizon_max) otherwise
h_score_raw  = max(h_commit, min_safety_horizon)
h_score_grid = min(H*dt_out, ceil(h_score_raw/dt_out)*dt_out)
n             = max(1, ceil(h_score_grid/dt_dyn));  dt_dyn_step = h_score_grid/n
```

`ceil` makes every `dt_dyn_step` no larger than `dt_dyn`; grid-ceiling prevents an undefined
partial output bin and is conservative in time. This is a new fingerprinted TRACTOR
trajectory revision and must also be used by its same-contract baselines; it is not exact-resume
compatible with the current round-based rollout. `h_commit` controls the L-dependent steering blend;
the L-independent safety floor applies only to
risk horizon so small `L` cannot hide a later collision. At each substep
`b=clip(dt_dyn_step/max(h_commit,dt_dyn_step),0,1)` and the blend requests
`delta_req=delta+b(delta_target-delta)`. With actuator modeling enabled, it alone applies steering
rate limiting—never both blend and actuator—and updates

```text
v_rate    = move_towards(v, v_ref, (accel if |v_ref|>=|v| else brake)*dt_dyn_step)
delta'    = move_towards(delta, delta_req, steering_rate*dt_dyn_step)
alpha_lag = 1-exp(-dt_dyn_step/speed_lag_tau)       # when tau>0
v_eff'    = v_lagged + alpha_lag*(v_rate-v_lagged)  # otherwise v_rate
```

The bicycle step uses midpoint response
`v_bar=(v_eff+v_eff')/2`, `delta_bar=(delta+delta')/2`,
`omega=v_bar*tan(delta_bar)/wheelbase`, then
`x'=x+v_bar*cos(yaw+omega*dt_dyn_step/2)dt_dyn_step`,
`y'=y+v_bar*sin(yaw+omega*dt_dyn_step/2)dt_dyn_step`,
`yaw'=yaw+omega*dt_dyn_step`.
When actuator modeling is disabled, speed applies directly and the blend owns the single steering-rate
clip. A stop candidate therefore decelerates according to the brake/lag state.

The nominal `P_nominal:(B,K,H,3)` pose tensor samples/interpolates the integrated state at exact
`t_q=(q+1)dt_out` and uses `candidate_horizon_mask[q] iff t_q<=h_score_grid`; padded rows cannot enter
risk/value aggregation. When `dt_out/dt` is not integral, interpolation occurs between adjacent
post-actuator states without another actuator update. Proposed covariance follows
`Sigma' = A Sigma A^T + G Q G^T` with analytic Jacobians, PSD process-noise contract and no silent
diagonal fill. A residual member may change only bounded vehicle response mean/process noise; it may
not learn obstacle motion, sensor error or localization drift. Localization covariance stays separate
until rasterization, where it is composed exactly once.

The target fingerprint must include the source hashes and resolved wheelbase, steering/curvature/speed limits,
accel, brake, steering rate, lag tau, `dt_dyn/dt_out`, output indexing/interpolation, horizon
bounds/safety-floor/grid-ceiling,
commit-blend flag/order, process-noise/residual contract and footprint.

### Target residual step for A8/A9 — forward path partial, supervision/validity open

Residual contract `vehicle_response_residual_step_v1` runs **once per `dt_dyn_step`**, after the
nominal actuator update and before midpoint bicycle integration. Each member receives a 46D vector:

```text
z_P                                                     32
scaled physical candidate [kappa,v_ref,L]                3
current measured/member response [v,yaw_rate,steering]   3
response-valid mask                                      3
nominal next response [v,yaw_rate,steering]               3
dt/dt_ref, elapsed/h_score_grid                           2
                                                        --
                                                         46
```

A `46→128→128→4` MLP outputs raw `delta_v`, `delta_steering`, `log_q_v`, `log_q_steering`.
`delta_v=b_v*tanh(raw_v)` and `delta_steering=b_delta*tanh(raw_delta)` are added to nominal endpoint
response then clipped to attested speed/steering bounds; yaw rate is recomputed as
`v*tan(steering)/wheelbase`, never independently corrected. Process variances use
`q=q_min+(q_max-q_min)*sigmoid(raw_q)` and enter the same analytic covariance recursion. All bounds
and scales are fingerprinted.

The corrected endpoint response becomes that member's next substep response/actuator state, so errors
accumulate causally; no hidden residual recurrence or output-bin-only correction is allowed. A8 has one
member. A9 members have independent parameters and bootstrap episode inclusion, but identical inputs,
order and bounds. Target `residual_step_valid` requires valid current response, positive `dt`, finite nominal
state and a present candidate; otherwise the residual branch is masked and that member/candidate is
invalid rather than silently nominal.

The unimplemented Stage-3 `L_vehicle_response` target is masked Gaussian NLL/Huber against the next
timestamp-aligned measured `[v,steering]`, with yaw rate used as a derived consistency check. Training
uses consecutive sequence substeps and actual `dt`; multi-step teacher forcing policy is fingerprinted.
Output-bin states are sampled only after all residual substeps through that time. Checkpoints contain
exact `residual_head_<m>` keys plus bounds, member order, bootstrap seeds and loss contract.

The rasterizer composes vehicle pose covariance with localization covariance exactly once, then
creates sparse footprint samples:

```text
P^(m)       (B,Q_m_eff,K,H,3)
Sigma^(m)   (B,Q_m_eff,K,H,3,3)
tube_row^(m), tube_col^(m)  int16(B,Q_m_eff,K,H,M)
tube_xy^(m)                 float32(B,Q_m_eff,K,H,M,2), metres in E_t^ego
W^(m)                       float32(B,Q_m_eff,K,H,M), original probability weights
tube_sample_valid^(m)       bool(B,Q_m_eff,K,H,M), valid in-grid lookup
```

`Q_m_eff=1` for the nominal-only core. Footprint geometry must use the single robot attestation also
used by collision checking, map inflation and Global masks.

For every member/candidate/time, nonnegative `W^(m)` plus OOB mass sum to one. Row/column are `-1` for
invalid/OOB slots, their validity is false and their weight is accumulated into `w_oob^(m)`. The
rasterizer returns valid in-grid samples with their original weights and
`w_oob^(m)=1-sum(valid in-grid weights)`. It must not renormalize the in-grid subset. Invalid
localization covariance makes tube coverage invalid and the candidate infeasible; geometric
out-of-bounds with otherwise valid covariance remains valid boundary-risk evidence.

## 7. Causal Tube-Occupancy Interaction

For member `m`, candidate `i`, time `h` and valid in-grid cell `p`, the required explicit views are

```text
I_static^(m)[i,h]  = sum_p W^(m)[i,h,p] * O_base[h,S,p]
I_dynamic^(m)[i,h] = sum_p W^(m)[i,h,p] * O_base[h,D,p]
I_BU^(m)[i,h]      = sum_p W^(m)[i,h,p] * O_base[h,U,p] + w_oob^(m)[i,h]
```

Thus boundary-or-unknown mass is conserved in `[0,1]`; it is not a fourth occupancy lookup.
Sensor/localization health remains separate validity/selector context and cannot erase or manufacture
`w_oob^(m)`. The member axis is retained through tube, product, private feature and `J^(m)`; only the
declared uniform value aggregation may reduce it. The operator may query shared scene features with tube
tokens and update only a candidate-private sparse copy. It may not write to `F_base` or another
candidate. Candidate permutation must permute outputs identically, and scoring one candidate alone
must match its score inside a batch within tolerance.

The temporal aggregator returns `J^(m):(B,Q_m_eff,K,H,128)`. The value path uses a uniform mean over
all configured valid members; dropping an invalid member and renormalizing is forbidden.

## 8. Return and risk heads

Two critics output `Z^Q:(B,K,2,25)`. Bellman learning uses stored executed transitions only. The target
is built by control flow, never by multiplying a possibly invalid/NaN next value by zero:

```text
if terminated:
    y = reward                         # do not encode/gather a next path
elif bellman_sample_valid and next_observation_valid:
    y = reward + discount_factor * (Z_target(next_state,next_action)
                                    - alpha_ent * log_pi(next_action|next_state))
else:
    exclude row from Bellman loss      # infrastructure cut or missing next snapshot
```

The current side uses `K_current=1` and scores the exact stored pre-guard
`action_normalized_requested` from that observed transition after validating its action/decoder hash.
It does not regenerate proposals, run the selector or substitute an MC-3 counterfactual. Reward and
next observation come from the command that actually passed through the fixed guard/controller; thus
the critic learns the value of a requested action under that fingerprinted guard environment. Guarded
and published 2D commands remain attribution fields and are never inverted into an artificial
`[kappa,v_ref,L]`.

A time-limit `truncated` row bootstraps only when a coherent final next observation is present; it
does not itself zero the bootstrap. Target action/value gathers are performed only after all next-row,
candidate and target-path validity checks pass. Target quantiles from both critics are sorted and the
upper tail is removed exactly as in the same-contract baseline: concatenate `2×25=50`, sort ascending
by value, drop `drop_per_net×N_Q=2×2=4` largest values, and use the remaining 46 targets in quantile
Huber loss. Equal-valued ties are numerically interchangeable; implementation sort/dtype and the
`drop_per_net=2` rule are fingerprinted. Changing the drop count creates a new training variant.
Imagined transitions may train representation, dynamics or risk, but never enter the primary Bellman
target.

The following Bellman next-action transaction is a **target contract, not current implementation**.
Current code removes invalid bootstrap rows and saves a device `torch.Generator` state, but it does not
sort stable replay join keys, provide a device-independent Philox transform or roll back RNG/optimizer
state after a late failure.

Target `target_policy_online_actor_philox_v1`:

1. terminal and invalid-next rows are removed before actor/target encoding and consume no random draw;
2. eligible rows sort by stable `(episode_id,step_index,sample_draw_ordinal)`; the sampler assigns a
   unique immutable ordinal to each batch occurrence, so duplicate replay rows receive distinct draws;
3. the **online actor**, evaluated with no parameter gradient on the complete EMA target-encoded
   belief, emits `mu,log_sigma`; there is no target actor;
4. counter-based `sorted_join_key_philox_v1` draws exactly three standard normals per eligible row;
   algorithm/version, seed, uint64 counter, sort/unsort rule and device-independent transform are
   fingerprinted;
5. `u=mu+sigma*epsilon`, `a'=tanh(u)` and the tanh-Jacobian-corrected `log_pi(a'|s')` come from the
   same pre-tanh sample—no resampling or nearest-candidate gather;
6. target decode→nominal rollout→tube→target interaction/aggregator→two target critics runs with
   `K_target=1`; risk heads, calibrator, selector, structured set and guard are absent;
7. RNG reservation, value+risk optimizer commit and EMA are one transaction. Any non-finite, AMP or
   late failure rolls back RNG and all steps; a successful transaction commits all of them.

Checkpoint/resume restores the exact RNG state/draw count. Interrupted and uninterrupted execution
must reproduce eligible keys, pre-tanh samples, actions, corrected log-probabilities and targets.

Core risk uses discrete competing hazards
`lambda[i,h,c] = P(event at h,c | survived before h)`. Survival is

```text
S_after[i,q] = product_(j=0..q) (1 - sum_c lambda[i,j,c]), q=0..H-1
P(event through horizon) = 1 - S_after[i,H-1]

event at (q,c):  NLL = -sum_(j<q) log(1-sum_c lambda[i,j,c]) - log(lambda[i,q,c])
no event through censor q:
                  NLL = -sum_(j<=q) log(1-sum_c lambda[i,j,c])
```

Losses mask censored/invalid labels. R5/R6 variants additionally predict ordered clearance and
stopping-margin quantiles. R6 uses matched bootstrap bundles and reports member dispersion; missing
members invalidate a candidate rather than silently lowering uncertainty.

Risk heads use one discriminated contract; only the selected row exists:

| ID | Output per candidate/member | Target and loss |
|---|---|---|
| R0 | scalar endpoint `(B,K,1)` | pre-guard requested-action `y_legacy` from the current `risk_target` generator/horizon; masked MSE |
| R1 | event-by-horizon `(B,K,1)` | event/no-event; masked BCE |
| R2 | hazard `(B,K,H,1)` | event step+censor horizon; survival NLL |
| R3 | endpoint class `(B,K,C+1)` | no-event/static/dynamic/boundary-or-unknown; masked CE |
| R4 | competing hazards `(B,K,H,C)` | event step/cause/censor; competing-risk NLL |
| R5 | R4 + clearance `(B,K,H,N_clear)` and stopping `(B,K,H,N_stop)` | R4 plus masked quantile losses |
| R6 | R5 with leading member axis `(B,Q_r,...)` | matched member bootstraps and the same losses |

R0 label fields are separate `legacy_risk_target`, `legacy_risk_valid`,
`legacy_risk_generator_hash` and `legacy_risk_horizon_hash` on the pre-guard requested normalized
action associated with that transition; the loader
may not infer them from counterfactual rows. R1–R6 use [DATA_AND_REPLAY.md](DATA_AND_REPLAY.md)'s exact
candidate schema. R0–R5 have `Q_r=1`; only R6 permits `Q_r>=2`. Inactive subheads/labels are absent,
not zero-filled. Each R5/R6 member is one indivisible hazard+clearance+stopping bundle with a common
inclusion bit and member ID/order.

Initial `N_clear=N_stop=7` at ordered quantiles `[.05,.10,.25,.50,.75,.90,.95]`; the two counts and
levels are independently fingerprinted. Deployable A7 and A8 use R5 (`Q_r=1`); A9 uses R6
(`Q_r=3` canonical). R0–R4 are isolation ablations with selection/deployment disabled unless a new
registered promotion contract supplies the missing severity/calibration requirements.

### Ensemble aggregation, calibration and selection

A9 uses the Cartesian contract `vehicle_outer_risk_inner_uniform_v1`: each vehicle member `m` keeps
its own `J^(m)` and every complete risk-member bundle `r` evaluates it, in lexicographic order
`(m=0,r=0..Q_r-1),(m=1,...)`. It creates `S=Q_m_eff*Q_r` matched samples; it does not pair hazard and
severity subheads independently and does not average `Q_m` before risk heads.

For candidate `i`, convert each sample's hazard to cumulative probability `p_s(i)` first. With all
configured samples valid,

```text
p_mean = (1/S) sum_s p_s
p_std  = sqrt(sum_s (p_s-p_mean)^2 / (S-1))          # A9, ddof=1
clearance_mean/stopping_mean and their std use the same fixed sample order
```

Any missing/non-finite member or subhead invalidates that candidate; denominator reduction or member
drops are forbidden. A7/A8 have one sample and dispersion exactly unavailable, not a fabricated zero.
Return value separately uses the fixed uniform mean over `Q_m_eff` vehicle members before its two
critic reductions; `Q_r` never enters return aggregation.

Runtime order is exact: **member prediction → per-member survival/severity → fixed uniform aggregate
→ held-out post-aggregate calibrator → conservative UCB/LCB composition → feasibility → value-based
selector**. For A9,
`p_ucb=clip(calibrate(p_mean)+beta_p*p_std,0,1)`. Clearance and stopping margin are both
higher-is-safer, so their conservative statistic is the registered
`LCB=calibrate(margin_mean)-beta_margin*margin_std`; a separately introduced lower-is-safer cost would
require a UCB contract. A7/A8 use calibrated singleton points with no dispersion term.
Calibration method/context, beta values, sample order, `ddof`, clipping, missing-member rule and every
threshold are fingerprinted. The calibrator never changes neural weights.

## 9. Current public API and state lifecycle

```text
encode(TractorInputs)
  -> BeliefState, DecisionContext
     # next hidden states and HealthState are fields of BeliefState

propose(BeliefState)
  -> CandidateSet

score(BeliefState, DecisionContext, CandidateSet)
  -> ReturnRiskOutput

CandidateSelector(CandidateSet, ReturnRiskOutput,
                  previous_action, previous_action_valid, calibration)
  -> SelectionOutput
```

Aggregation, calibration and selection are separate objects/functions rather than public methods on
`TractorTQC`. Singleton A7/A8 dispersion is returned as unavailable (`NaN` plus an explicit
`dispersion_available=false` mask); the operational singleton UCB/LCB term is omitted rather than
interpreting that missing estimate as measured zero uncertainty.

One accepted coherent sensor snapshot may call `encode` and atomically commit both named hidden
states/validity bits at most once.
Retries, candidate permutations and scoring calls cannot advance recurrence. Training reconstructs
hidden state from burn-in; runtime hidden tensors are never checkpoint state.

## 10. Target optimizer and gradient contract

Current optimizer parameter groups are pairwise disjoint. The current development trainer applies a
Bellman representation+value step, a frozen-feature risk-head step, actor/entropy steps and EMA in
sequence. It does not prevalidate or roll back the value step/RNG when the later risk step fails.
Everything below describes the remaining formal target.

Parameter ownership is pairwise disjoint:

- `optimizer_value_path`: belief, future scene, residual, interaction, aggregator, return critics
- `optimizer_risk_heads`: active risk bundle only
- `optimizer_actor`: actor only
- `optimizer_entropy`: temperature only

Target Stage 5 commits value and risk updates atomically, then actor/temperature on eligible batches. A
complete EMA target exists for the entire value path, not only critic heads. Target update is
`theta_bar←(1-tau)theta_bar+tau*theta`, proposed `tau=0.005`, after each successful joint value
transaction.

Actor optimization freezes value/risk weights while preserving gradients through decoded action,
rollout, tube and score to the actor. Belief features are detached for the actor step. The primary
method must test that actor parameters change, critic parameters do not, and action gradients are
finite/nonzero.

## 11. Current inference and fallback

Current aggregation flattens configured vehicle/risk members, calibrates aggregate event probability,
applies UCB/LCB only when member dispersion is available, filters feasibility and chooses
`argmax(score)`; equal scores therefore resolve to the first candidate index. Margin calibration,
multi-key tie breaking and complete calibration-context fingerprinting below remain target work.
Deployment requires a non-null calibrator
whose context hash matches the bundle; development without it may log raw ranking only and cannot
enable selection. For value, uniformly average valid vehicle-member quantiles first, concatenate the
two 25-quantile critics, sort, and define
`V_tail=mean(lowest n_tail)`, `n_tail=max(1,ceil(alpha_tail*50))`.

For each present candidate define

```text
smooth = ||a_i-a_prev||_2^2 if previous intent valid and no response reset else 0
feasible_i = model_valid and aggregate_valid and calibration_valid
             and p_ucb <= p_max
             and clearance_LCB(q=.05) >= clearance_min
             and stopping_LCB(q=.05) >= 0
score_i = V_tail - lambda_p*p_ucb - lambda_smooth*smooth
```

`alpha_tail`, thresholds and lambdas are bundle fields. A reset/invalid previous intent disables the
smoothness term with an explicit false mask; it does not fabricate a zero previous action. The target
tie contract selects the maximum score, then lower `p_ucb`, greater clearance LCB and lower candidate
index. Current code uses first-index `argmax` and returns the selected normalized action; returning the
physical trajectory and candidate-set hash remains target work.

If no candidate is feasible, return index `-1`, no learned action and a reason code; the fixed guard
publishes the registered conservative stop/slow fallback. A marked stop candidate is selected normally
only when it is itself valid and feasible. Pure-Pursuit produces the requested command and ActionGuard
remains final authority. Risk aggregation, calibration and selection are parameter-free except for the
separate fitted calibration artifact; they do not own neural gradients.

The 10 Hz target is `p99<100 ms` on named hardware with `<1%` deadline miss. These are gates, not
measurements. Any missed deadline, no feasible candidate, invalid localization/sensor state, excessive
disagreement or manifest mismatch invokes a conservative stop/slow fallback and logs attribution.

## 12. Registered model families

| ID | Difference | Purpose |
|---|---|---|
| A7 `tractor_core_v1` | nominal physics, R5, `Q_m=0,Q_r=1` | primary operator claim |
| A8 `tractor_residual_v1` | one bounded vehicle residual, R5, `Q_m=1,Q_r=1` | response-model support |
| A9 `tractor_ensemble_v1` | R6 ensembles, canonical `Q_m=3,Q_r=3` | epistemic/OOD extension |

Changing actor family, dense branched scene futures, Global co-training, imagined Bellman targets or
learned initial recurrent state creates a new fingerprinted family.

## 13. Release-blocking properties

1. shape/dtype/device and normalized↔physical round-trip tests pass;
2. pure ego motion preserves static-scene flow semantics;
3. tube mass, covariance composition and footprint bounds match analytic fixtures;
4. posterior/base tensors are action-invariant and candidate ordering is equivariant;
5. single and batched candidate scores agree;
6. cause hazards are nonnegative, sum below one and survival is monotone;
7. censoring and `terminated`/`truncated` targets match hand calculations;
8. changing MC-3 candidates cannot change current Bellman quantiles/loss for the stored requested action;
9. actor gradient routing and complete target EMA ownership pass;
10. deterministic save/resume reproduces candidate/RNG/update sequences;
11. invalid inputs and latency overruns fail closed.

Experimental acceptance belongs to [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md); data and artifact
identity belong to [DATA_AND_REPLAY.md](DATA_AND_REPLAY.md) and
[CHECKPOINT_AND_COMPATIBILITY.md](CHECKPOINT_AND_COMPATIBILITY.md).
