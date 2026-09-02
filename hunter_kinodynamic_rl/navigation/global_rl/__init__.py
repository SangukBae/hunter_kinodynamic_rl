"""Phase 4 Global RL MVP (plan section 8): discrete candidate-subgoal
sampling, action masking, map/scalar/candidate observation assembly,
option-level reward, a masked Dueling Double DQN, and its SMDP-target
replay buffer.

Pure numpy modules (:mod:`subgoal_sampler`, :mod:`action_mask`,
:mod:`observation`, :mod:`reward`, :mod:`replay_schema`, :mod:`replay`) are
importable without torch. :mod:`networks` and :mod:`agent` require torch
(mirrors :mod:`hunter_kinodynamic_rl.rl.networks.tqc`'s own gating
convention) -- this ``__init__.py`` deliberately imports nothing itself so
that importing this package never requires torch to be installed.

Opt-in (``config.schema.GlobalRLConfig.enabled``, default False) -- nothing
under this package is imported by the existing local-only or Phase 1-3
navigation stack.
"""
