"""Phase 2 hierarchical control loop: final-goal / active-subgoal
separation, subgoal lifecycle + stats, replanning triggers, failure
recovery, and the coordinator tying them together (plan section 6). Pure
Python -- no ROS, no Global RL dependency; a fixed or externally-supplied
subgoal sequence drives the loop until Phase 4 adds a learned subgoal
source."""
