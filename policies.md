# Goal classifier policy

## Purpose
Detect goals in 5-a-side football clips and identify which team scored.

## Teams
- Dark sportswear: athletic tracksuits / sportswear
- Dark suits: jackets, dress shirts, office-style suits

## Decision rules
1. Goal = ball fully crosses the goal line into the net.
2. Saves, blocks, post/bar hits, shots wide = NOT goals.
3. If goal is false, team must be null.
4. If goal is true, team must be exactly one of the two team names.

## Priority
1. Correct goal vs not-goal classification
2. Correct team when goal is true
