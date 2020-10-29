#-----------------------------------------------------------------------------
# Solves each 8x8 position with dfs and checks winner.
#-----------------------------------------------------------------------------

boardsize 8 8

play b a1
11 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b a2
12 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b a3
13 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b a4
14 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b a5
15 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b a6
16 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b a7
17 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b a8
18 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b1
21 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b b2
22 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b b3
23 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b4
24 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b5
25 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b6
26 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b7
27 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b b8
28 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b c1
31 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b c2
32 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b c3
33 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b c4
34 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b c5
35 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b c6
36 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b c7
37 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b c8
38 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b d1
41 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b d2
42 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b d3
43 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b d4
44 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b d5
45 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b d6
46 dfs-solve-state white
#? [black]

undo
dfs-clear-tt
play b d7
47 dfs-solve-state white
#? [white]

undo
dfs-clear-tt
play b d8
48 dfs-solve-state white
#? [white]
