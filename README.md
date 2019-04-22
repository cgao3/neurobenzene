# neurobenzene: An improved benzene project for playing and solving Hex with the help of deep neural networks

This reposotory is built upon [benzene-vanilla-cmake](https://github.com/cgao3/benzene-vanilla-cmake). 

### Prerequisites 

Dependent libraries from vanilla benzene: 

boost, berkeley-db. 

On Ubuntu, install via 
```
sudo apt-get install libboost-all-dev 
sudo apt-get install libdb-dev 
``` 

On Mac, use `brew` instead.

As November 25th 2018, this repository supports C API from [TensorFlow 1.12](https://www.tensorflow.org/install/lang_c), 
therefore it is unnecessary to compile `TensorFlow` from source anymore! 

But, you need to setup C API first, by the following instructions: 

- GPU 
    Only supported on Linux. Sepcifically, pre-released `Tensorflow C API 1.12` requires `CUDA-9.0` and `CUDNN 7.1.4`, make sure you have them in your system. 
    ```sh
    cd tensorflow_c_gpu/
    source setup_tf.sh
    ```

- CPU (Mac or Linux)
    ```sh
    cd tensorflow_c_cpu/
    source setup_tf.sh
    ```

At `tensorflow_c_cpu` or `tensorflow_c_gpu`, there is test source code `hello_tf.c`, you may try 
    ```sh
    gcc hello_tf.c -o test_hello -ltensorflow -ltensorflow_framework
    ./test_hello
    ```
To verify that your `TensorFlow C API` have been correctly set up. If so, it should display 
```
Hello from TensorFlow C library version 1.12.0
```

### How to build? 

After everything is ready, build the project by: 
```sh
$mkdir build
$cd build
$cmake ../
$make 
```

On Mac, there might be `linking error` to `db` or `boost`, in such case, you may revise your `~/.bash_profile` (Mac OS), e.g., 
```
# berkeley-db
export CPATH=$CPATH:/usr/local/Cellar/berkeley-db/6.2.23/include
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/Cellar/berkeley-db/6.2.23/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Cellar/berkeley-db/6.2.23/lib

#boost
export CPATH=$CPATH:/usr/local/Cellar/boost/1.64.0_1/include
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/usr/local/Cellar/boost/1.64.0_1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Cellar/boost/1.64.0_1/lib
```

### Running some tests 

Try ``./build/src/mohex/mohex3HNN``

```
 showboard
= 
 
  2a78b48864b73c1f
  a  b  c  d  e  f  g  h  i  j  k  
 1\.  .  .  .  .  .  .  .  .  .  .\1
  2\.  .  .  .  .  .  .  .  .  .  .\2
   3\.  .  .  .  .  .  .  .  .  .  .\3
    4\.  .  .  .  .  .  .  .  .  .  .\4
     5\.  .  .  .  .  .  .  .  .  .  .\5
      6\.  .  .  .  .  .  .  .  .  .  .\6
       7\.  .  .  .  .  .  .  .  .  .  .\7
        8\.  .  .  .  .  .  .  .  .  .  .\8
         9\.  .  .  .  .  .  .  .  .  .  .\9
         10\.  .  .  .  .  .  .  .  .  .  .\10
          11\.  .  .  .  .  .  .  .  .  .  .\11
              a  b  c  d  e  f  g  h  i  j  k  

nn_ls
= Loaded neural net: /home/chenyang/cg/neurobenzene/share/nn/11x11train4.txt.out-post.ckpt-1.const.pb
Use nn_load nn_model for new model, note nn_model should be a full path or just model name if it is available in share/nn/
List available nn models at share/nn/
13x13train30.txt.out-post.ckpt-2.const.pb
11x11train4.txt.out-post.ckpt-1.const.pb

play b g3
= 

showboard
= 
 
  9af481a52c1456bb
  a  b  c  d  e  f  g  h  i  j  k  
 1\.  .  .  .  .  .  .  .  .  .  .\1
  2\.  .  .  .  .  .  .  .  .  .  .\2
   3\.  .  .  .  .  .  B  .  .  .  .\3
    4\.  .  .  .  .  .  .  .  .  .  .\4
     5\.  .  .  .  .  .  .  .  .  .  .\5
      6\.  .  .  .  .  .  .  .  .  .  .\6
       7\.  .  .  .  .  .  .  .  .  .  .\7
        8\.  .  .  .  .  .  .  .  .  .  .\8
         9\.  .  .  .  .  .  .  .  .  .  .\9
         10\.  .  .  .  .  .  .  .  .  .  .\10
          11\.  .  .  .  .  .  .  .  .  .  .\11
              a  b  c  d  e  f  g  h  i  j  k  

genmove w
63079 info: Best move cannot be determined, must search state.
63079 info: --- MoHexPlayer::Search() ---
63079 info: Color: white
63079 info: MaxGames: 99999999
63079 info: NumberThreads: 1
63079 info: MaxNodes: 10416666 (999999936 bytes)
63079 info: TimeForMove: 10
63079 info: Time for PreSearch: 0.387516s
63079 info: 
  9af481a52c1456bb
  a  b  c  d  e  f  g  h  i  j  k  
 1\.  .  .  *  *  *  *  *  *  *  .\1
  2\.  .  *  *  *  *  *  *  *  *  .\2
   3\.  *  *  *  *  *  B  *  *  *  .\3
    4\.  *  *  *  *  *  *  *  *  *  .\4
     5\.  *  *  *  *  *  *  *  .  *  .\5
      6\.  *  *  *  *  *  *  *  *  *  .\6
       7\.  *  .  *  *  *  *  *  *  *  .\7
        8\.  *  *  *  *  *  *  *  *  *  .\8
         9\.  *  *  *  *  *  *  *  *  *  .\9
         10\.  *  *  *  *  *  *  *  *  .  .\10
          11\.  *  *  *  *  *  *  *  .  .  .\11
              a  b  c  d  e  f  g  h  i  j  k  
63079 info: no subtree to reuse since old position not exists
63079 info: No subtree to reuse.
63079 info: Creating thread 0
63079 info: StartSearch()[0]
 0:05 | 0.424 | 8110 | 3.4 | e7 g6 g5 ^ f6 f5 e6 e5 i4
SgUctSearch: move cannot change anymore
63079 info: 
Elapsed Time   9.35722s
Count          14802
GamesPlayed    14802
Nodes          61741
Knowledge      35 (0.2%)
Expansions     656.0
Time           8.80
GameLen        120.0 dev=0.0 min=120.0 max=120.0
InTree         3.6 dev=1.6 min=0.0 max=11.0
KnowDepth      2.0 dev=1.0 min=0.0 max=4.0
Aborted        0%
Games/s        1681.9
Score          0.42
Sequence       e7 g6 g5 f6 f5 e6 e5 i4 h5
moveselect dithering threshold: 0	 num stones in current state:1
63079 info: 
= e7

quit
= 
```

### Train the neural net

In subdir ``simplefeature3``, there is a relatively independent Python project 
used to train the neural net. 

### Commands and Parameters

Newly added commands:
```
nn_load <path_to_nn_model> # if no parameter followed, display the loaded model
nn_evaluate #evaluate the current board state by a forward pass

param_mohex moveselect_ditherthreshold <int> 
# softmax selection according to #count when the number of stones is less than threshold
param_mohex use_playout_const <float, [0,1]> # how to combine playout result

```

Display default parameters:

```
param_mohex
= 
[bool] backup_ice_info 1
[bool] extend_unstable_search 1
[bool] lock_free 0
[bool] lazy_delete 1
[bool] perform_pre_search 1
[bool] prior_pruning 1
[bool] ponder 0
[bool] reuse_subtree 1
[bool] search_singleton 0
[bool] use_livegfx 0
[bool] use_parallel_solver 0
[bool] use_rave 1
[bool] use_root_data 1
[bool] use_time_management 0
[bool] weight_rave_updates 0
[bool] virtual_loss 1
[string] bias_term 0
[string] expand_threshold 10
[string] fillin_map_bits 16
[string] first_play_urgency 0.35
[string] playout_global_gamma_cap 0.157
[string] knowledge_threshold "256"
[string] number_playouts_per_visit 1
[string] max_games 99999999
[string] max_memory 1999999872
[string] max_nodes 10416666
[string] max_time 10
[string] move_select count
[string] num_threads 1
[string] progressive_bias 2.47
[string] vcm_gamma 282
[string] randomize_rave_frequency 30
[string] rave_weight_final 830
[string] rave_weight_initial 2.12
[string] time_control_override -1
[string] uct_bias_constant 0.22
[string] moveselect_ditherthreshold 0
[string] use_playout_const 0

```
new commands

```
nn_load 
#if no argument given, display loaded nn, otherwise load the nn either using 1) model name in share/nn/ if path is relative
or 2) absolute path to nn model

nn_ls #list available nn models at share/nn/

nn_evaluate # call single nn pass, display p,q,v values.

mohex-self-play n filetosave 
#mohex-self-play n games, then save the games to file

param_mohex root_dirichlet_prior 0.06
#arg is a flot number, if 0 means no dirichlet at all; 
#if root_dirichlet_prior is non-zero, then 
# p(s,a) = 0.75 *p(s,a) + 0.25*nosie, where noise is sampled from Dir(root_dirichlet_prior)
```

### MoHex-self-play game format
When one `mohex-self-play` command is finished, played games are saved offline with format: 

- Each line is one game 
- A game consists of a sequence of black and white moves, where black starts first 
- Each move follows a probability distribution from which the move was drawn
- At the end of each line lies the game value, w.r.t the player to play after the last move

One example game: 

``
B[b3] W[j4][j4:0.37;i5:0.06;h6:0.02;g7:0.05;f8:0.12;e9:0.12;d10:0.22;] B[i5][k4:0.08;i5:0.85;] W[i4][k2:0.01;l2:0.04;j3:0.01;k3:0.02;l3:0.25;i4:0.30;k4:0.10;j5:0.04;k5:0.15;k6:0.05;] B[h5][h5:0.95;] W[h4][h4:0.95;] B[g5][f5:0.02;g5:0.95;] W[g4][g4:0.98;] B[f5][e5:0.09;f5:0.89;] W[f4][f4:0.98;] B[e5][d5:0.35;e5:0.63;] W[e4][e4:0.96;] B[d5][c5:0.17;d5:0.79;] W[d4][d4:0.95;] B[c5][b5:0.19;c5:0.80;] W[c4][c4:0.96;] B[a5][a5:0.97;] W[b5][b5:0.97;] B[a6][a6:0.97;] W[b6][b6:0.97;] B[a7][a7:0.98;] W[b7][b7:0.96;] B[a8][a8:0.98;] W[b8][b8:0.98;] B[a9][a9:0.98;] W[b9][b9:0.97;] B[a10][a10:0.98;] W[b10][b10:0.91;b11:0.04;b12:0.03;] B[a11][a11:0.97;] W[b12][b11:0.02;b12:0.98;] B[b11][c10:0.14;b11:0.83;] W[c11][c11:0.54;d11:0.42;] B[c10][c10:0.95;] W[d10][d10:0.67;e10:0.19;f10:0.05;] B[d8][d8:0.68;d9:0.08;f9:0.03;e10:0.17;] W[d9][c2:0.01;g8:0.02;h8:0.07;d9:0.67;f9:0.03;g9:0.02;h9:0.03;f10:0.06;g10:0.06;] B[c9][c9:0.97;] W[h8][g8:0.04;h8:0.25;i8:0.12;h9:0.21;f10:0.22;g10:0.04;e11:0.01;f11:0.06;] B[i8][j7:0.02;g8:0.02;i8:0.91;f9:0.01;] W[i7][i7:0.73;h9:0.09;i9:0.14;] B[j7][k6:0.16;j7:0.71;g8:0.01;f9:0.08;] W[h9][j6:0.39;h9:0.57;] B[e10][j6:0.26;i9:0.23;j9:0.06;e10:0.42;] W[d12][c2:0.13;e8:0.02;f8:0.30;e11:0.10;d12:0.41;] B[j6][j6:0.91;i9:0.03;f11:0.03;] W[i9][i9:0.66;i10:0.30;] B[j9][k8:0.10;j9:0.85;f11:0.02;] W[j8][j8:0.96;] B[l7][l7:0.98;] W[k7][j5:0.01;k7:0.81;k8:0.04;f11:0.12;] B[l6][k6:0.04;l6:0.92;] W[k6][k2:0.05;l3:0.05;k4:0.05;k6:0.77;k8:0.04;f11:0.01;] B[l5][l5:0.96;] W[k8][j5:0.05;k5:0.06;k8:0.62;f11:0.23;] B[l8][l8:0.96;] W[k9][j5:0.09;k5:0.10;k9:0.75;f11:0.03;] B[l9][l9:0.95;k11:0.01;] W[k11][j5:0.04;k5:0.08;k10:0.13;k11:0.72;] B[k10][k10:0.92;f11:0.05;k12:0.01;] W[j11][j5:0.01;k5:0.10;j11:0.86;] B[j10][j10:0.96;k12:0.01;] W[j5][j5:0.57;k5:0.11;i11:0.29;] B[k5][k5:0.96;] W[i6][i6:0.96;i11:0.01;] B[l3][l3:0.51;f11:0.19;h11:0.21;i11:0.08;] W[i11][l2:0.06;i11:0.93;] B[k12][f8:0.10;m10:0.20;e11:0.15;f11:0.11;g11:0.09;h11:0.11;k12:0.22;] W[l10][l10:1.00;] B[h11][f8:0.04;g10:0.03;e11:0.15;f11:0.27;g11:0.19;h11:0.31;] W[g12][c2:0.02;d2:0.03;g12:0.95;] B[e12][f8:0.08;g10:0.01;g11:0.05;e12:0.56;f12:0.02;h12:0.25;] W[f11][f11:0.98;g11:0.01;] B[g10][f8:0.07;g10:0.54;h12:0.39;] W[e11][e11:1.00;] -1.000000
``

### Closed loop


A closed loop shell which does
1) call mohex-self-play produce a set of games
2) refine neural net on those games by running ../simplefeature3/neuralnet/resnet.py  
3) freeze trained neural net via ../simplefeature3/freeze_graph/freeze_graph_main.py
4) use nn_load to load new nn model 
5) Go back to step 1

 Since step 1) is rather time and computation consuming,
 e.g., 20-block version AlphaGo Zero generated over 4 million games.
 How fast is mohex-self-play on a single GPU GTX1080 computer? 
 If 1000 simulations per move, a move consumes about 1s (would be slower if using `expand_threshold 0`).
 So if a game has 60 moves, then it takes 10^6 minutes for 1 million game
 1 million minutes is 10**6/60.0 ~= 700 days

Closed loop scripts are at `closedloop/`

### MCTS

In-tree formula


$$score(s,a) = (1-w)\Big(Q(s,a) + c\sqrt{\frac{\log{N(s)}}{N(s,a)}}\Big) + wR(s,a)+c_{pb}\frac{p(s,a)}{\sqrt{N(s,a)+1}}$$ 

If ``use_rave`` is false, then 

$$score(s,a) = Q(s,a) + c_{pb}\frac{p(s,a)\sqrt{N(s)}}{N(s,a)+1}$$ 

``playout``: 
Use ``param_mohex use_playout_const`` to control the weight of random playout result backed up, default is 0. 
Setting it to ``1`` disables value net. 
### Further Reading

See 
+ Chao Gao, Siqi Yan, Ryan Hayward, Martin Mueller. [A transferable neural network for Hex](http://webdocs.cs.ualberta.ca/~hayward/papers/transferable.pdf) CG 2018.
+ Chao Gao, Kei TakaDa, Ryan Hayward. [Hex 2018: MoHex3HNN over DeepEzo](http://webdocs.cs.ualberta.ca/~hayward/papers/taiwan18.pdf). **21st Computer Olympiad Report: program MoHex3HNN based on three-head net and transfer learning (128 filters per layer, 10 residual blocks, iteratively trained on ~0.4 million self-play generated games) won 11x11 and 13x13 Hex tournaments. The challenger DeepEzo uses minimax search based RL for training. Previous champion MoHex2.0 is now a weak baseline compared to MoHex3HNN and DeepEzo**
+ Chao Gao, Martin Mueller, Ryan Hayward.
    [Three-Head Neural Network Architecture for Monte Carlo Tree Search](https://www.ijcai.org/proceedings/2018/0523.pdf). IJCAI-ECAI 2018.  
+ Chao Gao, Martin Mueller, Ryan Hayward. 
    [Adversarial Policy Gradient for Alterating Markov Games](https://openreview.net/forum?id=ByINFNJDz). Sixth International Conference on Learning Representations (ICLR 2018), Workshop track, 2018.  
+ Chao Gao, Ryan Hayward, Martin Mueller. 
    [Move Prediction using Deep Convolutional Neural Networks in Hex](https://ieeexplore.ieee.org/document/8226781/). IEEE Transaction on Games, 2017. This paper investigates the move prediction problem in Hex. By learning on MoHex 2.0 self-play generated data, MoHex-CNN achieves 70% winrate against MoHex 2.0 on 13x13 board size after using the learned knowledge as its in-tree prior probability. 
+ Chao Gao, Ryan Hayward, Martin Mueller. 
    [Focused depth-first proof number search using convolutional neural networks for the game of Hex](https://www.ijcai.org/proceedings/2017/513). Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI-2017). AAAI Press, 2017. 

### Authors

This project is developed by:


* **Chao Gao** - *main developer*
* **Jakub Pawlewicz** 

### License
This project is licensed under the GNU Lesser General Public License
as published by the Free Software Foundation - version 3. 
