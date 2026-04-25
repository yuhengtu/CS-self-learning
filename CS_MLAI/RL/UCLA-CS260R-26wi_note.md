# Links

gridworld_dp: https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html



Hws: https://github.com/ucla-rlcourse/cs260r-assignment-2026winter/

hw1: policy iteration & value iteration

hw2: tabular Q-learning, DQN, REINFORCE / policy gradient with or without baseline

hw3: TD3 & PPO

hw4: Behavior Cloning (BC), HG-DAgger, DPO



https://github.com/ucla-rlcourse/RLexample

https://github.com/ucla-rlcourse/RLexample/blob/master/my_learning_agent.py

https://github.com/ucla-rlcourse/RLexample/blob/master/pg-pong.py; Pong from Pixels http://karpathy.github.io/2016/05/31/rl/

Policy iteration and value iteration on FrozenLake: https://github.com/ucla-rlcourse/RLexample/tree/master/MDP

Sarsa and Q-Learning: https://github.com/ucla-rlcourse/RLexample/tree/master/modelfree

![image-20260305013955809](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603050140001.png)

Mountain Car for Q-learning with value fn: https://github.com/ucla-rlcourse/RLexample/blob/master/funcapproximate/mountain_car.ipynb

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081742335.png" alt="image-20260308174225253" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081742210.png" alt="image-20260308174236124" style="zoom:25%;" />



DeepRL: https://github.com/ucla-rlcourse/DeepRL-Tutorials

DQN: https://github.com/ucla-rlcourse/DeepRL-Tutorials/blob/master/01.DQN.ipynb, https://github.com/xmfbit/DQN-FlappyBird

Double DQN: https://github.com/ucla-rlcourse/DeepRL-Tutorials/blob/master/03.Double_DQN.ipynb

Dueling DQN: https://github.com/ucla-rlcourse/DeepRL-Tutorials/blob/master/04.Dueling_DQN.ipynb

Prioritized Experience Replay DQN: https://github.com/ucla-rlcourse/DeepRL-Tutorials/blob/master/06.DQN_PriorityReplay.ipynb



Derivative-free policy fn with Evolution Strategy/Cross-Entropy Method https://openai.com/index/evolution-strategies/, https://github.com/ucla-rlcourse/RLexample/blob/master/derivativefree/es_cem_agent.py, https://gist.github.com/kashif/5dfa12d80402c559e060d567ea352c06



Policy Gradient: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

REINFORCE code on CartPole: https://github.com/ucla-rlcourse/RLexample/blob/master/policygradient/reinforce.py

Policy Gradient on Pong: https://github.com/ucla-rlcourse/RLexample/blob/master/policygradient/pg-pong-pytorch.py

Policy Gradient with Baseline on Pong: https://github.com/ucla-rlcourse/RLexample/blob/master/policygradient/pgb-pong-pytorch.py

Actor Critic on Pong: https://github.com/ucla-rlcourse/RLexample/blob/master/policygradient/ac-pong-pytorch.py



https://spinningup.openai.com/en/latest/

https://github.com/DLR-RM/stable-baselines3

https://github.com/vwxyzjn/cleanrl

Deep RL Experimentation: https://www.youtube.com/watch?v=8EcdaCk9KaQ, http://joschu.net/docs/nuts-and-bolts.pdf

PPO: https://github.com/ucla-rlcourse/DeepRL-Tutorials/blob/master/14.PPO.ipynb

The 37 Implementation Details of PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

DDPG: https://github.com/sfujim/TD3/blob/master/DDPG.py

TD3: https://github.com/sfujim/TD3/blob/master/TD3.py

S2C: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

A3C: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py, https://github.com/greydanus/baby-a3c/blob/master/baby-a3c.py

A2C: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

Comparison between A3C and A2C: https://danieltakeshi.github.io/2018/06/28/a2c-a3c/



Imitation learning: https://github.com/HumanCompatibleAI/imitation



# Week 1 Intro

![image-20260108103813766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081038130.png)

![image-20260308175105449](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081751498.png)

https://old.reddit.com/r/MachineLearning/comments/xfmqny/d_what_happened_to_reinforcement_learning/

![image-20260308175337374](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081753419.png)

For RL, the model may achieve super-human performance (Upper bound for supervised learning is human-performance)

![image-20260205173500092](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051735023.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081043299.png" alt="image-20260108104308274" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081043883.png" alt="image-20260108104331858" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131016411.png" alt="image-20260113101614260" style="zoom:25%;" />

---

Trajectory $H_t=O_1,R_1,A_1,...,A_{t-1},O_t,R_t$: the sequence of observations, rewards, actions

State S; $S_t=f(H_t)$; Env state and agent state: $S_t^e=f^e(H_t)$, $S_t^a=f^a(H_t)$



1. Full observability $O_t=S_t^e=S_t^a$: agent directly observes the environment state, Markov decision process (MDP)

2. Partial observability: agent indirectly observes the environment, partially observable Markov decision process (POMDP), can be converted into MDPs

   E.g., black jack (only see public cards), Atari game with pixel observation

   Agent must construct its beliefs of the environment state: $S_t^a=(P(S_t^e=s_1),...,P(S_t^e=s_n))$



1. Policy

   ![image-20260108110623168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081106190.png)
   
2. Value function / Q-function

   ![image-20260108110942124](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081109152.png)

   ![image-20260108111048555](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081110581.png)

   ![image-20260108111031121](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081110145.png)

3. (World) Model. Sometimes world model is given, like F = ma

   Predict the next state: $\mathbb{P}[S_{t+1}=s^{\prime}\mid S_t=s,A_t=a]$

   Predict the next reward: $\mathbb{E}\left[R_{t+1}\mid S_t=s,A_t=a\right]$



Maze Example

![image-20260108111317310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081113347.png)

Policy-based & Value-based

![image-20260108111349652](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081113688.png)![image-20260108111405501](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081114536.png)

---

Types of RL Agents

1. What the Agent Learns

- Value-based
  - Explicit: Value function
  - Implicit: Policy (can derive a policy from value function)
- Policy-based
  - Explicit: policy
  - No value function
- Actor-Critic:
  - Explicit: policy and value function

2. if there is model

- Model-based
- Model-free

![image-20260108111809807](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081118849.png)

---

Planning VS RL

- Planning
  - Given model about how the environment works.
  - Compute how to act to maximize expected reward without external interaction.
- RL
  - Agent doesn’t know how world works, Interacts with world to implicitly learn how world works
  - Agent improves policy (also involves planning)

![image-20260108112019040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081120064.png)![image-20260108112034231](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081120270.png)

---

Contextual Bandits is widely applied to content recommendations, dynamic pricing: https://www.geteppo.com/blog/netflix-lyft-yahoo-contextual-bandits

![image-20260108112418217](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081124242.png)

---

OpenAI: (used to be) specialized in RL: https://github.com/openai/retro/tree/develop, https://github.com/openai/gym

![image-20260308180300754](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081803841.png)

https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

![image-20260308180346176](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081803214.png)

![image-20260308180419082](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081804167.png)



# Week 2 MDP & policy/value iteration

k-Armed Bandit as simplified RL

- no delayed reward, instance feedback
- only one state, thus independent of consecutive behaviors

At each step t the agent selects an action $a_t\in\mathcal{A}$, then the environment generates a reward $r_t\sim\mathcal{R}^{a_t}=P(r|a_t)$. The goal of agent is to maximize cumulative reward $\sum_{\tau=1}^{T}r_{\tau}$

Exploration:

- there is only one state, so Q only depends on a, thus Q is a vector with dimension k: $Q(a)=\mathbb{E}(r|a)$
- estimation of Q at time step t: $Q_t(a)=\frac{\sum_{i=1}^{t-1}r_i\cdot1_{A_i=a}}{\sum_{i=1}^{t-1}1_{A_i=a}}$, total rewards from a / # times to try a
- e.g., pull 12 times, arm1 3 times get reward 1, arm2 2 times get reward 5, arm3 1 times get reward 4. Q = [1/3, 2/5, 1/4]

Exploitation:

- Greedy: $A_t=\arg\max_aQ_t(a)$
- $\epsilon$-Greedy: greedy mostly, but with small probability  (usually $\epsilon=0.1$) select random actions

![image-20260113151950727](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131519808.png)

---

1. Markov Processes

The history of states: $h_t=\{s_1,s_2,s_3,...,s_t\}$

State st is Markovian if and only if: $p(s_{t+1}|s_t)=p(s_{t+1}|h_t), p(s_{t+1}|s_t,a_t)=p(s_{t+1}|h_t,a_t)$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131057566.png" alt="image-20260113105705433" style="zoom:25%;" />

2. (finite) Markov Reward Process: Markov Chain + reward function $R(s_t=s)=\mathbb{E}[r_t|s_t=s]$

- Horizon: Number of maximum time steps in each episode/trajectory (Per game: 100 moves for Go, 80 moves for chess)

- Return: Discounted sum of rewards from time step t to horizon, $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+...+\gamma^{T-t-1}R_T$

- State value: Expected return from t in state s, $V_t(s)=\mathbb{E}[G_t|s_t=s]$

For finite states, R can be represented as a vector

![image-20260113111310081](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131113176.png)

Bellman equation $V=R+\gamma PV$

Analytic solution: $V=(1-\gamma P)^{-1}R$, matrix inverse $O(N^3)$, only possible for small MRPs

Algorithms to avoid matrix inverse:

- Monte-Carlo

  ![image-20260113113337853](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131133932.png)

  ![image-20260113113358078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131133106.png)

- Temporal-Difference

![image-20260113154156270](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131541388.png)



3. Markov Decision Process: Markov Reward Process + decisions (actions $A$)

- transition matrix is $P(s_{t+1}=s^{\prime}|s_t=s,a_t=a)$
- reward function $R(s_t=s,a_t=a)=\mathbb{E}[r_t|s_t=s,a_t=a]$



Turn Markov Decision Process into Markov Reward Process: $P^\pi(s^{\prime}|s)=\sum_{a\in A}\pi(a|s)P(s^{\prime}|s,a), R^\pi(s)=\sum_{a\in A}\pi(a|s)R(s,a)$

MP/MRP VS MDP

![image-20260115102313260](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151023409.png)



Bellman Expectation Equation (expectation over policy $\pi$):

1. $v^\pi(s)=\sum_{a\in A}\pi(a|s)(R(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)v^\pi(s^{\prime}))$

![image-20260115180114784](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151801901.png)

2. $q^{\pi}(s,a)=R(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)\sum_{a^{\prime}\in A}\pi(a^{\prime}|s^{\prime})q^{\pi}(s^{\prime},a^{\prime})$

![image-20260115180139775](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151801842.png)

---

Two kinds of problems in MDP:

- Policy Evaluation / Value Prediction
- Control: search the optimal policy

Both can be solved by dynamic programming



Policy Evaluation / Value Prediction

e.g., usually R depends on s and a, here we simplify s.t. R only depends on s

![image-20260115181523161](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151815332.png)

initialize at all zero, update/backup rule at iteration t: $v_{t}^{\pi}(s)=\sum_{a}P(\pi(s)=a)(r(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)v_{t-1}^{\pi}(s^{\prime}))$, until converge

1. deterministic policy $\pi$ = Left and $\gamma=0$ for any state: $V^\pi=[5,0,0,0,0,0,10]$

2. deterministic policy $\pi$ = Left and $\gamma=0.5$ for any state: $V^\pi=[10,\mathrm{~5,~2.5,~1.25,~0.625,~0.3125,~10.15625}]$

3. Stochastic policy P($\pi$=Left) = 0.5 and P($\pi$=Right) = 0.5 and $\gamma=0.5$ for any state s: 

   $V(s_4) = 0.5(\text{p go left})*(0+0.5(\gamma)*V(s_3)) + 0.5*(0+0.5*V(s_5))$




Policy Evaluation, 2D examples:

![image-20260115184658233](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151846348.png)

step1: $v_{t-1}^{\pi}(s^{\prime}) = 0$, only consider instant reward $r(s,a)$

![image-20260115185618642](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151856792.png)



Control

Can have multiple optimal policy, but only one optimal value

Brute-force Policy Search: Number of deterministic policies is $|\mathcal{A}|^{|\mathcal{S}|}$ (say 5 actions, 10 states, each state there is 5 possbile deterministic actions), for each deterministic policy we do policy evaluation

Two efficient ways:

- Policy iteration

- Value Iteration



Summary

![image-20260116004040385](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601160040488.png)

---

If state set is very large, DP may be very slow. Asychronous DP do not sweep over state space

- In-place dynamic programming

  ![image-20260205183152497](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051831564.png)

- Prioritized sweeping: update some state more frequently

  ![image-20260205183319280](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051833310.png)

- Real-time dynamic programming: do not wait until the episode ends

  ![image-20260205183704901](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051837934.png)



# Week 3 Tabular model-free prediction/control

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051845278.png" alt="image-20260205184537201" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051845436.png" alt="image-20260205184552375" style="zoom:25%;" />



MC does not assume state is Markov, can only applied to episodic MDPs (each episode terminates)

MC Policy Evaluation

![image-20260205185032821](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051850885.png)



TD vs MC vs DP

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051912895.png" alt="image-20260205191205863" style="zoom:25%;" />

![image-20260205191444659](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051914685.png)

![image-20260304235444984](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603071914232.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051916984.png" alt="image-20260205191616948" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051916537.png" alt="image-20260205191629469" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051917167.png" alt="image-20260205191641924" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051917160.png" alt="image-20260205191748060" style="zoom:25%;" />



Model-free control: Generalized Policy Iteration (GPI) with MC or TD in the loop



off-policy benefits

![image-20260305000106798](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603050001876.png)



Sarsa & Q-learning

![image-20260305013344208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603050133330.png)

![image-20260305013521791](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603050135859.png)



off-policy MC/TD with importance sampling

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603071859195.png" alt="image-20260307185911085" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603071900157.png" alt="image-20260307190032088" style="zoom:30%;" />



Q-learning does not need importance sampling, [reason](https://www.quora.com/Why-doesn-t-DQN-use-importance-sampling-Dont-we-always-use-this-method-to-correct-the-sampling-error-produced-by-the-off-policy)

- Q-learning uses a deterministic policy so no action probability
- Q-learning is expected over the transition distribution, not over policy distribution, thus no need to correct different policy distributions

Q-learning can be considered as sample update of value iteration

- value iteration: use the expected value over the transition dynamics
- Q-learning: use the sample collected from the environment

---

Eligibility Traces

similar to n-step TD, provide another middle form between MC and TD

$G_{t:t+1} = r_{t+1} + \gamma V(s_{t+1})$

$G_{t:t+2} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})$
$$
\begin{aligned}

G_t^{\lambda}
&= (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t \\
&= (1-\lambda)G_{t:t+1} + (1-\lambda)\lambda G_{t:t+2} + (1-\lambda)\lambda^2 G_{t:t+3} + \cdots + (1-\lambda)\lambda^{T-t-2} G_{t:T-1} + \lambda^{T-t-1} G_t
\end{aligned}
$$
λ = 1 -> MC; λ = 0 -> TD(0)

![image-20260308003214266](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603080032340.png)

Based on this, we can have TD(λ), SARSA(λ), Q-learning(λ)



# Week 4 Value fn & DQN

value fn approximation or VFA (linear value fn/NN); Tabular is a special case of linear fn

![image-20260308172103772](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081721891.png)

![image-20260308172203839](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081722921.png)



MC Return $G_t$  is **unbiased** but noisy sample of true value $v^\pi(S_t)$: $\mathbb{E}[G_t] = v^\pi(S_t)$


TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ is a **biased** estimate of true value $v^\pi(S_t)$ because it is drawn from a previous estimate: $\mathbb{E}[R_{t+1} + \gamma \hat{v}(S_{t+1}, w)] \neq v^\pi(S_t)$

TD(0) uses semi-gradient, which means the target depends on parameter w, but during differentiation we pretend it does not: $\Delta \mathbf{w} = \alpha \left( R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w})$

TD/Sarsa/Q-learning with value fn don’t follow the gradient of any objective functioncan, can diverge when off-policy or using non-linear fn

---

Batch RL: replay buffer / Least-squares policy evaluation (a third method for policy eval besides MC and TD)

Given T (st, at) observations generated from the policy to be evaluated, use least squares $\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{t=1}^{T} (v_t - \hat{v}(s_t,\mathbf{w}))^2$

LSMC: Least squares MC: $v_t^{\pi} \approx G_t$

LSTD: Least squares TD: $v_t^{\pi} \approx R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$

[

LSPI: I don't know wtf the prof and his slide are talking about for this part

https://users.cs.duke.edu/~parr/jmlr03.pdf

]



Convergence of Control Methods

![image-20260308191613129](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603081916256.png)

---

DQN from DeepMind’s Nature paper

Instability: Nonlinear function approximation, Bootstrapping, Off-policy training

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082028896.png" alt="image-20260308202812778" style="zoom:25%;" />



Improving DQN

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082035956.png" alt="image-20260308203528810" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082036833.png" alt="image-20260308203606726" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082036850.png" alt="image-20260308203629730" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082037008.png" alt="image-20260308203737895" style="zoom:25%;" />



# Week 5 Policy-based RL

policy based

![image-20260308204453141](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082044242.png)

![image-20260308204544571](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082045674.png)

the action on grey states must be the same

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082048979.png" alt="image-20260308204810879" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082048445.png" alt="image-20260308204858335" style="zoom:25%;" />



objective fn

![image-20260308204958057](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082049160.png)

optimization methods

![image-20260308205153010](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603082051115.png)



Derivative-free methods

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090127631.png" alt="image-20260309012740523" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090129461.png" alt="image-20260309012953378" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090134947.png" alt="image-20260309013425869" style="zoom:20%;" />



Policy Gradient

$\nabla_\theta \pi_\theta(a\mid s) = \pi_\theta(a\mid s)\frac{\nabla_\theta \pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)$, where $\nabla_\theta \log \pi_\theta(a\mid s)$ is known as score function in stats

Softmax policy/Gaussian policy/NN policy

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090147645.png" alt="image-20260309014747558" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090147923.png" alt="image-20260309014754848" style="zoom:25%;" />

Policy Gradient for Multi-step MDPs, see DIS 4 for more

![image-20260309015522115](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090156543.png)

![image-20260309015535359](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090155428.png)

![image-20260309015754191](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090157297.png)

![image-20260309015832508](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603090158586.png)

this is unbiased but high variance because we use MC

we can reduce variance by

- using causality
- using baseline
- TD



policy gradient is shifting the policy distri to achiving higher reward

![image-20260309154028581](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091540695.png)



by using causuality, transform this to the formula into Westlake-MathFoundationRL LEC9 formula:

$R(\tau)\sum_t \nabla \log \pi(a_t \mid s_t)$ -> $\sum_t G_t \nabla \log \pi(a_t \mid s_t)$ -> $\sum_t q^\pi(s_t, a_t)\nabla \log \pi(a_t \mid s_t)$

where $R(\tau) = \sum_{t=0}^{T} r_t$, $G_t = \sum_{k=t}^{T} r_k$

![image-20260309154521864](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091545982.png)

![image-20260309154715750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091547840.png)

If $R(\tau) = \sum_{k=0}^{T} \gamma^k r_k$, $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$

we have $R(\tau)\sum_t \nabla \log \pi(a_t \mid s_t)$ -> $\sum_t \gamma^t G_t \nabla \log \pi(a_t \mid s_t)$ -> $\sum_t \gamma^t q^\pi(s_t, a_t)\nabla \log \pi(a_t \mid s_t)$



Policy Gradient with Baseline

in pytorch we do not explicitly calculate gradient, we just give the loss; $\|b(s_t)-\hat{R}_t\|^2$ this is for calculating the empirical baseline

![image-20260309162051108](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091620193.png)



Actor-Critic

![image-20260309162522878](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091625974.png)



Compatible Function Approximation

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ Q^{\pi_\theta}(s,a)\nabla_\theta \log \pi_\theta(a|s) \right]$, does replace $Q^{\pi_\theta}(s,a)$ with value fn $Q_w(s,a)$ bias the policy gradient?

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091630379.png" alt="image-20260309163000290" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091648606.png" alt="image-20260309164800486" style="zoom:25%;" />



Advantage Actor-Critic

![image-20260309165113147](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091651245.png)

![image-20260309165144394](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091651486.png)



Policy gradient overcome non-differentiable computation

input → policy network → sample action → environment → reward

![image-20260309165611816](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091656918.png)



Summary

State-of-the-art RL methods are almost all policy-based: A2C, A3C, TRPO, PPO

![image-20260309170231632](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091702731.png)



# Week 6 Policy-based RL

![image-20260309171058614](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091710764.png)

![image-20260309171235935](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091712022.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091828639.png" alt="image-20260309182849589" style="zoom:25%;" />

![image-20260309180851978](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091808027.png)

---

Policy gradient is on-policy and thus inefficient; and also unstable

![image-20260309171457535](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091714736.png)



TRPO/PPO idea

importance sampling code example: https://machinelearning1.wordpress.com/2017/10/22/importance-sampling-a-tutorial/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091716990.png" alt="image-20260309171645947" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091737242.png" alt="image-20260309173716095" style="zoom:25%;" />



Prior Work: Natural Policy Gradient, https://agustinus.kristia.de/blog/natural-gradient/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091729761.png" alt="image-20260309172943623" style="zoom: 25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091729829.png" alt="image-20260309172951744" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091730253.png" alt="image-20260309173014111" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091730270.png" alt="image-20260309173057229" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091742844.png" alt="image-20260309174238803" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091742638.png" alt="image-20260309174248548" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091743878.png" alt="image-20260309174300738" style="zoom:25%;" />



TRPO: Trust Region Policy Optimization

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091746239.png" alt="image-20260309174626197" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091746203.png" alt="image-20260309174641067" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091756403.png" alt="image-20260309175648358" style="zoom:20%;" />



Appendix of TRPO

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091747768.png" alt="image-20260309174712682" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091747061.png" alt="image-20260309174721975" style="zoom:25%;" />



TRPO -> ACKTR: Calculating Natural Gradient with KFAC, https://openai.com/index/openai-baselines-acktr-a2c/

ACKTR idea from: Optimizing Neural Networks with Kronecker-factorized Approximate Curvature

![image-20260309175732526](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091757569.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091759659.png" alt="image-20260309175909566" style="zoom:25%;" />



TRPO -> PPO: adaptive KL Penalty, https://openai.com/index/openai-baselines-ppo/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091800035.png" alt="image-20260309180014942" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091800273.png" alt="image-20260309180034181" style="zoom:30%;" />



PPO -> PPO2: clipped PPO

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091801818.png" alt="image-20260309180118715" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091802668.png" alt="image-20260309180205564" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091802464.png" alt="image-20260309180217362" style="zoom:25%;" />

---

DDPG

![image-20260309181322187](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091813281.png)![image-20260309181455574](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091814626.png)



TD3: Twin Delayed DDPG

overestimate means predict > true

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091817481.png" alt="image-20260309181713383" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091818052.png" alt="image-20260309181815948" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091818389.png" alt="image-20260309181849293" style="zoom:20%;" />



Soft Actor-Critic (SAC): SOTA for robot learning

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091823769.png" alt="image-20260309182313671" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091823250.png" alt="image-20260309182346155" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091824686.png" alt="image-20260309182440589" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091824278.png" alt="image-20260309182449179" style="zoom:25%;" />



Reparameterization Trick: https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/

![image-20260309182622110](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091826212.png)



# Week 7 Model-based RL

Model-based RL: learning env model from experience (state transition & reward model), plan value/policy from model

Model-free RL: learn value/policy from experience

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091846195.png" alt="image-20260309184654040" style="zoom:25%;" />![image-20260309185013322](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091850439.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091856911.png" alt="image-20260309185632752" style="zoom:20%;" />

---

**Model-based RL**

- Pros:

  - higher sample efficiency, crucial for real-world applications such as robotic manipulation. Model can be learned efficiently by supervised learning

    ![image-20260309185424531](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091854593.png)

- Cons:

  - difficult to have guarantee of convergence (First learning a model then constructing value or policy fn leads to two sources of approximation error)

---

**Model-based value optimization**

![image-20260309185749836](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091857940.png)

the models for the env model

1. Table Lookup Model

![image-20260309185842577](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091902187.png)![image-20260309185946851](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091859002.png)

Others: Linear Expectation Model, Linear Gaussian Model, Gaussian Process Model, Deep Belief Network Model



Sample-Based Planning: use the learned env model to generate experiences, use model-free RL to learn from those experiences

![image-20260309190239206](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091902310.png)



Policy obtained from model-based RL is only as good as the estimated env model, for inaccurate env model, we can reason explicitly about the model uncertainty (how confident we are for the estimated state): Use probabilistic model such as Bayesian and Gaussian Process



Dyna: Learn and plan value/policy from both real and simulated experience

![image-20260309190524723](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091905831.png)![image-20260309190902559](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091909672.png)

assume deterministic env, (s,a) gives deterministic (s',r), store them in memory (the model(S,A))

![image-20260309191037039](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091910141.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603091916397.png" alt="image-20260309191633276" style="zoom:25%;" />

---

**Model-based policy optimization**

Policy gradient does not need transition $p(s_{t+1}|s_t, a_t)$, but maybe we can do better if we know

strongly influenced from the control theory that optimizes a controller

![image-20260310161909520](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101619637.png)![image-20260310161953774](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101619851.png)



Algorithms

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101621009.png" alt="image-20260310162058110" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101622669.png" alt="image-20260310162217510" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101623579.png" alt="image-20260310162315342" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101623983.png" alt="image-20260310162351729" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101625503.png" alt="image-20260310162549285" style="zoom:20%;" />

The env model

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101626337.png" alt="image-20260310162640181" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101628305.png" alt="image-20260310162844127" style="zoom:25%;" />

---

**Case studies on robot object manipulation and learning world models from images**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101630888.png" alt="image-20260310163051675" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101631635.png" alt="image-20260310163103456" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101631645.png" alt="image-20260310163125473" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101631419.png" alt="image-20260310163159248" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101634923.png" alt="image-20260310163406741" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101634566.png" alt="image-20260310163418391" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101634637.png" alt="image-20260310163457463" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101635575.png" alt="image-20260310163516373" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101637182.png" alt="image-20260310163705015" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101637140.png" alt="image-20260310163720968" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101638950.png" alt="image-20260310163840748" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101641680.png" alt="image-20260310164126503" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101642981.png" alt="image-20260310164202799" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101643837.png" alt="image-20260310164333651" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101643690.png" alt="image-20260310164344513" style="zoom:25%;" />



# Week 8 Imitation learning

Outline [see original slide for detailed case studies]

![image-20260310165031219](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101650396.png)

---

**Imitation Learning**: Supervised learning of the policy network

Using GenAI or world model to synthesize off-course situations

![image-20260310165152087](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101651278.png)![image-20260310165531946](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101657057.png)

---

**DAGGER**, run current policy and observe, ask human to label these observations

![image-20260310170104120](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101701217.png)![image-20260310165931788](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101659980.png)

---

**Interactive Imitation Learning**

![image-20260310170340783](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101703981.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101706912.png" alt="image-20260310170609712" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101708436.png" alt="image-20260310170815243" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101710964.png" alt="image-20260310171039774" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101717409.png" alt="image-20260310171730210" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101717094.png" alt="image-20260310171748913" style="zoom:20%;" />

Robot-Gated Intervention: Based on uncertainty estimation, agent requests human help when uncertain

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101718492.png" alt="image-20260310171848288" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101720008.png" alt="image-20260310172019799" style="zoom:25%;" />

---

**Inverse RL and Generative Adversarial Imitation Learning(GAIL)**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101721484.png" alt="image-20260310172120306" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101722265.png" alt="image-20260310172220082" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101722322.png" alt="image-20260310172231137" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101725932.png" alt="image-20260310172557720" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101726057.png" alt="image-20260310172633878" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101727800.png" alt="image-20260310172733612" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101728884.png" alt="image-20260310172820697" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101728204.png" alt="image-20260310172832009" style="zoom:20%;" />

---

**Improving the Supervised Imitation Learning**

policy model

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101731652.png" alt="image-20260310173115448" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101731561.png" alt="image-20260310173156373" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101732740.png" alt="image-20260310173212552" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101732697.png" alt="image-20260310173235495" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101733969.png" alt="image-20260310173350762" style="zoom:10%;" />

imitation data collection

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101735970.png" alt="image-20260310173549779" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101736947.png" alt="image-20260310173605759" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101736195.png" alt="image-20260310173629996" style="zoom:20%;" />

---

**Unifying RL and Imitation Learning**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101738452.png" alt="image-20260310173807248" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101739773.png" alt="image-20260310173908538" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101740900.png" alt="image-20260310174050689" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101741913.png" alt="image-20260310174150713" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101745622.png" alt="image-20260310174548410" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101753856.png" alt="image-20260310175315649" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101753417.png" alt="image-20260310175330234" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101755891.png" alt="image-20260310175504699" style="zoom:20%;" />

---

**Case study**

Guided Policy Search

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101804246.png" alt="image-20260310180411038" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101807958.png" alt="image-20260310180712747" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101808083.png" alt="image-20260310180821884" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101808850.png" alt="image-20260310180858642" style="zoom:15%;" />

Others

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101811800.png" alt="image-20260310181127592" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101812299.png" alt="image-20260310181224088" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101812206.png" alt="image-20260310181233997" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101813400.png" alt="image-20260310181306192" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101813538.png" alt="image-20260310181357304" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101814891.png" alt="image-20260310181426697" style="zoom:20%;" />

---

**Summary**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101815467.png" alt="image-20260310181540253" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101816131.png" alt="image-20260310181607924" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603101816749.png" alt="image-20260310181658527" style="zoom:20%;" />



# Week9 Distributed computing and RL system design

Outline [see original slide for detailed case studies]

![image-20260314101417438](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141014559.png)

---

**Parallelism in Distributed ML Systems**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141015435.png" alt="image-20260314101504406" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141017040.png" alt="image-20260314101750963" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141018523.png" alt="image-20260314101823497" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141020515.png" alt="image-20260314102008484" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141020176.png" alt="image-20260314102035114" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141021736.png" alt="image-20260314102103707" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141021817.png" alt="image-20260314102133789" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141023439.png" alt="image-20260314102316409" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141030379.png" alt="image-20260314103055346" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141031721.png" alt="image-20260314103137658" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141034734.png" alt="image-20260314103432700" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141034034.png" alt="image-20260314103454008" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141035711.png" alt="image-20260314103533643" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141036558.png" alt="image-20260314103648489" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141036474.png" alt="image-20260314103659405" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141037603.png" alt="image-20260314103750517" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141038806.png" alt="image-20260314103805740" style="zoom:20%;" />

---

**Development of Distributed RL Systems**

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141038512.png" alt="image-20260314103824442" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141039390.png" alt="image-20260314103953320" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141040053.png" alt="image-20260314104005992" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141040086.png" alt="image-20260314104055020" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141042425.png" alt="image-20260314104232359" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141042583.png" alt="image-20260314104252516" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141043246.png" alt="image-20260314104323180" style="zoom:15%;" />

https://openai.com/blog/baselines-acktr-a2c/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141046753.png" alt="image-20260314104621678" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141046060.png" alt="image-20260314104643991" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141048395.png" alt="image-20260314104832322" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141049269.png" alt="image-20260314104954186" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141050030.png" alt="image-20260314105006957" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141050060.png" alt="image-20260314105036988" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141050967.png" alt="image-20260314105050898" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141052790.png" alt="image-20260314105204720" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141052584.png" alt="image-20260314105215512" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141052094.png" alt="image-20260314105225025" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141053293.png" alt="image-20260314105339259" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141054501.png" alt="image-20260314105423427" style="zoom:20%;" />

---

**Case Study**

![image-20260314105542639](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141055715.png)

https://yuandong-tian.com/reproducibility.pdf

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141105901.png" alt="image-20260314110509861" style="zoom:25%;" />

https://openai.com/index/openai-five-defeats-dota-2-world-champions/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141057317.png" alt="image-20260314105704241" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141103063.png" alt="image-20260314110322974" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141104523.png" alt="image-20260314110405434" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141104831.png" alt="image-20260314110418755" style="zoom:25%;" />

https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141105614.png" alt="image-20260314110536521" style="zoom:25%;" />

https://sites.google.com/view/isaacgym-nvidia

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141107254.png" alt="image-20260314110735170" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141109257.png" alt="image-20260314110921173" style="zoom:25%;" />

https://isaac-sim.github.io/IsaacLab/main/index.html

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141109578.png" alt="image-20260314110934501" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141110420.png" alt="image-20260314111003337" style="zoom:25%;" />

From Indoor Simulation to Urban Simulation, see original slide for details

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202603141110549.png" alt="image-20260314111042470" style="zoom:25%;" />

Simulation infrastructures are crucial for RL training, which can narrow sim2real gap



# DIS

in policy iteration, after each policy eval loop the values are unbiased w.r.t. current policy

in value iteration, they are biased until converged
