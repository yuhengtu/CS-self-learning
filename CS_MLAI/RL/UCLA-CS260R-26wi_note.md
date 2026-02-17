# LEC1 Intro

![image-20260108103813766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081038130.png)

![image-20260108103914803](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081039824.png)

![image-20260205173500092](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051735023.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081043299.png" alt="image-20260108104308274" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081043883.png" alt="image-20260108104331858" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131016411.png" alt="image-20260113101614260" style="zoom:25%;" />



# LEC2

Trajectory $H_t=O_1,R_1,A_1,...,A_{t-1},O_t,R_t$: the sequence of observations, rewards, actions

State S; $S_t=f(H_t)$; Env state and agent state: $S_t^e=f^e(H_t)$, $S_t^a=f^a(H_t)$



1. Full observability $O_t=S_t^e=S_t^a$: agent directly observes the environment state, Markov decision process (MDP)

2. Partial observability: agent indirectly observes the environment, partially observable Markov decision process (POMDP), can be converted into MDPs

   E.g., black jack (only see public cards), Atari game with pixel observation

   Agent must construct its beliefs of the environment state: $S_t^a=(P(S_t^e=s_1),...,P(S_t^e=s_n))$



1. Policy $\pi(a|s)$: agent’s behavior function, map from state/observation to action.

   - Stochastic policy: Probabilistic sample. $\pi(a|s)=P[A_t=a|S_t=s]$
   - Deterministic policy: $a^*=\arg\max_a\pi(a|s)$, $a_t^*=\arg\max_aQ(s_t,a)$

   ![image-20260108110623168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081106190.png)

2. Value function, how good is each state or action (usually state):

   $v_\pi(s)\doteq\mathbb{E}_\pi[G_t\mid S_t=s]=\mathbb{E}_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k+1}|S_t=s\right],\text{ for all }s\in\mathcal{S}$

   Q-function (could be used to select among actions) is a 2D table of actions and states:

   $q_\pi(s,a)\doteq\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]=\mathbb{E}_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k+1}|S_t=s,A_t=a\right]$

   ![image-20260108110942124](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081109152.png)

   ![image-20260108111048555](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081110581.png)

   ![image-20260108111031121](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081110145.png)

3. (World) Model: A model predicts what the environment will do next. Sometimes world model is given, like F = ma

   Predict the next state: $\mathcal{P}_{ss^{\prime}}^a=\mathbb{P}[S_{t+1}=s^{\prime}\mid S_t=s,A_t=a]$

   Predict the next reward: $\mathcal{R}_s^a=\mathbb{E}\left[R_{t+1}\mid S_t=s,A_t=a\right]$



Maze Example

![image-20260108111317310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081113347.png)

Policy-based VS Value function-based

![image-20260108111349652](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081113688.png)![image-20260108111405501](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081114536.png)



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
  - Explicit: model
  - May or may not have policy and/or value function
- Model-free
  - Explicit: value function and/or policy function
  - No model

![image-20260108111809807](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081118849.png)



Planning VS RL

- Planning
  - Given model about how the environment works.
  - Compute how to act to maximize expected reward without external interaction.
- RL
  - Agent doesn’t know how world works, Interacts with world to implicitly learn how world works
  - Agent improves policy (also involves planning)

![image-20260108112019040](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081120064.png)![image-20260108112034231](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081120270.png)



Exploration and Exploitation (Agent only experiences what happens for the actions it tries)

For RL reward may be delayed

Contextual Bandits is widely applied to content  recommendations, dynamic pricing...

![image-20260108112418217](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601081124242.png)



# LEC3

Reward: scalar; expected cumulative reward: $E_\pi[G_t]$, where $G_t=\sum_k^\infty R_{t+k+1}$

Policy $\pi(a|s)$: agent’s behavior function

Value function $V(s)$: how good is each state or action

Action-value function $Q(s,a)$: how good is an action in a certain state

Model: agent’s state representation of the environment



k-Armed Bandit as simplified RL

- no delayed reward, instance feedback
- only one state, thus independent of consecutive behaviors

At each step t the agent selects an action $a_t\in\mathcal{A}$, then the environment generates a reward $r_t\sim\mathcal{R}^{a_t}=P(r|a_t)$. The goal of agent is to maximize cumulative reward $\sum_{\tau=1}^{T}r_{\tau}$

Exploration:

- there is only one state, so Q only depends on a, thus Q is a vector with dimension k: $Q(a)=\mathbb{E}(r|a)$
- estimation of Q at time step t: $Q_t(a)=\frac{\sum_{i=1}^{t-1}r_i\cdot1_{A_i=a}}{\sum_{i=1}^{t-1}1_{A_i=a}}$, prior to t, total rewards from a / # times to try a
- e.g., pull 12 times, arm1 3 times get reward 1, arm2 2 times get reward 5, arm3 1 times get reward 4. Q = [1/3, 2/5, 1/4]

Exploitation:

- Greedy: $A_t=\arg\max_aQ_t(a)$
- $\epsilon$-Greedy: greedy mostly, but with small probability  (usually $\epsilon=0.1$) select random actions $A_t=uniform(\mathcal{A})$

![image-20260113151950727](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131519808.png)



1. Markov Processes

The history of states: $h_t=\{s_1,s_2,s_3,...,s_t\}$

State st is Markovian if and only if: $p(s_{t+1}|s_t)=p(s_{t+1}|h_t), p(s_{t+1}|s_t,a_t)=p(s_{t+1}|h_t,a_t)$

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131057566.png" alt="image-20260113105705433" style="zoom:25%;" />

2. (finite) Markov Reward Process: Markov Chain + reward function $R(s_t=s)=\mathbb{E}[r_t|s_t=s]$

- Horizon: Number of maximum time steps in each episode/trajectory (Per game: 100 moves for Go, 80 moves for chess)

- Return: Discounted sum of rewards from time step t to horizon, $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+...+\gamma^{T-t-1}R_T$

- State value function $Vt(s)$: Expected return from t in state s, $V_t(s)=\mathbb{E}[G_t|s_t=s]$

For finite states, R can be represented as a vector

![image-20260113111310081](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131113176.png)

Bellman equation: $V(s)
= \underbrace{R(s)}_{\text{immediate reward}} + \underbrace{\gamma \sum_{s' \in S} P(s' \mid s)\, V(s')}_{\text{discounted expected future reward}}$

matrix form $V=R+\gamma PV$: $\begin{bmatrix}V(s_1)\\V(s_2)\\\vdots\\V(s_N)\end{bmatrix}=\begin{bmatrix}R(s_1)\\R(s_2)\\\vdots\\R(s_N)\end{bmatrix}+\gamma\begin{bmatrix}P(s_1|s_1)&P(s_2|s_1)&\ldots&P(s_N|s_1)\\P(s_1|s_2)&P(s_2|s_2)&\ldots&P(s_N|s_2)\\\vdots&\vdots&\ddots&\vdots\\P(s_1|s_N)&P(s_2|s_N)&\ldots&P(s_N|s_N)\end{bmatrix}\begin{bmatrix}V(s_1)\\V(s_2)\\\vdots\\V(s_N)\end{bmatrix}$

Analytic solution: $V=(1-\gamma P)^{-1}R$, matrix inverse $O(N^3)$, only possible for small MRPs

![image-20260113112532546](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131125667.png)

Algorithms to avoid matrix inverse:

- Monte-Carlo evaluation (sampling)

  ![image-20260113113337853](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131133932.png)

  ![image-20260113113358078](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131133106.png)

- Temporal-Difference learning

![image-20260113154156270](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601131541388.png)



3. Markov Decision Process: Markov Reward Process + decisions (actions $A$)

For finite set of states $S$ and finite set of actions $A$

- transition matrix is $P(s_{t+1}=s^{\prime}|s_t=s,a_t=a)$
- reward function $R(s_t=s,a_t=a)=\mathbb{E}[r_t|s_t=s,a_t=a]$



# LEC4

Policy: $\pi(a|s)=P(a_t=a|s_t=s)$, stationary (time-independent): $A_t\sim\pi(a|s)\text{ for any }t>0$

turn MDP into Markov reward process: $P^\pi(s^{\prime}|s)=\sum_{a\in A}\pi(a|s)P(s^{\prime}|s,a), R^\pi(s)=\sum_{a\in A}\pi(a|s)R(s,a)$

MP/MRP VS MDP

![image-20260115102313260](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151023409.png)

Value function $v^{\pi}(s)$: the expected return starting from state s and following policy $\pi$: $v^\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]$

Action-value function $q^\pi(s,a)$: the expected return starting from state s, taking action a, and then following policy $\pi$: $q^\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,A_t=a]$; let s' be next state, $q^\pi(s,a)=R_s^a+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)v^\pi(s^{\prime})$

relation between them: $v^\pi(s)=\sum_{a\in A}\pi(a|s)q^\pi(s,a)$



Recall: Bellman Equation $V(s)=R(s)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s)V(s^{\prime})$

Bellman Expectation Equation (expectation over policy $\pi$):

1. $v^\pi(s)=\sum_{a\in A}\pi(a|s)(R(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)v^\pi(s^{\prime}))$

![image-20260115180114784](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151801901.png)

2. $\large q^{\pi}(s,a)=R(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)\sum_{a^{\prime}\in A}\pi(a^{\prime}|s^{\prime})q^{\pi}(s^{\prime},a^{\prime})$

![image-20260115180139775](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151801842.png)



Two kinds of problems in MDP:

- Policy Evaluation / Value Prediction: evaluate a given policy, compute the value fn
- Control: search the optimal policy, and thus have the optimal value fn

Both can be solved by dynamic programming



Policy Evaluation / Value Prediction (Synchronous update)

e.g., usually R depends on s and a, here we simplify s.t. R only depends on s

![image-20260115181523161](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151815332.png)

initialize at all zero, update/backup rule at iteration t: $v_{t}^{\pi}(s)=\sum_{a}P(\pi(s)=a)(r(s,a)+\gamma\sum_{s^{\prime}\in S}P(s^{\prime}|s,a)v_{t-1}^{\pi}(s^{\prime}))$, until converge

1. deterministic policy $\pi$ = Left and $\gamma=0$ for any state: $V^\pi=[5,0,0,0,0,0,10]$

2. deterministic policy $\pi$ = Left and $\gamma=0.5$ for any state: $V^\pi=[10,\mathrm{~5,~2.5,~1.25,~0.625,~0.3125,~10.15625}]$

3. Stochastic policy P($\pi$=Left) = 0.5 andP($\pi$=Right) = 0.5 and $\gamma=0.5$ for any state s: 

   $V(s_4) = 0.5(\text{p go left})*(0+0.5(\gamma)*V(s_3)) + 0.5*(0+0.5*V(s_5))$




Policy Evaluation, 2D examples:

![image-20260115184658233](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151846348.png)

step1: $v_{t-1}^{\pi}(s^{\prime}) = 0$, only consider instant reward $r(s,a)$

![image-20260115185618642](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601151856792.png)

online example: https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html



Control

Can have multiple optimal policy, but only one optimal value

Brute-force Policy Search: Number of deterministic policies is $|\mathcal{A}|^{|\mathcal{S}|}$ (say 5 actions, 10 staes, each state there is 5 possbile deterministic actions), for each deterministic policy we do policy evaluation

Two efficient ways:

- Policy iteration: Iterate between:

  - Evaluate the policy (the state-action value, i.e. the q table, of the policy): do until converge
  - Update policy by acting greedily, i.e., take the action with highest q value for each state: $\pi_{i+1}(s)=\arg\max_aq^{\pi_i}(s,a)$

  when optimal, we achieve the Bellman optimality equation: $v^\pi(s)=\max_{a\in\mathcal{A}}q^\pi(s,a)$

- Value Iteration: turn Bellman Optimality Equation into update rule

  ![image-20260116002618607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601160026712.png)



Summary

![image-20260116004040385](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202601160040488.png)



If state set is very large, DP may be very slow. Asychronous DP do not sweep over state space

- In-place dynamic programming

  ![image-20260205183152497](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051831564.png)

- Prioritized sweeping: update some state more frequently

  ![image-20260205183319280](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051833310.png)

- Real-time dynamic programming: do not wait until the episode ends

  ![image-20260205183704901](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051837934.png)

Replay Buffer:

![image-20260205183950268](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051839340.png)



# LEC5

policy evaluation / value/policy iteration: known MDP

Model-free prediction: Estimate value function of an unknown MDP

Model-free control: Optimize value function of an unknown MDP



<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051845278.png" alt="image-20260205184537201" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051845436.png" alt="image-20260205184552375" style="zoom:25%;" />



MC does not assume state is Markov, only applied to episodic MDPs (each episode terminates)

Monte-Carlo Policy Evaluation

![image-20260205185032821](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051850885.png)



<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051851075.png" alt="image-20260205185144013" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051851096.png" alt="image-20260205185156030" style="zoom:25%;" />

bootstrapping: updating an estimate using other estimates (instead of waiting for the final outcome).

![image-20260205190955175](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051909216.png)

![image-20260205191205863](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051912895.png)

![image-20260205191323393](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051913481.png)

![image-20260205191444659](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051914685.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051916984.png" alt="image-20260205191616948" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051916537.png" alt="image-20260205191629469" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051917167.png" alt="image-20260205191641924" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202602051917160.png" alt="image-20260205191748060" style="zoom:25%;" />

Model-free control: Generalized Policy Iteration (GPI) with MC or TD in the loop





