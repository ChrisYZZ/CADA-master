# CADA-master
Stochastic gradient descent (SGD) has taken the stage as the primary workhorse for large-scale machine learning. It is often used with its adaptive variants such as AdaGrad, Adam, and AMSGrad. This paper proposes an adaptive stochastic gradient descent method for distributed machine learning, which can be viewed as the communication-adaptive counterpart of the celebrated Adam method - justifying its name CADA. The key components of CADA are a set of new rules tailored for adaptive stochastic gradients that can be implemented to save communication upload. The new algorithms adaptively reuse the stale Adam gradients, thus saving communication, and still have convergence rates comparable to original Adam. In numerical experiments, CADA achieves impressive empirical performance in terms of total communication round reduction.


Reference:@misc{chen2020cada,
      title={CADA: Communication-Adaptive Distributed Adam}, 
      author={Tianyi Chen and Ziye Guo and Yuejiao Sun and Wotao Yin},
      year={2020},
      eprint={2012.15469},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
