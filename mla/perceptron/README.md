<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
### 感知机
损失函数: 误分点到超平面的总距离, M 为误分点集合
$$
\begin{align}
dist &= \frac{1}{\|w\|} | w \bullet x_i + b | \\
L(w, b) &= -\sum_{x_i \in M} y_i(w \bullet x_i + b)
\end{align}
$$
损失函数的梯度:
$$
\begin{align}
\nabla_w{L(w, b)} &= -\sum_{x_i \in M} y_ix_i \\
\nabla_b{L(w, b)} &= -\sum_{x_i \in M} y_i
\end{align}
$$

随机梯度下降法(stochastic gradient descent)
随机取误分点 $ (x_i, y_i) $, 对 $ w, b $ 进行更新
$$
\begin{align}
w &= w + \eta y_i x_i \\
b &= b + \eta y_i
\end{align}
$$

## 感知机的对偶形式

对于 $ w, b $ 从上面可以看出最后学习到的形式如下
$$
\begin{align}
w &= \sum_{i=1}^{N} \alpha_i y_i x_i \\
b &= \sum_{i=1}^{N} \alpha_i y_i
\end{align}
$$
其中 $ \alpha_i = n_i\eta $, $ n_i $, __表示第 i 个点被当作误分点更新了几次__

