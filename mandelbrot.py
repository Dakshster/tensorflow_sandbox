import tensorflow as tf
import numpy as np
import PIL.Image


def DisplayFractal(a, fmt='jpeg'):
  """Display an array of iteration counts as a colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))

  with open('/tmp/mandelbrot.jpg', 'w') as f:
    PIL.Image.fromarray(a).save(f, fmt)


# main
if __name__ == "__main__":
  sess = tf.InteractiveSession()

  # 描画する対象グリッドを生成
  Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
  Z = X+1j*Y

  xs = tf.constant(Z.astype("complex64"))
  zs = tf.Variable(xs)
  # xs と同じ型の零行列を生成
  ns = tf.Variable(tf.zeros_like(xs, "float32"))

  tf.initialize_all_variables().run()

  # z = z^2 + x を新しく計算
  zs_ = zs*zs + xs

  # 発散しないかチェック（マンデルブロー集合の定義）
  not_diverged = tf.complex_abs(zs_) < 4

  # Operation to update the zs and the iteration count.
  #
  # Note: We keep computing zs after they diverge! This
  #       is very wasteful! There are better, if a little
  #       less simple, ways to do this.
  #
  step = tf.group(
    zs.assign(zs_),
    # boolean を float にキャスト(true -> 1, false -> 0)
    # ns が各グリッドに置ける数列のカウントになっている（発散条件を満たす場合にのみ加算）
    ns.assign_add(tf.cast(not_diverged, "float32"))
  )

  # 数列を第200位まで計算
  for i in range(200):
    step.run()

  DisplayFractal(ns.eval())

