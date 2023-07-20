
import mglearn
import matplotlib.pyplot as plt

#generating wave

X, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Trait")
plt.ylabel("Goal")

plt.show()