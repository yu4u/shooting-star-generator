from math import cos, sin, pi
import numpy as np
import cv2


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_affine_mat(theta, length):
    t1 = np.array([[1, 0, -length / 2],
                   [0, 1, -5],
                   [0, 0, 1]], dtype=np.float)
    r = np.array([[cos(theta), -sin(theta), 0],
                  [sin(theta), cos(theta), 0],
                  [0, 0, 1]], dtype=np.float)
    t2 = np.array([[1, 0, 130],
                   [0, 1, 60],
                   [0, 0, 1]], dtype=np.float)
    mat = np.dot(np.dot(t2, r), t1)

    return mat[:2, :]


def generate_shooting_star(duration, length=200, brightness=255, noise_level=50, head_w=3, head_h=5, tail_w=2,
                           tail_ratio=0.75):
    t_dummy = np.linspace(-4, 4, duration + 1)
    t = [int(sigmoid(t_dummy[i]) * length * 2) for i in range(duration + 1)]
    outputs = []
    trajectories = []

    for i in range(duration):
        img = np.zeros((20, length * 2, 1), dtype=np.uint8)
        y = 10

        # tail
        cv2.line(img, (t[i + 1], y), (t[i + 1] + int((t[i] - t[i + 1]) * tail_ratio), y), brightness // 2, tail_w)

        # head
        cv2.line(img, (t[i + 1] - head_h // 2, y), (t[i + 1] + head_h // 2, y), brightness, head_w)

        trajectory = np.array([t[i + 1] / 2, y / 2, 1], dtype=np.float32)
        trajectories.append(trajectory)

        noise = (np.random.randn(20, length * 2, 1) * noise_level)
        np.clip(noise, 0, 255, noise)
        cv2.subtract(img, noise.astype(np.uint8), img)
        outputs.append(cv2.resize(img, (length, 10)))

    return outputs, trajectories


def main():
    while True:
        duration = 4 + np.random.randint(20)
        length = 20 + np.random.randint(80)
        brightness = 128 + np.random.randint(128)
        noise_level = 25 + np.random.randint(80)
        head_w = 3 + np.random.randint(4)
        head_h = 4 + np.random.randint(4)
        tail_w = min(1 + np.random.randint(3), head_w)
        tail_ratio = np.random.rand()
        r = 2 * pi * np.random.rand()

        outputs, trajectories = generate_shooting_star(duration, length, brightness, noise_level, head_w, head_h,
                                                       tail_w, tail_ratio)
        affine_mat = get_affine_mat(r, length)

        for output, trajectory in zip(outputs, trajectories):
            dst = cv2.warpAffine(output, affine_mat, (320, 180))
            x, y = affine_mat @ trajectory
            cv2.rectangle(dst, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), (255, 255, 255), 1)
            cv2.imshow("shooting star", dst)
            key = cv2.waitKey(50)

            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
