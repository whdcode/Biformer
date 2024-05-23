from matplotlib import pyplot as plt
from scipy.ndimage import rotate, zoom
import numpy as np
import random
from skimage.exposure.exposure import adjust_gamma, equalize_hist
from torchvision.transforms import transforms

class DataAug():
    def __init__(self):
        super().__init__()


class xy_rotate(DataAug):
    def __init__(self, mn=0, mx=0, axis=(0, 1), rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.axis = axis
        self.rate = rate

    def __call__(self, data):
        # axes = [(1, 0), (2, 1), (2, 0)]
        # axis = axes[np.random.randint(len(axes))]
        if random.random() >= self.rate:
            data = rotate(data, angle=np.random.randint(self.mn, self.mx), axes=self.axis, reshape=False)
            data[data < 0.] = 0.
        return data


class xyz_rotate(DataAug):
    def __init__(self, mn=0, mx=0, rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.rate = rate
        self.axis = [(0, 1), (0, 2), (1, 2)]

    def __call__(self, data):
        if random.random() >= (1 - self.rate):
            axis = self.axis[np.random.randint(3)]
            data = rotate(data, angle=np.random.randint(self.mn, self.mx), axes=axis, reshape=False)
            data[data < 0.] = 0.
        return data


class gamma_adjust(DataAug):
    def __init__(self, mn=0, mx=0, rate=0.5):
        super().__init__()
        self.mn = mn
        self.mx = mx
        self.rate = rate

    def __call__(self, data):
        if random.random() >= self.rate:
            data = adjust_gamma(data, gamma=round(random.uniform(self.mn, self.mx), 1))
        return data


class contrast(DataAug):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return np.clip(random.uniform(0.8, 1.3) * data, -1, 1)


class equa_hist(DataAug):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return equalize_hist(data)


class flip(DataAug):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate

    def __call__(self, data):
        if random.random() >= (1 - self.rate):
            return np.fliplr(data)
        else:
            return data


class sample(DataAug):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return data[0:-1:2, 0:-1:2, 0:-1:2]


class mask(DataAug):
    def __init__(self, rate=0.5, mask_nums=1, size=None, intersect=True):
        super().__init__()
        if size is None:
            size = [5, 10, 5]
        self.size = size
        self.mask_nums = mask_nums
        self.intersect = intersect
        self.rate = rate

    def __call__(self, data):
        # dhw
        x, y, z = data.shape
        d, h, w = self.size
        i_list = []
        m = 0
        if random.random() >= (1 - self.rate):
            while m < self.mask_nums:
                xi = random.randint(0, x - d)
                yi = random.randint(0, y - h)
                zi = random.randint(0, z - w)
                if not self.intersect:
                    flag = False
                    for _, (ai, bi, ci) in enumerate(i_list):
                        if abs(ai - xi) < d and abs(bi - yi) < h and abs(ci - zi) < w:
                            flag = True
                            break
                    if flag:
                        continue
                m += 1
                i_list.append((xi, yi, zi))
                mask = np.ones(data.shape)
                mask[xi:xi + d, yi:yi + h, zi:zi + w] = 0
                data = data * mask
            return data
        else:
            return data


class noisy(object):
    def __init__(self, radio, probability):
        self.radio = radio
        self.probability = probability

    def __call__(self, data):
        if np.random.uniform(0, 1) > self.probability:
            return data

        l, w, h = data.shape
        num = int(l * w * h * self.radio)
        for _ in range(num):
            x = np.random.randint(0, l)
            y = np.random.randint(0, w)
            z = np.random.randint(0, h)
            noise = np.random.uniform(0, self.radio)
            data[x, y, z] = data[x, y, z] + noise
        return data

class RandomCrop3D(object):
    def __init__(self, output_size, padding=False, radio=0.5):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        self.padding = padding
        self.radio = radio

    def __call__(self, data):
        if random.uniform(0, 1) > (1 - self.radio):
            x, y, z = data.shape
            new_x, new_y, new_z = self.output_size

            if x == new_x and y == new_y and z == new_z:
                return data

            x_margin = x - new_x
            y_margin = y - new_y
            z_margin = z - new_z

            x_min = random.randint(0, x_margin) if x_margin > 0 else 0
            y_min = random.randint(0, y_margin) if y_margin > 0 else 0
            z_min = random.randint(0, z_margin) if z_margin > 0 else 0

            x_max = x_min + new_x
            y_max = y_min + new_y
            z_max = z_min + new_z

            cropped = data[x_min:x_max, y_min:y_max, z_min:z_max]
            target_shape = (112, 112, 112)
            zoom_factor = tuple(np.array(target_shape) / np.array(cropped.shape))
            cropped = zoom(cropped, zoom_factor, order=1)

            if self.padding:
                padding_x = (0, 0)
                padding_y = (0, 0)
                padding_z = (0, 0)

                if x_margin < 0:
                    padding_x = (-x_margin // 2, (-x_margin + 1) // 2)
                if y_margin < 0:
                    padding_y = (-y_margin // 2, (-y_margin + 1) // 2)
                if z_margin < 0:
                    padding_z = (-z_margin // 2, (-z_margin + 1) // 2)

                padded = np.pad(cropped, ((0, 0), padding_x, padding_y, padding_z), mode='constant', constant_values=0)
                return padded
            else:
                return cropped
        return data


if __name__ == '__main__':
    smri_data_path = "E:\\datasets\\using_datasets\\112all_ad&hc_npy_data\\ad\\003_S_4152_1.npy"
    smri_data = np.load(smri_data_path, allow_pickle=True)[0][0]
    smri_shape = smri_data.shape
    tran = noisy(1, 0)
    # tran1 = xy_rotate(0, 60, rate=0.5)
    # tran2 = xyz_rotate(-10, 10, rate=0.1)
    # tran3 = gamma_adjust(0.5, 0.7, rate=0.1)
    # tran4 = contrast()
    # tran5 = sample()
    # tran6 = mask(10, intersect=False)
    # tran = equa_hist()
    # tran8 = flip(rate=0.1)
    #
    # transform_byme = transforms.Compose([
    #     xyz_rotate(-10, 10, rate=0.5),
    #     flip(rate=0.5),
    #     mask(rate=0.5, mask_nums=2, intersect=False),
    #     contrast(),
    # ])

    data_sug = tran(smri_data)
    # data_sug1 = tran1(smri_data)
    # data_sug2 = tran2(smri_data)
    # data_sug3 = tran3(smri_data)
    # data_sug4 = tran4(smri_data)
    # data_sug5 = tran5(smri_data)
    # data_sug5_shape = data_sug5.shape
    # data_sug6 = tran6(smri_data)
    # data_sug7 = tran7(smri_data)
    # data_sug8 = tran8(smri_data)
    #
    # data_sugji = transform_byme(smri_data)
    # data_sug2 = skimage.transform.rotate(smri_data, 60)  # 旋转60度，不改变大小
    # data_sug3 = exposure.exposure.adjust_gamma(smri_data, gamma=0.7)  # 变亮
    # data_sug4 = transform_byme(smri_data)

    # # 可视化sMRI数据的扩增前后
    # fig1, axs1 = plt.subplots(figsize=(10, 10))
    # # axs1.imshow(data_sugji[:, :, smri_shape[2] // 2], cmap='gray')
    # axs1.set_title('compose')

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(smri_data[:, smri_shape[2] // 2, :], cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(data_sug[:, smri_shape[2] // 2, :], cmap='gray')
    axs[1].set_title('noisy')
    # axs[0][1].imshow(data_sug2[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[0][1].set_title('rotate_xyz')
    # axs[0][2].imshow(data_sug3[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[0][2].set_title('adjust_gamma')
    # axs[1][0].imshow(data_sug4[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[1][0].set_title('contrast')
    # axs[1][1].imshow(data_sug5[:, :, data_sug5_shape[2] // 2], cmap='gray')
    # axs[1][1].set_title('sample')
    # axs[1][2].imshow(data_sug6[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[1][2].set_title('mask')
    # axs[1][3].imshow(data_sug7[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[1][3].set_title('equa_hist')
    # axs[2][0].imshow(data_sug8[:, :, smri_shape[2] // 2], cmap='gray')
    # axs[2][0].set_title('flip')
    plt.show()




class randomflip180(object):
    def __call__(self, data):
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
            data = np.flip(data, axis=2)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
            data = np.flip(data, axis=0)
        return data.copy()


class randomflip(object):
    def __call__(self, data):
        # print(data.shape)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
        return data.copy()


class noisy(object):
    def __init__(self, radio):
        self.radio = radio

    def __call__(self, data):
        _, l, w, h = data.shape
        num = int(l * w * h * self.radio)
        for _ in range(num):
            x = np.random.randint(0, l)
            y = np.random.randint(0, w)
            z = np.random.randint(0, h)
            data[0,x, y, z] = data[0,x, y, z] + np.random.uniform(0, self.radio)
        return data


transform_data = transforms.Compose([
    randomflip(),
    randomflip180(),
    noisy(0.001)])
