
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import cv2
import pandas as pd
import random
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    A.Normalize(mean=[0.], std=[1.]),

    ToTensorV2()

])

valid_aug = A.Compose([
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])



class Feeder(Dataset):
    def __init__(self, df_path,fc_path,vf_path,vh_path,fm_path,debug=False,aug='', mode='train'):
        with open(df_path, 'rb') as f:
            self.df = pickle.load(f)
        with open(fc_path, 'rb') as f:
            feature_cols = pickle.load(f)
        with open(vf_path, 'rb') as f:
            self.video2frames = pickle.load(f)
        with open(vh_path, 'rb') as f:
            self.video2helmets = pickle.load(f)
        self.frame = self.df.frame.values
        self.feature = self.df[feature_cols].fillna(-1).values
        self.players = self.df[['nfl_player_id_1', 'nfl_player_id_2']].values
        self.game_play = self.df.game_play.values
        self.aug = aug
        self.mode = mode
        self.debug=debug
        self.randomErasing=RandomErasing()
        if mode=='train':
            self.aug=train_aug

        else:
            self.aug=valid_aug

    def __len__(self):
        return len(self.df)

    # @lru_cache(1024)
    # def read_img(self, path):
    #     return cv2.imread(path, 0)

    def __getitem__(self, idx):
        window = 24
        frame = self.frame[idx]

        if self.mode == 'train':
            frame = frame + random.randint(-6, 6)

        players = []
        for p in self.players[idx]:
            if p == 'G':
                players.append(p)
            else:
                players.append(int(p))

        imgs = []
        for view in ['Endzone', 'Sideline']:
            video = self.game_play[idx] + f'_{view}.mp4'

            tmp = self.video2helmets[video]
            #             tmp = tmp.query('@frame-@window<=frame<=@frame+@window')
            tmp[tmp['frame'].between(frame - window, frame + window)]
            tmp = tmp[tmp.nfl_player_id.isin(players)]  # .sort_values(['nfl_player_id', 'frame'])
            tmp_frames = tmp.frame.values
            tmp = tmp.groupby('frame')[['left', 'width', 'top', 'height']].mean()
            # 0.002s

            bboxes = []
            for f in range(frame - window, frame + window + 1, 1):
                if f in tmp_frames:
                    x, w, y, h = tmp.loc[f][['left', 'width', 'top', 'height']]
                    bboxes.append([x, w, y, h])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])
            bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
            bboxes = bboxes[::4]

            if bboxes.sum() > 0:
                flag = 1
            else:
                flag = 0
            # 0.03s

            for i, f in enumerate(range(frame - window, frame + window + 1, 4)):
                img_new = np.zeros((256, 256), dtype=np.float32)

                if flag == 1 and f <= self.video2frames[video]:
                    img = cv2.imread(f'./frames/{video}_{f:04d}.jpg', 0)

                    x, w, y, h = bboxes[i]
                    # print(bboxes[i])
                    # print(len(img))
                    # print(int(y + h / 2) - 128,int(y + h / 2) + 128,
                    #       int(x + w / 2) - 128,int(x + w / 2) + 128)
                    img = img[int(y + h / 2) - 128:int(y + h / 2) + 128,
                          int(x + w / 2) - 128:int(x + w / 2) + 128].copy()
                    img_new[:img.shape[0], :img.shape[1]] = img

                imgs.append(img_new)
        # 0.06s

        feature = np.float32(self.feature[idx])

        img = np.array(imgs).transpose(1, 2, 0)
        # print(type(img))
        # print(img.shape)
        img = self.aug(image=img)["image"]
        if self.mode=='train':
            img=self.randomErasing(img)
        label = np.float32(self.df.contact.values[idx])

        return img, feature, label,idx

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__=='__main__':
    feeder=Feeder('./train_filtered.pkl','./train_featurecol.pkl','./train_video2frames.pkl','./train_video2helmets.pkl'
                                    ,'./frames',aug=valid_aug)
    print(len(feeder))
    for i in range(0,len(feeder)):
        img, feature, label,idx =feeder[i]
        if label>0.0:
            print(label)



