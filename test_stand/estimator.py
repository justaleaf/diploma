import numpy as np
import torch
from pretrained_mods import xception
import os
import timm
import torch.nn as nn
import test
import pandas as pd
from tqdm import tqdm
import shutil
from pretrained_mods import efficientnetb1lstm
from pretrained_mods import mesonet
from pretrained_mods import resnetlstm

class Estimator():
    def __init__():
        pass
    
    @classmethod
    def detect_single(cls, video_path=None, label=None, method="xception_uadfv"):
        """Perform deepfake detection on a single video with a chosen method."""
        # prepare the method of choice
        sequence_model = False
        model, img_size, normalization = prepare_method(method=method, dataset=None, mode='test')
        used = method
        # if method == "xception_uadfv":
        #     model, img_size, normalization = prepare_method(
        #         method=method, dataset=None, mode='test')
        #     used = "Xception_UADFV"
        # elif method == "xception_celebdf":
        #     model, img_size, normalization = prepare_method(
        #         method=method, dataset=None, mode='test')
        #     used = "Xception_CELEB-DF"
        # elif method == "xception_dfdc":
        #     model, img_size, normalization = prepare_method(
        #         method=method, dataset=None, mode='test')
        #     used = "Xception_DFDC"

        if video_path:
                
            data = [[1, video_path]]
            df = pd.DataFrame(data, columns=['label', 'video'])
            loss = test.inference(
                    model, df, img_size, normalization, dataset=None, method=method, face_margin=0.3, sequence_model=sequence_model, num_frames=20, single=True)
            if round(loss) == 1:
                # Deepfake
                if label == 'is_fake':
                    result = 'Это видео успешно определено как дипфейк'
                else:
                    result = 'Видео ошибочно определено как дипфейк'
                print("Deepfake detected.")
                return used, result
            else:
                # Real
                if label == 'is_fake':
                    result = 'Это видео ошибочно определено как оригинальное'
                else:
                    result = 'Видео успешно определено как оригинальное'
                print("This is a real video.")
                return used, result


    @classmethod
    def benchmark(self, dataset=None, method=None, data_path=None, seed=12):
        reproducibility_seed(seed)
        # is_method_available
        self.dataset = dataset
        self.method = method
        self.data_path = data_path
        face_margin = 0.3
        if self.dataset == 'uadfv':
            num_frames = 20
            # setup the dataset folders
            setup_uadfv_benchmark(self.data_path, self.method)
        elif self.dataset == 'celebdf':
            num_frames = 20
            setup_celebdf_benchmark(self.data_path, self.method)
        elif self.dataset == 'uadfv':
            num_frames = 20
            setup_uadfv_benchmark(self.data_path, self.method)
        elif self.dataset == 'dftimit_hq':
            num_frames = 20
            setup_dftimit_hq_benchmark(self.data_path, self.method)
        elif self.dataset == 'dftimit_lq':
            num_frames = 20
            setup_dftimit_lq_benchmark(self.data_path, self.method)
        elif self.dataset == 'generateddf':
            num_frames = 20
        elif self.dataset == 'dfdc':
            # benchmark on only 5 frames per video, because of dataset size
            num_frames = 5
        else:
            raise ValueError(f"{self.dataset} does not exist.")

        df = label_data(dataset_path=self.data_path,
                            dataset=self.dataset, test_data=True)

        model, img_size, normalization = prepare_method(
            method=self.method, dataset=self.dataset, mode='test')
        # if self.method == "xception_uadfv" or self.method == 'xception_celebdf' or self.method == 'xception_dfdc':
        #     model, img_size, normalization = prepare_method(
        #         method=self.method, dataset=self.dataset, mode='test')

        # elif self.method == 'mesonet_uadfv' or self.method == 'mesonet_celebdf' or self.method == 'mesonet_dftimit_hq' or self.method == 'mesonet_dftimit_lq' or self.method == 'mesonet_dfdc':
        #     model, img_size, normalization = prepare_method(
        #         method=self.method, dataset=self.dataset, mode='test')

        print(f"Detecting deepfakes with \033[1m{self.method}\033[0m ...")
        res = test.inference(
            model, df, img_size, normalization, dataset=self.dataset, method=self.method, face_margin=face_margin, num_frames=num_frames)
        return res
    
def reproducibility_seed(seed):
    print(f"The random seed is set to {seed}.")
    # set numpy random seed
    np.random.seed(seed)
    # set pytorch random seed for cpu and gpu
    torch.manual_seed(seed)
    # get deterministic behavior
    torch.backends.cudnn.deterministic = True

def is_method_available(method):
    if method not in ['xception_uadfv', 'xception_celebdf', 'xception_dfdc', 'mesonet', 'mesonet_uadfv', 'mesonet_celebdf', 'mesonet_dftimit_hq', 'mesonet_dftimit_lq', 'mesonet_dfdc', 'resnet_lstm_uadfv', 'resnet_lstm_celebdf', 'resnet_lstm_dftimit_hq', 'resnet_lstm_dftimit_lq', 'resnet_lstm_dfdc', 'efficientnetb1_lstm_uadfv', 'efficientnetb1_lstm_celebdf', 'efficientnetb1_lstm_dfdc', 'efficientnetb1_lstm_dftimit_lq', 'efficientnetb1_lstm_dftimit_hq', 'efficientnetb7_uadfv', 'efficientnetb7_celebdf', 'efficientnetb7_dfdc', 'efficientnetb7_dftimit_lq', 'efficientnetb7_dftimit_hq']:
        raise ValueError("Method is not available for benchmarking.")
    else:
        pass

def prepare_method(method, dataset, mode='train'):
    """Prepares the method that will be used for training or benchmarking."""
    if method == 'xception' or method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dfdc':
        img_size = 299
        normalization = 'xception'
        if mode == 'test':
            model = xception.imagenet_pretrained_xception()
            # load the xception model that was pretrained on the respective datasets training data
            if method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dfdc':
                model_params = torch.load(
                    os.getcwd() + f'/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'mesonet' or method == 'mesonet_uadfv' or method == 'mesonet_celebdf' or method == 'mesonet_dftimit_hq' or method == 'mesonet_dftimit_lq' or method == 'mesonet_dfdc':
        # 256 image size as proposed in the MesoNet paper (https://arxiv.org/abs/1809.00888)
        img_size = 256
        # use [0.5,0.5,0.5] normalization scheme, because no imagenet pretraining
        normalization = 'xception'
        if mode == 'test':
            if method == 'mesonet_uadfv' or method == 'mesonet_celebdf' or method == 'mesonet_dftimit_hq' or method == 'mesonet_dftimit_lq' or method == 'mesonet_dfdc':
                # load MesoInception4 model
                model = mesonet.MesoInception4()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'resnet_lstm' or method == 'resnet_lstm_uadfv' or method == 'resnet_lstm_celebdf' or method == 'resnet_lstm_dftimit_hq' or method == 'resnet_lstm_dftimit_lq' or method == 'resnet_lstm_dfdc':
        img_size = 224
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'resnet_lstm_uadfv' or method == 'resnet_lstm_celebdf' or method == 'resnet_lstm_dftimit_hq' or method == 'resnet_lstm_dftimit_lq' or method == 'resnet_lstm_dfdc':
                # load MesoInception4 model
                model = resnetlstm.ResNetLSTM()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb1_lstm' or method == 'efficientnetb1_lstm_uadfv' or method == 'efficientnetb1_lstm_celebdf' or method == 'efficientnetb1_lstm_dftimit_hq' or method == 'efficientnetb1_lstm_dftimit_lq' or method == 'efficientnetb1_lstm_dfdc':
        img_size = 240
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb1_lstm_uadfv' or method == 'efficientnetb1_lstm_celebdf' or method == 'efficientnetb1_lstm_dftimit_hq' or method == 'efficientnetb1_lstm_dftimit_lq' or method == 'efficientnetb1_lstm_dfdc':
                # load EfficientNetB1+LSTM
                model = efficientnetb1lstm.EfficientNetB1LSTM()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb7' or method == 'efficientnetb7_uadfv' or method == 'efficientnetb7_celebdf' or method == 'efficientnetb7_dftimit_hq' or method == 'efficientnetb7_dftimit_lq' or method == 'efficientnetb7_dfdc':
        # 380 image size as introduced here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        img_size = 380
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb7_uadfv' or method == 'efficientnetb7_celebdf' or method == 'efficientnetb7_dftimit_hq' or method == 'efficientnetb7_dftimit_lq' or method == 'efficientnetb7_dfdc':
                # successfully used by https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721 (noisy student weights)
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                model.classifier = nn.Linear(2560, 1)
                # load the efficientnet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    else:
        raise ValueError(
            f"{method} is not available. Please use one of the available methods.")


def label_data(dataset_path=None, dataset='uadfv', method='xception', face_crops=False, test_data=False, fulltrain=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    """
    # structure data from folder in data frame for loading
    if dataset_path is None:
        raise ValueError("Please specify a dataset path.")
    if not test_data:
        if dataset == 'uadfv':
            # prepare training data
            video_path_real = os.path.join(dataset_path + "/real/")
            video_path_fake = os.path.join(dataset_path + "/fake/")
            # if no face crops available yet, read csv for videos
            if not face_crops:
                # read csv for videos
                test_dat = pd.read_csv(os.getcwd(
                ) + "/uadfv_test.csv", names=['video'], header=None)
                test_list = test_dat['video'].tolist()

                full_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in videos:
                        # label 0 for real video
                        full_list.append(video)

                for _, _, videos in os.walk(video_path_fake):
                    for video in videos:
                        # label 1 for deepfake video
                        full_list.append(video)

                # training data (not used for testing)
                new_list = [
                    entry for entry in full_list if entry not in test_list]

                # add labels to videos
                data_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                # put data into dataframe
                df = pd.DataFrame(data=data_list)

            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_fake) + str(row['original'])

                else:
                    # if face crops available go to path with face crops
                    # add labels to videos

                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
        elif dataset == 'celebdf':
            # prepare celebdf training data by
            # reading in the testing data first
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            # structure data from folder in data frame for loading
            if not face_crops:
                video_path_real = os.path.join(dataset_path + "/Celeb-real/")
                video_path_fake = os.path.join(
                    dataset_path + "/Celeb-synthesis/")
                real_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # label 0 for real image
                        real_list.append({'label': 0, 'video': video})

                fake_list = []
                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # label 1 for deepfake image
                        fake_list.append({'label': 1, 'video': video})

                # put data into dataframe
                df_real = pd.DataFrame(data=real_list)
                df_fake = pd.DataFrame(data=fake_list)
                # add real and fake path to video file name
                df_real['video_name'] = df_real['video']
                df_fake['video_name'] = df_fake['video']
                df_real['video'] = video_path_real + df_real['video']
                df_fake['video'] = video_path_fake + df_fake['video']
                # put testing vids in list
                testing_vids = list(df_test['video'])
                # remove testing videos from training videos
                df_real = df_real[~df_real['video'].isin(testing_vids)]
                df_fake = df_fake[~df_fake['video'].isin(testing_vids)]
                # undersampling strategy to ensure class balance of 50/50
                df_fake_sample = df_fake.sample(
                    n=len(df_real), random_state=24).reset_index(drop=True)
                # concatenate both dataframes to get full training data (964 training videos with 50/50 class balance)
                df = pd.concat([df_real, df_fake_sample], ignore_index=True)
            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available go to path with face crops
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_crops_real + video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_crops_fake + video})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")

            # prepare dfdc training data
            # structure data from folder in data frame for loading
            all_meta_train, all_meta_test, full_margin_aug_val = dfdc_metadata_setup()
            if not face_crops:
                # read in the reals
                if fulltrain:
                    all_meta_train['videoname'] = all_meta_train['video']
                    all_meta_train['video'] = dataset_path + \
                        '/train/' + all_meta_train['videoname']
                    all_meta_train = all_meta_train.sort_values(
                        'folder').reset_index(drop=True)
                    df = all_meta_train[all_meta_train['folder'] > 35]
                    print(df)
                else:
                    print("Validation DFDC data.")
                    full_margin_aug_val['videoname'] = full_margin_aug_val['video']
                    full_margin_aug_val['video'] = dataset_path + \
                        '/train/' + full_margin_aug_val['videoname']
                    df = full_margin_aug_val
            else:
                # if face crops available
                # if sequence and if face crops available go to path with face crops and prepare sequence data
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video':  os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available and not a sequence model go to path with face crops
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    print(df)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")
        elif dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
            # prepare dftimit_lq training data by
            # structure data from folder in data frame for loading
            test_df_real = pd.read_csv(
                os.getcwd() + "/dftimit_test_real.csv")

            test_df_real['testlist'] = test_df_real['path'].str[:5] + \
                test_df_real['videoname'].apply(str)
            testing_vids_real = test_df_real['testlist'].tolist()
            test_df_fake = pd.read_csv(
                os.getcwd() + "/dftimit_test_fake.csv")
            test_df_fake['testlist'] = test_df_fake['videoname'].apply(str)
            testing_vids_fake = test_df_fake['testlist'].tolist()
            # join test vids in list
            test_vids = testing_vids_real + testing_vids_fake
            if not face_crops:
                # read in the reals
                reals = pd.read_csv(
                    os.getcwd() + "/dftimit_reals.csv")
                reals['testlist'] = reals['path'].str[:5] + \
                    reals['videoname'].apply(str)
                reals['path'] = reals['path'] + reals['videofolder'] + \
                    '/' + reals['videoname'].apply(str) + '.avi'
                # remove testing videos from training videos
                reals = reals[~reals['testlist'].isin(test_vids)]
                reals['videoname'] = reals['videoname'].apply(str) + '.avi'
                del reals['videofolder']
                reals['label'] = 0
                reals['path'] = dataset_path + '/dftimitreal/' + reals['path']
                if dataset == 'dftimit_hq':
                    fake_path = os.path.join(dataset_path, 'higher_quality')
                elif dataset == 'dftimit_lq':
                    fake_path = os.path.join(dataset_path, 'lower_quality')
                # get list of fakes
                data_list = []
                data_list_name = []
                for path, dirs, files in os.walk(fake_path):
                    for filename in files:
                        if filename.endswith(".avi"):
                            data_list.append(os.path.join(path, filename))
                            data_list_name.append(filename)
                fakes = pd.DataFrame(list(zip(data_list, data_list_name)), columns=[
                                     'path', 'videoname'])

                fakes['testlist'] = fakes['videoname'].str[:-4]
                fakes = fakes[~fakes['testlist'].isin(test_vids)]
                fakes['label'] = 1
                # put fakes and reals in one dataframe
                df = pd.concat([reals, fakes])
                df = df.rename(columns={"path": "video"})
        elif dataset == 'dfdc':
            # prepare dfdc training data
            # structure data from folder in data frame for loading
            all_meta_train, all_meta_test, full_margin_aug_val = dfdc_metadata_setup()
            if not face_crops:
                # read in the reals
                if fulltrain:
                    all_meta_train['videoname'] = all_meta_train['video']
                    all_meta_train['video'] = dataset_path + \
                        '/train/' + all_meta_train['videoname']
                    all_meta_train = all_meta_train.sort_values(
                        'folder').reset_index(drop=True)
                    df = all_meta_train[all_meta_train['folder'] > 35]
                    print(df)
                else:
                    print("Validation DFDC data.")
                    full_margin_aug_val['videoname'] = full_margin_aug_val['video']
                    full_margin_aug_val['video'] = dataset_path + \
                        '/train/' + full_margin_aug_val['videoname']
                    df = full_margin_aug_val
            else:
                # if face crops available
                # if sequence and if face crops available go to path with face crops and prepare sequence data
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video':  os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available and not a sequence model go to path with face crops
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    print(df)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")
        elif dataset == 'generateddf':
            # prepare training data
            video_path_real = os.path.join(dataset_path + "/real/")
            video_path_fake = os.path.join(dataset_path + "/fake/")
            # if no face crops available yet, read csv for videos
            if not face_crops:
                # read csv for videos
                test_dat = pd.read_csv(os.getcwd(
                ) + "/uadfv_test.csv", names=['video'], header=None)
                test_list = test_dat['video'].tolist()

                full_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in videos:
                        # label 0 for real video
                        full_list.append(video)

                for _, _, videos in os.walk(video_path_fake):
                    for video in videos:
                        # label 1 for deepfake video
                        full_list.append(video)

                # training data (not used for testing)
                new_list = [
                    entry for entry in full_list if entry not in test_list]

                # add labels to videos
                data_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                # put data into dataframe
                df = pd.DataFrame(data=data_list)

            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_fake) + str(row['original'])

                else:
                    # if face crops available go to path with face crops
                    # add labels to videos

                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
    else:
        # prepare test data
        if dataset == 'uadfv':
            video_path_test_real = os.path.join(dataset_path + "/test/real/")
            video_path_test_fake = os.path.join(dataset_path + "/test/fake/")
            data_list = []
            for _, _, videos in os.walk(video_path_test_real):
                for video in tqdm(videos):
                    # append test video
                    data_list.append(
                        {'label': 0, 'video': video_path_test_real + video})

            for _, _, videos in os.walk(video_path_test_fake):
                for video in tqdm(videos):
                    # label 1 for deepfake image
                    data_list.append(
                        {'label': 1, 'video': video_path_test_fake + video})
            return pd.DataFrame(data_list)
        elif dataset == 'celebdf':
            # reading in the celebdf testing data
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            print(f"{len(df_test)} test videos.")
            return df_test
        elif dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
            test_df_real = pd.read_csv(
                os.getcwd() + "/dftimit_test_real.csv")
            test_df_real['testlist'] = test_df_real['path'].str[:5] + \
                test_df_real['videoname'].apply(str)
            testing_vids_real = test_df_real['testlist'].tolist()
            test_df_fake = pd.read_csv(
                os.getcwd() + "/dftimit_test_fake.csv")
            test_df_fake['testlist'] = test_df_fake['videoname'].apply(str)
            testing_vids_fake = test_df_fake['testlist'].tolist()
            # join test vids in list
            test_vids = testing_vids_real + testing_vids_fake
            # read in the reals
            reals = pd.read_csv(
                os.getcwd() + "/dftimit_reals.csv")
            reals['testlist'] = reals['path'].str[:5] + \
                reals['videoname'].apply(str)
            reals['path'] = reals['path'] + reals['videofolder'] + \
                '/' + reals['videoname'].apply(str) + '.avi'
            # remove testing videos from training videos
            reals = reals[reals['testlist'].isin(test_vids)]
            reals['videoname'] = reals['videoname'].apply(str) + '.avi'
            del reals['videofolder']
            reals['label'] = 0
            reals['path'] = dataset_path + '/dftimitreal/' + reals['path']
            if dataset == 'dftimit_hq':
                fake_path = os.path.join(dataset_path, 'higher_quality')
            elif dataset == 'dftimit_lq':
                fake_path = os.path.join(dataset_path, 'lower_quality')
            # get list of fakes
            data_list = []
            data_list_name = []
            for path, dirs, files in os.walk(fake_path):
                for filename in files:
                    if filename.endswith(".avi"):
                        data_list.append(os.path.join(path, filename))
                        data_list_name.append(filename)
            fakes = pd.DataFrame(list(zip(data_list, data_list_name)), columns=[
                                 'path', 'videoname'])
            fakes['testlist'] = fakes['videoname'].str[:-4]
            fakes = fakes[fakes['testlist'].isin(test_vids)]
            fakes['label'] = 1
            # put fakes and reals in one dataframe
            df_test = pd.concat([reals, fakes], ignore_index=True)
            del df_test['testlist']
            del df_test['videoname']
            df_test = df_test.rename(columns={'path': 'video'})
            return df_test
        elif dataset == 'dfdc':
            # prepare dfdc training data
            # structure data from folder in data frame for loading
            all_meta_train, all_meta_test, full_margin_aug_val = dfdc_metadata_setup()
            all_meta_test['videoname'] = all_meta_test['video']
            all_meta_test['video'] = dataset_path + \
                '/test/' + all_meta_test['videoname']
            # randomly sample 1000 test videos
            df_test_reals = all_meta_test[all_meta_test['label'] == 0]
            df_test_fakes = all_meta_test[all_meta_test['label'] == 1]
            df_test_reals = df_test_reals.sample(
                n=500, replace=False, random_state=24)
            df_test_fakes = df_test_fakes.sample(
                n=500, replace=False, random_state=24)
            df_test = pd.concat(
                [df_test_reals, df_test_fakes], ignore_index=True)
            print(df_test)
            return df_test
        elif dataset == 'generateddf':
            video_path_test_real = os.path.join(dataset_path + "/real/")
            video_path_test_fake = os.path.join(dataset_path + "/fake/")
            data_list = []
            for _, _, videos in os.walk(video_path_test_real):
                for video in tqdm(videos):
                    # append test video
                    data_list.append(
                        {'label': 0, 'video': video_path_test_real + video})
            print(data_list)
            for _, _, videos in os.walk(video_path_test_fake):
                for video in tqdm(videos):
                    # label 1 for deepfake image
                    data_list.append(
                        {'label': 1, 'video': video_path_test_fake + video})
            return pd.DataFrame(data_list)
    # if test_data:
    #     print(f"{len(df)} test videos.")
    # else:
    #     if face_crops:
    #         print(f"Lead to: {len(df)} face crops.")
    #     else:
    #         print(f"{len(df)} train videos.")
    # print()
    # return df


def prepare_sequence_data(dataset, df):
    """
    Prepares the dataframe for sequence models.
    """
    print(df)
    df = df.sort_values(by=['video']).reset_index(drop=True)
    # add original column
    df['original'] = ""
    if dataset == 'uadfv':
        # label data
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row.loc['label'] == 0:
                df.loc[idx, 'original'] = row.loc['video'][:4]
            elif row.loc['label'] == 1:
                df.loc[idx, 'original'] = row.loc['video'][:9]
    elif dataset == 'celebdf' or dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # remove everything after last underscore
            df.loc[idx, 'original'] = row.loc['video'].rpartition("_")[0]
    elif dataset == 'dfdc':
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # remove everything after last underscore
            df.loc[idx, 'original'] = row.loc['video'].rpartition("_")[0][-10:]
    # count frames per video
    df1 = df.groupby(['original']).size().reset_index(name='count')
    df = pd.merge(df, df1, on='original')
    # remove videos that don't where less than 20 frames
    # were detected to ensure equal frame size of 20 for sequence
    # for dfdc only 5 frames because dataset is so large
    if dataset == 'dfdc':
        df = df[df['count'] == 5]
    else:
        df = df[df['count'] == 20]
    df = df[['label', 'original']]
    # ensure that dataframe includes each video with 20 frames once
    df = df.groupby(['label', 'original']).size().reset_index(name='count')
    df = df[['label', 'original']]
    return df


def dfdc_metadata_setup():
    """Returns training, testing and validation video meta data frames for the DFDC dataset."""
    #read in metadata
    print("Reading metadata...this can take a minute.")
    df_train0 = pd.read_json(os.getcwd() + '/metadata/metadata0.json')
    df_train0.loc[df_train0.shape[0]] = 0
    df_train0.rename({3: 'folder'}, axis='index', inplace=True)
    df_train1 = pd.read_json(os.getcwd() + '/metadata/metadata1.json')
    df_train1.loc[df_train1.shape[0]] = 1
    df_train1.rename({3: 'folder'}, axis='index', inplace=True)
    df_train2 = pd.read_json(os.getcwd() + '/metadata/metadata3.json')
    df_train2.loc[df_train2.shape[0]] = 3
    df_train2.rename({3: 'folder'}, axis='index', inplace=True)
    df_train3 = pd.read_json(os.getcwd() + '/metadata/metadata4.json')
    df_train3.loc[df_train3.shape[0]] = 4
    df_train3.rename({3: 'folder'}, axis='index', inplace=True)
    df_train4 = pd.read_json(os.getcwd() + '/metadata/metadata5.json')
    df_train4.loc[df_train4.shape[0]] = 5
    df_train4.rename({3: 'folder'}, axis='index', inplace=True)
    df_train5 = pd.read_json(os.getcwd() + '/metadata/metadata6.json')
    df_train5.loc[df_train5.shape[0]] = 6
    df_train5.rename({3: 'folder'}, axis='index', inplace=True)
    df_train6 = pd.read_json(os.getcwd() + '/metadata/metadata8.json')
    df_train6.loc[df_train6.shape[0]] = 8
    df_train6.rename({3: 'folder'}, axis='index', inplace=True)
    df_train7 = pd.read_json(os.getcwd() + '/metadata/metadata9.json')
    df_train7.loc[df_train7.shape[0]] = 9
    df_train7.rename({3: 'folder'}, axis='index', inplace=True)
    df_train8 = pd.read_json(os.getcwd() + '/metadata/metadata10.json')
    df_train8.loc[df_train8.shape[0]] = 10
    df_train8.rename({3: 'folder'}, axis='index', inplace=True)
    df_train9 = pd.read_json(os.getcwd() + '/metadata/metadata12.json')
    df_train9.loc[df_train9.shape[0]] = 12
    df_train9.rename({3: 'folder'}, axis='index', inplace=True)
    df_train10 = pd.read_json(os.getcwd() + '/metadata/metadata13.json')
    df_train10.loc[df_train10.shape[0]] = 13
    df_train10.rename({3: 'folder'}, axis='index', inplace=True)
    df_train11 = pd.read_json(os.getcwd() + '/metadata/metadata14.json')
    df_train11.loc[df_train11.shape[0]] = 14
    df_train11.rename({3: 'folder'}, axis='index', inplace=True)
    df_train12 = pd.read_json(os.getcwd() + '/metadata/metadata15.json')
    df_train12.loc[df_train12.shape[0]] = 15
    df_train12.rename({3: 'folder'}, axis='index', inplace=True)
    df_train13 = pd.read_json(os.getcwd() + '/metadata/metadata16.json')
    df_train13.loc[df_train13.shape[0]] = 16
    df_train13.rename({3: 'folder'}, axis='index', inplace=True)
    df_train14 = pd.read_json(os.getcwd() + '/metadata/metadata17.json')
    df_train14.loc[df_train14.shape[0]] = 17
    df_train14.rename({3: 'folder'}, axis='index', inplace=True)
    df_train15 = pd.read_json(os.getcwd() + '/metadata/metadata19.json')
    df_train15.loc[df_train15.shape[0]] = 19
    df_train15.rename({3: 'folder'}, axis='index', inplace=True)
    df_train16 = pd.read_json(os.getcwd() + '/metadata/metadata20.json')
    df_train16.loc[df_train16.shape[0]] = 20
    df_train16.rename({3: 'folder'}, axis='index', inplace=True)
    df_train17 = pd.read_json(os.getcwd() + '/metadata/metadata22.json')
    df_train17.loc[df_train17.shape[0]] = 22
    df_train17.rename({3: 'folder'}, axis='index', inplace=True)
    df_train18 = pd.read_json(os.getcwd() + '/metadata/metadata23.json')
    df_train18.loc[df_train18.shape[0]] = 23
    df_train18.rename({3: 'folder'}, axis='index', inplace=True)
    df_train19 = pd.read_json(os.getcwd() + '/metadata/metadata24.json')
    df_train19.loc[df_train19.shape[0]] = 24
    df_train19.rename({3: 'folder'}, axis='index', inplace=True)
    df_train20 = pd.read_json(os.getcwd() + '/metadata/metadata25.json')
    df_train20.loc[df_train20.shape[0]] = 25
    df_train20.rename({3: 'folder'}, axis='index', inplace=True)
    df_train21 = pd.read_json(os.getcwd() + '/metadata/metadata26.json')
    df_train21.loc[df_train21.shape[0]] = 26
    df_train21.rename({3: 'folder'}, axis='index', inplace=True)
    df_train39 = pd.read_json(os.getcwd() + '/metadata/metadata27.json')
    df_train39.loc[df_train39.shape[0]] = 27
    df_train39.rename({3: 'folder'}, axis='index', inplace=True)
    df_train22 = pd.read_json(os.getcwd() + '/metadata/metadata29.json')
    df_train22.loc[df_train22.shape[0]] = 29
    df_train22.rename({3: 'folder'}, axis='index', inplace=True)
    df_train23 = pd.read_json(os.getcwd() + '/metadata/metadata30.json')
    df_train23.loc[df_train23.shape[0]] = 30
    df_train23.rename({3: 'folder'}, axis='index', inplace=True)
    df_train24 = pd.read_json(os.getcwd() + '/metadata/metadata31.json')
    df_train24.loc[df_train24.shape[0]] = 31
    df_train24.rename({3: 'folder'}, axis='index', inplace=True)
    df_train25 = pd.read_json(os.getcwd() + '/metadata/metadata32.json')
    df_train25.loc[df_train25.shape[0]] = 32
    df_train25.rename({3: 'folder'}, axis='index', inplace=True)
    df_train26 = pd.read_json(os.getcwd() + '/metadata/metadata34.json')
    df_train26.loc[df_train26.shape[0]] = 34
    df_train26.rename({3: 'folder'}, axis='index', inplace=True)
    df_train27 = pd.read_json(os.getcwd() + '/metadata/metadata35.json')
    df_train27.loc[df_train27.shape[0]] = 35
    df_train27.rename({3: 'folder'}, axis='index', inplace=True)
    df_train28 = pd.read_json(os.getcwd() + '/metadata/metadata36.json')
    df_train28.loc[df_train28.shape[0]] = 36
    df_train28.rename({3: 'folder'}, axis='index', inplace=True)
    df_train29 = pd.read_json(os.getcwd() + '/metadata/metadata37.json')
    df_train29.loc[df_train29.shape[0]] = 37
    df_train29.rename({3: 'folder'}, axis='index', inplace=True)
    df_train30 = pd.read_json(os.getcwd() + '/metadata/metadata38.json')
    df_train30.loc[df_train30.shape[0]] = 38
    df_train30.rename({3: 'folder'}, axis='index', inplace=True)
    df_train31 = pd.read_json(os.getcwd() + '/metadata/metadata40.json')
    df_train31.loc[df_train31.shape[0]] = 40
    df_train31.rename({3: 'folder'}, axis='index', inplace=True)
    df_train32 = pd.read_json(os.getcwd() + '/metadata/metadata41.json')
    df_train32.loc[df_train32.shape[0]] = 41
    df_train32.rename({3: 'folder'}, axis='index', inplace=True)
    df_train33 = pd.read_json(os.getcwd() + '/metadata/metadata43.json')
    df_train33.loc[df_train33.shape[0]] = 43
    df_train33.rename({3: 'folder'}, axis='index', inplace=True)
    df_train34 = pd.read_json(os.getcwd() + '/metadata/metadata44.json')
    df_train34.loc[df_train34.shape[0]] = 44
    df_train34.rename({3: 'folder'}, axis='index', inplace=True)
    df_train35 = pd.read_json(os.getcwd() + '/metadata/metadata45.json')
    df_train35.loc[df_train35.shape[0]] = 45
    df_train35.rename({3: 'folder'}, axis='index', inplace=True)
    df_train36 = pd.read_json(os.getcwd() + '/metadata/metadata46.json')
    df_train36.loc[df_train36.shape[0]] = 46
    df_train36.rename({3: 'folder'}, axis='index', inplace=True)
    df_train37 = pd.read_json(os.getcwd() + '/metadata/metadata47.json')
    df_train37.loc[df_train37.shape[0]] = 47
    df_train37.rename({3: 'folder'}, axis='index', inplace=True)
    df_train38 = pd.read_json(os.getcwd() + '/metadata/metadata49.json')
    df_train38.loc[df_train38.shape[0]] = 49
    df_train38.rename({3: 'folder'}, axis='index', inplace=True)
    df_train40 = pd.read_json(os.getcwd() + '/metadata/metadata18.json')
    df_train40.loc[df_train40.shape[0]] = 18
    df_train40.rename({3: 'folder'}, axis='index', inplace=True)
    df_train41 = pd.read_json(os.getcwd() + '/metadata/metadata2.json')
    df_train41.loc[df_train41.shape[0]] = 2
    df_train41.rename({3: 'folder'}, axis='index', inplace=True)
    df_train42 = pd.read_json(os.getcwd() + '/metadata/metadata28.json')
    df_train42.loc[df_train42.shape[0]] = 28
    df_train42.rename({3: 'folder'}, axis='index', inplace=True)
    df_train43 = pd.read_json(os.getcwd() + '/metadata/metadata42.json')
    df_train43.loc[df_train43.shape[0]] = 42
    df_train43.rename({3: 'folder'}, axis='index', inplace=True)
    df_train44 = pd.read_json(os.getcwd() + '/metadata/metadata7.json')
    df_train44.loc[df_train44.shape[0]] = 7
    df_train44.rename({3: 'folder'}, axis='index', inplace=True)
    df_train45 = pd.read_json(os.getcwd() + '/metadata/metadata11.json')
    df_train45.loc[df_train45.shape[0]] = 11
    df_train45.rename({3: 'folder'}, axis='index', inplace=True)
    df_train46 = pd.read_json(os.getcwd() + '/metadata/metadata21.json')
    df_train46.loc[df_train46.shape[0]] = 21
    df_train46.rename({3: 'folder'}, axis='index', inplace=True)
    df_train47 = pd.read_json(os.getcwd() + '/metadata/metadata33.json')
    df_train47.loc[df_train47.shape[0]] = 33
    df_train47.rename({3: 'folder'}, axis='index', inplace=True)
    df_train48 = pd.read_json(os.getcwd() + '/metadata/metadata39.json')
    df_train48.loc[df_train48.shape[0]] = 39
    df_train48.rename({3: 'folder'}, axis='index', inplace=True)
    df_train49 = pd.read_json(os.getcwd() + '/metadata/metadata48.json')
    df_train49.loc[df_train49.shape[0]] = 48
    df_train49.rename({3: 'folder'}, axis='index', inplace=True)
    # combine metadata
    print("Formatting metadata...")
    df_train = [df_train0, df_train1, df_train2, df_train3, df_train4,
                df_train5, df_train6, df_train7, df_train8, df_train9, df_train10,
                df_train11, df_train12, df_train13, df_train14, df_train15,
                df_train16, df_train17, df_train18, df_train19, df_train20, df_train21,
                df_train22, df_train23, df_train24, df_train25, df_train26,
                df_train27, df_train28, df_train29, df_train30, df_train31, df_train32,
                df_train33, df_train34, df_train35, df_train36, df_train37,
                df_train38, df_train39, df_train40, df_train41, df_train42, df_train43, df_train44,
                df_train45, df_train46, df_train47, df_train48, df_train49]
    all_meta = pd.concat(df_train, axis=1)
    all_meta = all_meta.T  # transpose
    all_meta['video'] = all_meta.index  # create video column from index
    all_meta.reset_index(drop=True, inplace=True)  # drop index
    del all_meta['split']
    # recode labels
    all_meta['label'] = all_meta['label'].apply(
        lambda x: 0 if x == 'REAL' else 1)
    del all_meta['original']
    # sample 16974 fakes from 45 folders -> that's approx. 378 fakes per folder
    train_df = all_meta[all_meta['folder'] < 45]
    # 16974 reals in train data and 89629 fakes
    reals = train_df[train_df['label'] == 0]
    #del reals['folder']
    reals['folder']
    fakes = train_df[train_df['label'] == 1]
    fakes_sampled = fakes[fakes['folder'] == 0].sample(378, random_state=24)
    # sample the same number of fake videos from every folder
    for num in range(45):
        if num == 0:
            continue
        sample = fakes[fakes['folder'] == num].sample(378, random_state=24)
        fakes_sampled = fakes_sampled.append(sample, ignore_index=True)
    # drop 36 videos randomly to have exactly 16974 fakes
    np.random.seed(24)
    drop_indices = np.random.choice(fakes_sampled.index, 36, replace=False)
    fakes_sampled = fakes_sampled.drop(drop_indices)
    #del fakes_sampled['folder']
    fakes_sampled['folder']
    all_meta_train = pd.concat([reals, fakes_sampled], ignore_index=True)
    # get 1000 samples from training data that are used for margin and augmentation validation
    real_sample = all_meta_train[all_meta_train['label'] == 0].sample(
        300, random_state=24)
    fake_sample = all_meta_train[all_meta_train['label'] == 1].sample(
        300, random_state=24)
    full_margin_aug_val = real_sample.append(fake_sample, ignore_index=True)
    # create test set
    test_df = all_meta[all_meta['folder'] > 44]
    del test_df['folder']
    all_meta_test = test_df.reset_index(drop=True)

    return all_meta_train, all_meta_test, full_margin_aug_val


def switch_one_zero(num):
    """Switch label 1 to 0 and 0 to 1
        so that fake videos have label 1.
    """
    if num == 1:
        num = 0
    else:
        num = 1
    return num


def setup_celebdf_benchmark(data_path, method):
    """
    Setup the folder structure of the Celeb-DF Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/celeb-deepfakeforensics
                                and scroll down to the dataset section.
                                Click on the link \"this form\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./celebdf/
                                        Celeb-real/
                                        Celeb-synthesis/
                                        YouTube-real/
                                        List_of_testing_videos.txt
                                """)
    if data_path.endswith("celebdf"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m Celeb-DF \033[0m dataset with ...")
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./celebdf/
                                    Celeb-real/
                                    Celeb-synthesis/
                                    YouTube-real/
                                    List_of_testing_videos.txt
                        """)


def setup_uadfv_benchmark(data_path, method):
    """
    Setup the folder structure of the UADFV Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi
                                and scroll down to the UADFV section.
                                Click on the link \"here\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./fake_videos/
                                            fake/
                                            real/
                                """)
    if data_path.endswith("fake_videos.zip"):
        raise ValueError("Please make sure to extract the zipfile.")
    if data_path.endswith("fake_videos"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m UADFV \033[0m dataset with ...")
        # create test directories if they don't exist
        if not os.path.exists(data_path + '/test/'):
            structure_uadfv_files(path_to_data=data_path)
        else:
            # check if path exists but files are not complete (from https://stackoverflow.com/a/2632251)
            num_files = len([f for f in os.listdir(
                data_path + '/test/') if os.path.isfile(os.path.join(data_path + '/test/real/', f))])
            num_files += len([f for f in os.listdir(data_path + '/test/')
                              if os.path.isfile(os.path.join(data_path + '/test/fake/', f))])
            # check whether all 28 test videos are in directories
            if num_files != 28:
                # recreate all 28 files
                shutil.rmtree(data_path + '/test/')
                structure_uadfv_files(path_to_data=data_path)
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./fake_videos/
                                        fake/
                                        real/
                        """)


def structure_uadfv_files(path_to_data):
    """Creates test folders and moves test videos there."""
    os.mkdir(path_to_data + '/test/')
    os.mkdir(path_to_data + '/test/fake/')
    os.mkdir(path_to_data + '/test/real/')
    test_data = pd.read_csv(
        os.getcwd() + "/uadfv_test.csv", names=['video'], header=None)
    for idx, row in test_data.iterrows():
        if len(str(row.loc['video'])) > 8:
            # video is fake, therefore copy it into fake test folder
            shutil.copy(path_to_data + '/fake/' +
                        row['video'], path_to_data + '/test/fake/')
        else:
            # video is real, therefore move it into real test folder
            shutil.copy(path_to_data + '/real/' +
                        row['video'], path_to_data + '/test/real/')


def setup_dftimit_hq_benchmark(data_path, method):
    """
    Setup the folder structure of the DFTIMIT HQ Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to http://conradsanderson.id.au/vidtimit/ to download the real videos and to
                                https://www.idiap.ch/dataset/deepfaketimit to download the deepfake videos.
                                Extract the files and organize the folders follwing this folder structure:
                                ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                                """)
    if data_path.endswith("DeepfakeTIMIT"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m DF-TIMIT-HQ \033[0m dataset with ...")
    else:
        raise ValueError("""Make sure your data_path argument ends with \"DeepfakeTIMIT\" and organize the dataset directory in this way:
                            ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                        """)


def setup_dftimit_lq_benchmark(data_path, method):
    """
    Setup the folder structure of the DFTIMIT HQ Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to http://conradsanderson.id.au/vidtimit/ to download the real videos and to
                                https://www.idiap.ch/dataset/deepfaketimit to download the deepfake videos.
                                Extract the files and organize the folders follwing this folder structure:
                                ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                                """)
    if data_path.endswith("DeepfakeTIMIT"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m DF-TIMIT-LQ \033[0m dataset with ...")
    else:
        raise ValueError("""Make sure your data_path argument ends with \"DeepfakeTIMIT\" and organize the dataset directory in this way:
                            ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                        """)


def dfdc_metadata_setup():
    """Returns training, testing and validation video meta data frames for the DFDC dataset."""
    #read in metadata
    print("Reading metadata...this can take a minute.")
    df_train0 = pd.read_json(os.getcwd() + '/metadata/metadata0.json')
    df_train0.loc[df_train0.shape[0]] = 0
    df_train0.rename({3: 'folder'}, axis='index', inplace=True)
    df_train1 = pd.read_json(os.getcwd() + '/metadata/metadata1.json')
    df_train1.loc[df_train1.shape[0]] = 1
    df_train1.rename({3: 'folder'}, axis='index', inplace=True)
    df_train2 = pd.read_json(os.getcwd() + '/metadata/metadata3.json')
    df_train2.loc[df_train2.shape[0]] = 3
    df_train2.rename({3: 'folder'}, axis='index', inplace=True)
    df_train3 = pd.read_json(os.getcwd() + '/metadata/metadata4.json')
    df_train3.loc[df_train3.shape[0]] = 4
    df_train3.rename({3: 'folder'}, axis='index', inplace=True)
    df_train4 = pd.read_json(os.getcwd() + '/metadata/metadata5.json')
    df_train4.loc[df_train4.shape[0]] = 5
    df_train4.rename({3: 'folder'}, axis='index', inplace=True)
    df_train5 = pd.read_json(os.getcwd() + '/metadata/metadata6.json')
    df_train5.loc[df_train5.shape[0]] = 6
    df_train5.rename({3: 'folder'}, axis='index', inplace=True)
    df_train6 = pd.read_json(os.getcwd() + '/metadata/metadata8.json')
    df_train6.loc[df_train6.shape[0]] = 8
    df_train6.rename({3: 'folder'}, axis='index', inplace=True)
    df_train7 = pd.read_json(os.getcwd() + '/metadata/metadata9.json')
    df_train7.loc[df_train7.shape[0]] = 9
    df_train7.rename({3: 'folder'}, axis='index', inplace=True)
    df_train8 = pd.read_json(os.getcwd() + '/metadata/metadata10.json')
    df_train8.loc[df_train8.shape[0]] = 10
    df_train8.rename({3: 'folder'}, axis='index', inplace=True)
    df_train9 = pd.read_json(os.getcwd() + '/metadata/metadata12.json')
    df_train9.loc[df_train9.shape[0]] = 12
    df_train9.rename({3: 'folder'}, axis='index', inplace=True)
    df_train10 = pd.read_json(os.getcwd() + '/metadata/metadata13.json')
    df_train10.loc[df_train10.shape[0]] = 13
    df_train10.rename({3: 'folder'}, axis='index', inplace=True)
    df_train11 = pd.read_json(os.getcwd() + '/metadata/metadata14.json')
    df_train11.loc[df_train11.shape[0]] = 14
    df_train11.rename({3: 'folder'}, axis='index', inplace=True)
    df_train12 = pd.read_json(os.getcwd() + '/metadata/metadata15.json')
    df_train12.loc[df_train12.shape[0]] = 15
    df_train12.rename({3: 'folder'}, axis='index', inplace=True)
    df_train13 = pd.read_json(os.getcwd() + '/metadata/metadata16.json')
    df_train13.loc[df_train13.shape[0]] = 16
    df_train13.rename({3: 'folder'}, axis='index', inplace=True)
    df_train14 = pd.read_json(os.getcwd() + '/metadata/metadata17.json')
    df_train14.loc[df_train14.shape[0]] = 17
    df_train14.rename({3: 'folder'}, axis='index', inplace=True)
    df_train15 = pd.read_json(os.getcwd() + '/metadata/metadata19.json')
    df_train15.loc[df_train15.shape[0]] = 19
    df_train15.rename({3: 'folder'}, axis='index', inplace=True)
    df_train16 = pd.read_json(os.getcwd() + '/metadata/metadata20.json')
    df_train16.loc[df_train16.shape[0]] = 20
    df_train16.rename({3: 'folder'}, axis='index', inplace=True)
    df_train17 = pd.read_json(os.getcwd() + '/metadata/metadata22.json')
    df_train17.loc[df_train17.shape[0]] = 22
    df_train17.rename({3: 'folder'}, axis='index', inplace=True)
    df_train18 = pd.read_json(os.getcwd() + '/metadata/metadata23.json')
    df_train18.loc[df_train18.shape[0]] = 23
    df_train18.rename({3: 'folder'}, axis='index', inplace=True)
    df_train19 = pd.read_json(os.getcwd() + '/metadata/metadata24.json')
    df_train19.loc[df_train19.shape[0]] = 24
    df_train19.rename({3: 'folder'}, axis='index', inplace=True)
    df_train20 = pd.read_json(os.getcwd() + '/metadata/metadata25.json')
    df_train20.loc[df_train20.shape[0]] = 25
    df_train20.rename({3: 'folder'}, axis='index', inplace=True)
    df_train21 = pd.read_json(os.getcwd() + '/metadata/metadata26.json')
    df_train21.loc[df_train21.shape[0]] = 26
    df_train21.rename({3: 'folder'}, axis='index', inplace=True)
    df_train39 = pd.read_json(os.getcwd() + '/metadata/metadata27.json')
    df_train39.loc[df_train39.shape[0]] = 27
    df_train39.rename({3: 'folder'}, axis='index', inplace=True)
    df_train22 = pd.read_json(os.getcwd() + '/metadata/metadata29.json')
    df_train22.loc[df_train22.shape[0]] = 29
    df_train22.rename({3: 'folder'}, axis='index', inplace=True)
    df_train23 = pd.read_json(os.getcwd() + '/metadata/metadata30.json')
    df_train23.loc[df_train23.shape[0]] = 30
    df_train23.rename({3: 'folder'}, axis='index', inplace=True)
    df_train24 = pd.read_json(os.getcwd() + '/metadata/metadata31.json')
    df_train24.loc[df_train24.shape[0]] = 31
    df_train24.rename({3: 'folder'}, axis='index', inplace=True)
    df_train25 = pd.read_json(os.getcwd() + '/metadata/metadata32.json')
    df_train25.loc[df_train25.shape[0]] = 32
    df_train25.rename({3: 'folder'}, axis='index', inplace=True)
    df_train26 = pd.read_json(os.getcwd() + '/metadata/metadata34.json')
    df_train26.loc[df_train26.shape[0]] = 34
    df_train26.rename({3: 'folder'}, axis='index', inplace=True)
    df_train27 = pd.read_json(os.getcwd() + '/metadata/metadata35.json')
    df_train27.loc[df_train27.shape[0]] = 35
    df_train27.rename({3: 'folder'}, axis='index', inplace=True)
    df_train28 = pd.read_json(os.getcwd() + '/metadata/metadata36.json')
    df_train28.loc[df_train28.shape[0]] = 36
    df_train28.rename({3: 'folder'}, axis='index', inplace=True)
    df_train29 = pd.read_json(os.getcwd() + '/metadata/metadata37.json')
    df_train29.loc[df_train29.shape[0]] = 37
    df_train29.rename({3: 'folder'}, axis='index', inplace=True)
    df_train30 = pd.read_json(os.getcwd() + '/metadata/metadata38.json')
    df_train30.loc[df_train30.shape[0]] = 38
    df_train30.rename({3: 'folder'}, axis='index', inplace=True)
    df_train31 = pd.read_json(os.getcwd() + '/metadata/metadata40.json')
    df_train31.loc[df_train31.shape[0]] = 40
    df_train31.rename({3: 'folder'}, axis='index', inplace=True)
    df_train32 = pd.read_json(os.getcwd() + '/metadata/metadata41.json')
    df_train32.loc[df_train32.shape[0]] = 41
    df_train32.rename({3: 'folder'}, axis='index', inplace=True)
    df_train33 = pd.read_json(os.getcwd() + '/metadata/metadata43.json')
    df_train33.loc[df_train33.shape[0]] = 43
    df_train33.rename({3: 'folder'}, axis='index', inplace=True)
    df_train34 = pd.read_json(os.getcwd() + '/metadata/metadata44.json')
    df_train34.loc[df_train34.shape[0]] = 44
    df_train34.rename({3: 'folder'}, axis='index', inplace=True)
    df_train35 = pd.read_json(os.getcwd() + '/metadata/metadata45.json')
    df_train35.loc[df_train35.shape[0]] = 45
    df_train35.rename({3: 'folder'}, axis='index', inplace=True)
    df_train36 = pd.read_json(os.getcwd() + '/metadata/metadata46.json')
    df_train36.loc[df_train36.shape[0]] = 46
    df_train36.rename({3: 'folder'}, axis='index', inplace=True)
    df_train37 = pd.read_json(os.getcwd() + '/metadata/metadata47.json')
    df_train37.loc[df_train37.shape[0]] = 47
    df_train37.rename({3: 'folder'}, axis='index', inplace=True)
    df_train38 = pd.read_json(os.getcwd() + '/metadata/metadata49.json')
    df_train38.loc[df_train38.shape[0]] = 49
    df_train38.rename({3: 'folder'}, axis='index', inplace=True)
    df_train40 = pd.read_json(os.getcwd() + '/metadata/metadata18.json')
    df_train40.loc[df_train40.shape[0]] = 18
    df_train40.rename({3: 'folder'}, axis='index', inplace=True)
    df_train41 = pd.read_json(os.getcwd() + '/metadata/metadata2.json')
    df_train41.loc[df_train41.shape[0]] = 2
    df_train41.rename({3: 'folder'}, axis='index', inplace=True)
    df_train42 = pd.read_json(os.getcwd() + '/metadata/metadata28.json')
    df_train42.loc[df_train42.shape[0]] = 28
    df_train42.rename({3: 'folder'}, axis='index', inplace=True)
    df_train43 = pd.read_json(os.getcwd() + '/metadata/metadata42.json')
    df_train43.loc[df_train43.shape[0]] = 42
    df_train43.rename({3: 'folder'}, axis='index', inplace=True)
    df_train44 = pd.read_json(os.getcwd() + '/metadata/metadata7.json')
    df_train44.loc[df_train44.shape[0]] = 7
    df_train44.rename({3: 'folder'}, axis='index', inplace=True)
    df_train45 = pd.read_json(os.getcwd() + '/metadata/metadata11.json')
    df_train45.loc[df_train45.shape[0]] = 11
    df_train45.rename({3: 'folder'}, axis='index', inplace=True)
    df_train46 = pd.read_json(os.getcwd() + '/metadata/metadata21.json')
    df_train46.loc[df_train46.shape[0]] = 21
    df_train46.rename({3: 'folder'}, axis='index', inplace=True)
    df_train47 = pd.read_json(os.getcwd() + '/metadata/metadata33.json')
    df_train47.loc[df_train47.shape[0]] = 33
    df_train47.rename({3: 'folder'}, axis='index', inplace=True)
    df_train48 = pd.read_json(os.getcwd() + '/metadata/metadata39.json')
    df_train48.loc[df_train48.shape[0]] = 39
    df_train48.rename({3: 'folder'}, axis='index', inplace=True)
    df_train49 = pd.read_json(os.getcwd() + '/metadata/metadata48.json')
    df_train49.loc[df_train49.shape[0]] = 48
    df_train49.rename({3: 'folder'}, axis='index', inplace=True)
    # combine metadata
    print("Formatting metadata...")
    df_train = [df_train0, df_train1, df_train2, df_train3, df_train4,
                df_train5, df_train6, df_train7, df_train8, df_train9, df_train10,
                df_train11, df_train12, df_train13, df_train14, df_train15,
                df_train16, df_train17, df_train18, df_train19, df_train20, df_train21,
                df_train22, df_train23, df_train24, df_train25, df_train26,
                df_train27, df_train28, df_train29, df_train30, df_train31, df_train32,
                df_train33, df_train34, df_train35, df_train36, df_train37,
                df_train38, df_train39, df_train40, df_train41, df_train42, df_train43, df_train44,
                df_train45, df_train46, df_train47, df_train48, df_train49]
    all_meta = pd.concat(df_train, axis=1)
    all_meta = all_meta.T  # transpose
    all_meta['video'] = all_meta.index  # create video column from index
    all_meta.reset_index(drop=True, inplace=True)  # drop index
    del all_meta['split']
    # recode labels
    all_meta['label'] = all_meta['label'].apply(
        lambda x: 0 if x == 'REAL' else 1)
    del all_meta['original']
    # sample 16974 fakes from 45 folders -> that's approx. 378 fakes per folder
    train_df = all_meta[all_meta['folder'] < 45]
    # 16974 reals in train data and 89629 fakes
    reals = train_df[train_df['label'] == 0]
    #del reals['folder']
    reals['folder']
    fakes = train_df[train_df['label'] == 1]
    fakes_sampled = fakes[fakes['folder'] == 0].sample(378, random_state=24)
    # sample the same number of fake videos from every folder
    for num in range(45):
        if num == 0:
            continue
        sample = fakes[fakes['folder'] == num].sample(378, random_state=24)
        fakes_sampled = fakes_sampled.append(sample, ignore_index=True)
    # drop 36 videos randomly to have exactly 16974 fakes
    np.random.seed(24)
    drop_indices = np.random.choice(fakes_sampled.index, 36, replace=False)
    fakes_sampled = fakes_sampled.drop(drop_indices)
    #del fakes_sampled['folder']
    fakes_sampled['folder']
    all_meta_train = pd.concat([reals, fakes_sampled], ignore_index=True)
    # get 1000 samples from training data that are used for margin and augmentation validation
    real_sample = all_meta_train[all_meta_train['label'] == 0].sample(
        300, random_state=24)
    fake_sample = all_meta_train[all_meta_train['label'] == 1].sample(
        300, random_state=24)
    full_margin_aug_val = real_sample.append(fake_sample, ignore_index=True)
    # create test set
    test_df = all_meta[all_meta['folder'] > 44]
    del test_df['folder']
    all_meta_test = test_df.reset_index(drop=True)

    return all_meta_train, all_meta_test, full_margin_aug_val
