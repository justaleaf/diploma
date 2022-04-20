import os
import random
import shutil


class DatasetGenerator():
    datasets = []
    generated_path = None
    path_to_project = None
    datasets_directory = None
    
    def __init__(self, path_to_project='/home/woghan/projects/diploma/diploma/', datasets_directory='datasets/'):
        self.datasets = ['celebdf', 'dfdcdataset',  'uadfv'] # - timit
        self.generated_path = path_to_project + datasets_directory + 'generateddf/'
        self.path_to_project = path_to_project
        self.datasets_directory = datasets_directory
        return
    

    def generate(self, size=40):

        if self.check():
            print('All datasets are available to use')
            if os.path.exists(self.generated_path):
                if os.path.exists(self.generated_path + '/fake') and os.path.exists(self.generated_path + '/real'):
                    print('All generated folders are created')
                else:
                    os.makedirs(self.generated_path + '/fake')
                    os.makedirs(self.generated_path + '/real')
            else:
                print(f'{self.generated_path} not exists, creating....')
                os.makedirs(self.generated_path)
                os.makedirs(self.generated_path + '/fake')
                os.makedirs(self.generated_path + '/real')

            print('Ready to generate new dataset')
            vids = self.get_videos()
            for df in self.datasets:
                # new_df['real'] += random.sample(vids[df]['real'], k=size)
                # new_df['fake'] += random.sample(vids[df]['fake'], k=size)
                if df == 'celebdf':
                    real_folder = 'Celeb-real/'
                    fake_folder = 'Celeb-synthesis/'
                elif df == 'DeepfakeTIMIT':
                    pass
                elif df == 'dfdcdataset':
                    real_folder = 'test/'
                    fake_folder = 'train/'
                elif df == 'uadfv':
                    real_folder = 'fake_videos/real/'
                    fake_folder = 'fake_videos/fake/'
                real = random.sample(vids[df]['real'], k=size)
                fake = random.sample(vids[df]['fake'], k=size)
                for i in range(size):
                    shutil.copy(self.path_to_project + self.datasets_directory + df + '/' + real_folder + real[i], self.generated_path + 'real')
                    shutil.copy(self.path_to_project + self.datasets_directory + df + '/'  + fake_folder + fake[i], self.generated_path + 'fake')
                print('New dataset was generated.')
                print(f'Location: { self.generated_path }')
        else:
            print('Check failed')
        return


    def check(self):
        folder_structure = {
            'celebdf': ['Celeb-real', 'Celeb-synthesis', 'YouTube-real'],
            # 'DeepfakeTIMIT': ['dftimitreal', 'higher_quality', 'lower_quality'],
            'dfdcdataset': ['test', 'train'],
            'uadfv': ['fake_videos']
        }
        count = 0
        for df in self.datasets:
            # check structure
            full_path = self.path_to_project + self.datasets_directory + df
            subdirs = [f.name for f in os.scandir(full_path) if f.is_dir()]
            if set(folder_structure[df]) == set(subdirs):
                print(f"Dataset \033[1m{df}\033[0m is ready to use")
                count += 1
            else:
                print(f"Dataset \033[1m{df}\033[0m file structure is broken")
        return count == len(self.datasets)
    

    def get_videos(self):
        vids = {}
        for df in self.datasets:
            vids[df] = {}
            if df == 'celebdf':
                real_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/Celeb-real')]
                fake_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/Celeb-synthesis')]
                vids[df]['real'] = real_vids
                vids[df]['fake'] = fake_vids
            elif df == 'DeepfakeTIMIT':
                pass
            elif df == 'dfdcdataset':
                real_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/test')]
                fake_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/train')]
                vids[df]['real'] = real_vids
                vids[df]['fake'] = fake_vids
            elif df == 'uadfv':
                real_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/fake_videos/real')]
                fake_vids = [f for f in os.listdir(self.path_to_project + self.datasets_directory + df + '/fake_videos/fake')]
                vids[df]['real'] = real_vids
                vids[df]['fake'] = fake_vids
        return vids


a = DatasetGenerator()
a.generate()