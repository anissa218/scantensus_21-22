
## SELECTING 20 images from 55 patients (folders sdytrain1001 to sdytrain1056 included and not sdytrain1033)

import os
from os import path
import random
import shutil
from pathlib import Path

data = []
root = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/sdy/'
last = '/study_00'
sdyfolders = []

for i in range(1,56):
    n = 1000 + i
    foldername = 'sdytrain' + str(n)
    folder_path = os.path.join(root,foldername,last)
    sdyfolders.append(folder_path)

import os
from os import path
import random

data = []
root = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/sdy/'
last = 'study_00'
target_folder = '/Users/anissaalloula/Desktop/scantensus/scantensus-anissa/selected_sdys'

sdyfolders = []

##create a list of all paths to subfolders sdytrain1001 to 1056

#increased to 57 instead of 56 b/c not folder 33 only has 4 elements for some reason
for i in range(1,57):
    n = 1000 + i
    foldername = 'sdytrain' + str(n)
    folder_path = os.path.join(root,foldername,last)
    sdyfolders.append(folder_path)

##randomly select 20 images from each of those 55 subfolders and copy to a new folder
for subfolder in sdyfolders:
    random.seed(4324)
    if len(os.listdir(subfolder)) > 19:
        random_file = random.sample(os.listdir(subfolder),20)
        for file in random_file:
            path_random_file = os.path.join(subfolder,file)
            target_path = os.path.join(target_folder,file)
            shutil.copy(path_random_file, target_path)

## Henry and anissa manually filtered through the 1100 images to exclude any with a significant discontinuity
## ie images with a big difference between one side of the image and the other

## 39 images were excluded of the ids below:
excluded_images = ['01-4fa54e29eb408e9c2d4b2a985e86163e2c5e52247abf9a2dc126016e507d60a4-0020.png',
 '01-0f5c2d98aeca3bf5f67b1d803a44b5c4341431878a12e528e35886fa04b24a83-0040.png',
 '01-571f6880dd212759244c173916e2f82c5e8b4f7cb3b944d869196b6c35139818-0121.png',
 '01-564dadbaa191daf35f61454812432d1790184ba9721c39032cc3d17f868d3e35-0066.png',
 '01-975c4f4fd0a50a15b3f52bcee4925aba300950103ade05f7255eae126147061a-0004.png',
 '01-79c61135a8523fe2e24f49b0ea8c0b368022cf6b565da666bf49ca860854360a-0012.png',
 '01-5c96020cb1f71ee6ff17c753c34e26c4382a2fedd4528035feb7d3569af2094f-0054.png',
 '01-79d20d4c1fb6fc9b65cec3c36cf21f82f6a1df072589a5502ac495810452f4cd-0109.png',
 '01-5cbb5b60cebd8a9b23c9a268e99f537113beea21fb95537793d020496b182dcb-0131.png',
 '01-a9fd81a16d70495da72ad7b7af63a36f55252bcaf5a55ef5142d8433bf47f8a0-0054.png',
 '01-5c96020cb1f71ee6ff17c753c34e26c4382a2fedd4528035feb7d3569af2094f-0055.png',
 '01-ea7fff9ef5c7abd944fc3f2c834baa96a12d97df97c20db44a9ec79e52193d50-0000.png',
 '01-50d9f3aa316fc3718ecc8f1624fafd9bb041ac7d43ee89b47e9a374f5fd8bfb4-0002.png',
 '01-2f3253c7356e8ecaaa562f23fa2db9b4f55195e8bd82bd71d2d5530ae3619c4f-0000.png',
 '01-9a10132637f944f6b279ac212d729fd61408a0bfaff4f24faa321688727de05a-0008.png',
 '01-441e565834e3561c4276caa8bfe0d775ba863a4de9e3a96ce8043f720f8b596e-0166.png',
 '01-039a5b730e6fa2320f23fa1518b15107329b3d83c011aeaaae44ae35eadfb29c-0000.png',
 '01-564dadbaa191daf35f61454812432d1790184ba9721c39032cc3d17f868d3e35-0010.png',
 '01-cb75b05040208d659153c3e320a0ea6a2db6ea411a93037119958a34f9d7b516-0049.png',
 '01-5c96020cb1f71ee6ff17c753c34e26c4382a2fedd4528035feb7d3569af2094f-0023.png',
 '01-ea7fff9ef5c7abd944fc3f2c834baa96a12d97df97c20db44a9ec79e52193d50-0100.png',
 '01-975c4f4fd0a50a15b3f52bcee4925aba300950103ade05f7255eae126147061a-0077.png',
 '01-0e84ef3dae104a1b2345009d7c21a7131d14a1385aa94d7158e69cd08ba649f9-0000.png',
 '01-91fb856588951bcea2573b3f7e9c865fb3cb73959e789f268afdd5139795c069-0091.png',
 '01-8be6c98d3b29dffa74fd49129e6901656022c4cd2421a40cb95e643d171535d8-0012.png',
 '01-7f63a6ace693a1874025f70ece53f04db18ef8c24b3a11444b540d19e6f258ae-0074.png',
 '01-ea7fff9ef5c7abd944fc3f2c834baa96a12d97df97c20db44a9ec79e52193d50-0090.png',
 '01-4efe7a84cf130d45ac3506ff7e49815f4126e18227279f88b3c67e7256aabc23-0002.png',
 '01-975c4f4fd0a50a15b3f52bcee4925aba300950103ade05f7255eae126147061a-0046.png',
 '01-571f6880dd212759244c173916e2f82c5e8b4f7cb3b944d869196b6c35139818-0016.png',
 '01-4efe7a84cf130d45ac3506ff7e49815f4126e18227279f88b3c67e7256aabc23-0005.png',
 '01-208a9676175c11399b7376b4bb599d565bbff52b2ba0a318079b4ed4318bd3be-0115.png',
 '01-208a9676175c11399b7376b4bb599d565bbff52b2ba0a318079b4ed4318bd3be-0074.png',
 '01-6d900ce34b220f033428fd8308c0d6de5a83ffc6c4d2177b942b2073084abd98-0019.png',
 '01-79c61135a8523fe2e24f49b0ea8c0b368022cf6b565da666bf49ca860854360a-0025.png',
 '01-32155aec0ef78e42f921008c0167b3b49c48bd383d43d7a4f0ba930bdad40e86-0334.png',
 '01-45c14833dbd2d7577baf17f5d611268da9fac9c3b42ccbab0e798fe5a410f296-0008.png',
 '01-67a543669ebb386837b20367fe06f6038d68ea09ba7096db952e7589cc769d07-0060.png',
 '01-f59d1ea3e31def423223a153cda17da36c897d3ba50e0ec4244eaf3f7b4ed8f5-0083.png']





