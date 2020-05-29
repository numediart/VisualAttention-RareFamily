import cv2
import os
import matplotlib.pyplot as plt
from DeepRare_2019_lib import DeepRare2019
from keras.applications.vgg16 import VGG16



sal = DeepRare2019() # instantiate class
sal.model = VGG16() # call VGG16 and send it to the class
sal.filt = 1 # make filtering
sal.face = 1 # use face (to be used only with VGG16)



directory = r'input'
plt.figure(1)

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        go_path = os.path.join(directory, filename)

        img = cv2.imread(go_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sal.img = img

        saliency_map, saliency_details, face = sal.compute() # here no visualization, only final sal maps but if you use the line above and uncomment the lines in the lib you can get the groups

        plt.subplot(421)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Initial Image')

        plt.subplot(422)
        plt.imshow(saliency_map)
        plt.axis('off')
        plt.title('Final Saliency Map')

        plt.subplot(423)
        plt.imshow(saliency_details[:, :, 0])
        plt.axis('off')
        plt.title('LowLevel1 Saliency Map')

        plt.subplot(424)
        plt.imshow(saliency_details[:, :, 1])
        plt.axis('off')
        plt.title('LowLevel2 Saliency Map')

        plt.subplot(425)
        plt.imshow(saliency_details[:, :, 2])
        plt.axis('off')
        plt.title('MidLevel1 Saliency Map')

        plt.subplot(426)
        plt.imshow(saliency_details[:, :, 3])
        plt.axis('off')
        plt.title('MidLevel2 Saliency Map')

        plt.subplot(427)
        plt.imshow(saliency_details[:, :, 4])
        plt.axis('off')
        plt.title('HighLevel Saliency Map')

        plt.subplot(428)
        plt.imshow(face)
        plt.axis('off')
        plt.title('Large faces')

        plt.show()

        if not os.path.exists('output_raw'):
            os.makedirs('output_raw')
        file_no_extension = os.path.splitext(filename)[0]
        file_out = file_no_extension + '.jpg'
        go_path_raw = os.path.join('output_raw', file_out)
        cv2.imwrite(go_path_raw, 255*saliency_map)

    else:
        continue
