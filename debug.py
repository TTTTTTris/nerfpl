import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


# log_path = '/data/liuzhen/meta_learning_data/DataSets/operator_tasks_celea_face_100_178/task_000000/logs/SIREN_fit_exp_1/version_0/metrics.csv'

# df = pd.read_csv(log_path)
# print(df.info())

# print(df['train/psnr1'].max())
# print(df['train/psnr2'].max())

# ax = df.plot(x='epoch', y=['train/psnr1', 'train/psnr2'], kind='line', grid=True)
# fig = ax.get_figure()
# fig.savefig('fig.png')


# with open(log_path, 'a+', encoding='utf-8', newline='') as file:
#     rows = len((open(log_path)).readlines())
#     print(rows)

#     # fieldnames = ['first_name', 'xxx']
#     # writer = csv.DictWriter(file, fieldnames=fieldnames)

#     # if rows == 0:
#     #    writer.writeheader()

#     # writer.writerow({'first_name': 10, 'xxx': 'Beans'})
#     # writer.writerow({'first_name': 20, 'xxx': 'Spam'})
#     # writer.writerow({'first_name': 80, 'xxx': 'Spam'})



img = np.zeros((640, 640, 3), np.uint8) 
img_green = np.zeros((512, 512, 3), np.uint8)
img_green[:] = 78,238,148 

cv2.line(img, (350, 350), (img.shape[1], img.shape[0]), (153, 50, 204), 10)

cv2.rectangle(img, (0, 0), (350, 350), (255, 187, 197), cv2.FILLED)

cv2.circle(img, (100, 450), 90, (255, 255, 0), 5)

cv2.putText(img, "DON'T STOP! ", (360, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (245, 255, 220), 3)
cv2.putText(img, "KEEP GOING! ", (360, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (245, 245, 220), 3)

cv2.imshow("Green Image", img_green)
cv2.imshow("Image", img)

cv2.waitKey(0)

