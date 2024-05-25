import torch
import time
import numpy as np
import os

class Test:
    def __init__(self, configs):
        self.t_time = 0.0
        self.t_sec = 0.0
        self.net = configs['netname']()
        self.test = configs['dataset']['test']
        self.val_dataloader = torch.utils.data.DataLoader(self.test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pth = configs['pth_repo']
        self.save_path = configs['test_path']
        self.print_staistaic_text = self.save_path + 'print_staistaic_text.txt'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def start(self):
        print("Loading .......   path:{}".format(self.pth))
        self.net.load_state_dict(torch.load(self.pth)['model'])
        self.net.to(self.device)
        test_normstatic=1
        self.val_step(test_normstatic,self.pth[-5],self.val_dataloader)

    def val_step(self,test_normstatic, epoch, dataset):
        print('-----------------start test--------------------')


        self.csv_onlylable = []

        self.net = self.net.eval()
        star_time = time.time()

        for i, data in enumerate(dataset):
            images = data[0].to(self.device)  # B 104
            labels = data[1].to(self.device)
            with torch.no_grad():
                prediction = self.net(images)

                p1 = prediction
                p1 = torch.where(p1 >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
                l1 = labels

                temp_onlylable = torch.cat([l1, p1], dim=-1)
                self.csv_onlylable.append(temp_onlylable.cpu().detach().numpy().squeeze())


        duration = time.time() - star_time
        speed = 1 / (duration / len(dataset))
        print('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('-----------------start test--------------------')
        file_handle.write('\n')
        file_handle.write('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle.write('\n')
        file_handle.close()

        self.net = self.net.train()
        self.tocsv_onlylable(epoch)

        print('-----------------test over--------------------')





    def tocsv_onlylable(self, epoch):

        np_data = np.array(self.csv_onlylable)

        label = np_data[:, :6]
        pred = np_data[:, 6:]

        accurate_samples = np.all(pred == label, axis=1)
        accuracy = np.mean(accurate_samples)

        label_first_three = np_data[:, :3]  # First three
        pred_first_three = np_data[:, 6:9]  # Predicted first three

        label_last_three = np_data[:, 3:6]
        pred_last_three = np_data[:, 9:]

        # Accuracy of the predicted first three
        accurate_samples_first_three = np.all(pred_first_three == label_first_three, axis=1)
        accuracy_first_three = np.mean(accurate_samples_first_three)

        # Accuracy of the predicted second three
        accurate_samples_last_three = np.all(pred_last_three == label_last_three, axis=1)
        accuracy_last_three = np.mean(accurate_samples_last_three)

        # # Metrics
        epsilon = 1e-7
        pred_epsilon = np.clip(pred, epsilon, 1 - epsilon)
        bce_loss = -(label * np.log(pred_epsilon) + (1 - label) * np.log(1 - pred_epsilon))
        bce_loss = bce_loss.mean()
        #
        #
        print(
            'tets_bce_loss:{}'.format(bce_loss)
        )

        print(
            'accuracy:{}'.format(accuracy)
        )

        print(
            'accuracy_first_three:{}'.format(accuracy_first_three)
        )

        print(
            'accuracy_last_three:{}'.format(accuracy_last_three)
        )

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('bce_loss:{}'.format(bce_loss))
        file_handle.write('\n')


        file_handle.write('accuracy:{}'.format(accuracy))
        file_handle.write('\n')

        file_handle.write('accuracy_first_three:{}'.format(accuracy_first_three))
        file_handle.write('\n')

        file_handle.write('accuracy_last_three:{}'.format(accuracy_last_three))
        file_handle.write('\n')

        file_handle.write('-----------------test_over--------------------')
        file_handle.write('\n')
        file_handle.close()
        np.savetxt(self.save_path + 'pred_results.csv', np_data, delimiter=',')
