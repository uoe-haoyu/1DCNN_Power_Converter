from dataset import get_pathdata
import model_repo
import test
if __name__ == '__main__':
    testdata_path = r'./wval_data\N=4\test_norm.csv'
    save_path = r'./predict\\'  # save predicted results
    pth = r'./trained_model/N=4/CNN_N4.pth' # used traiend model
    netname = model_repo.CNN_1D                       # used model

    config = {
        'netname': netname.Net,
        'dataset': {'test': get_pathdata(testdata_path),},
        'pth_repo': pth,
        'test_path': save_path,
    }
    tester = test.Test(config)
    tester.start()
