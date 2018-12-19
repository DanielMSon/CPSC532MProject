import pandas as pd
# sklearn packages for base models other than NN
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle

# use pytorch to implement neural network
try:
    import torch
    import torch.nn as nn 
    import torch.nn.functional as F 
    import torch.utils.data
    import torchvision
    import torch.optim as optim
except ImportError:
    print("PyTorch not imported")
    pass

# scientific calculation tools
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

class Net(nn.Module):
    """ 
    A simple regression neural network. The hyperparameters are arbitraty as of now.
    TODO: tune hyper parameter
    """

    def __init__(self, num_features):
        super(Net, self).__init__()
        # self.input_feats = input_feats

        # input layer, 1 hidden layer, output layer
        l1 = 37
        l2 = 5  

        self.fc1 = nn.Linear(num_features, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 1)     # keep the last layer before output helps reducing the occurace of NaN

    def forward(self, x):
        """ feedforward 
        
        Arguments:
            x {tensor} -- input
        
        Returns:
            {tensor} -- output
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class NeuralNetRegressor():
    def __init__(self, num_features, lr=0.01, momentum=0.9, 
                batch_size=4, shuffle_data=True, num_workers=2, 
                gpu=False, verbose=False, epochs=2):
        """ constructor for NN regressor
        
        Arguments:
            num_features {int} -- d, the number of features for an example
        
        Keyword Arguments:
            lr {float} -- learning rate (default: {0.01})
            momentum {float} -- momentum (default: {0.9})
            batch_size {int} -- mini-batch size. Torch will always go with a batch when SGD is used (default: {4})
            shuffle_data {bool} -- shuffle data in SGD (default: {True})
            num_workers {int} -- how many subprocess used to dataloading (default: {2})
            gpu {bool} -- whether to use gpu for training (default: {False})
            verbose {bool} -- whether to print out running loss (default: {False})
            epochs {int} -- how many epochs the training runs (defatult: {2})
        """

        # self.num_features = num_features
        self.net = Net(num_features)
        self.lr = lr    # learning rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.num_workers = num_workers
        self.gpu = gpu
        self.verbose = verbose
        self.epochs = epochs

        if torch.cuda.is_available():
            self.device = torch.device('cuda')

    def fit(self, X_train, y_train):
        """traing neural network
        
        Arguments:
            X_train {ndarray} -- X training
            y_train {ndarray} -- y training
        
        Keyword Arguments:
            epochs {int} -- how many epochs (default: {2})
        """

        # convert to tensors
        X_train = torch.from_numpy(X_train).float() 
        y_train = torch.from_numpy(y_train).unsqueeze(1).float()

        # convert to dataloader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=self.shuffle_data, num_workers=self.num_workers
        )
        
        # transfer net to GPU if available
        if self.gpu == True:
            try:
                self.net.to(self.device)
            except:
                print("Cuda device not found. Run on CPU")


        self.criterion = nn.MSELoss()   # loss function, mean-squared-error
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        # training
        for epoch in range(self.epochs):
            
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                X_tr, y_tr = data
                
                # transfer to GPU if available
                if self.gpu == True:
                    try:
                        X_tr, y_tr = X_tr.to(self.device), y_tr.to(self.device)
                    except:
                        print("Cuda device not found. Run on CPU")

                # zero param gradient
                self.optimizer.zero_grad() 

                # forward + backward + optimize
                y_pred = self.net(X_tr)
                self.loss = self.criterion(y_pred, y_tr)
                self.loss.backward()
                self.optimizer.step()  

                # print stat
                running_loss += self.loss.item()

                if self.verbose == True and i % 200 == 199:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        
        print("Finished training")

    def predict(self, X):
        """predict with trained NN
        
        Arguments:
            X {ndarray} -- X test
        
        Returns:
            yhats {ndarray} -- predictions
        """

        yhats = torch.tensor([])
        if self.gpu == True:
            try:
                yhats = yhats.to(self.device)
            except:
                print("Cuda device not found. Run on CPU")

        X_test = torch.from_numpy(X).float()
        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=self.shuffle_data, num_workers=self.num_workers
        )

        # predict without tracking gradient
        with torch.no_grad():
            for data in test_loader:
                [x_te] = data 

                if self.gpu == True:
                    try:
                        x_te = x_te.to(self.device)
                    except:
                        print("Cuda device not found. Run on CPU")
                # tranfer pred to gpu if available 

                yhat = self.net(x_te)                
                yhats = torch.cat((yhats, yhat), 0)

        if self.gpu == True:
            try:
                yhats = yhats.to(torch.device('cpu'))
            except:
                print("Cuda device not found. Run on CPU")

        yhats = yhats.numpy()
        return yhats

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


if __name__ == "__main__":
    df = pd.pandas.read_csv("train.csv")
    df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])

    #Initialize ChiSquare Class
    cT = ChiSquare(df)

    #Feature Selection
    testColumns = ['Embarked','Cabin','Pclass','Age','Name','dummyCat']
    for var in testColumns:
        cT.TestIndependence(colX=var,colY="Survived" )  