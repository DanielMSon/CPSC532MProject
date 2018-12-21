import pandas as pd
# sklearn packages for base models other than NN
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle

from utils import evaluate_model
import matplotlib.pyplot as plt
import os

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
        l1 = 128
        l2 = 256
        l3 = 256
        l4 = 256

        self.fc1 = nn.Linear(num_features, l1)

        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, l4)
        
        self.fc_out = nn.Linear(l4, 1)     # keep the last layer before output helps reducing the occurace of NaN

    def forward(self, x):
        """ feedforward 
        
        Arguments:
            x {tensor} -- input
        
        Returns:
            {tensor} -- output
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)

        return x

class NeuralNetRegressor():
    def __init__(self, num_features, lr=0.01, momentum=0.9, lammy=1e-5,
                batch_size=4, shuffle_data=True, num_workers=2, 
                gpu=False, verbose=True, epochs=2):
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
        self.lammy = lammy

        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        # self.criterion = nn.MSELoss()   # loss function, mean-squared-error
        self.criterion = nn.L1Loss()

    def fit(self, X_train, y_train):
        """traing neural network
        
        Arguments:
            X_train {ndarray} -- X training
            y_train {ndarray} -- y training
        
        Keyword Arguments:
            epochs {int} -- how many epochs (default: {2})
        """
        n, d = X_train.shape
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


        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.lammy)
        
        # training
        best_running_loss = None
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
                params = torch.cat([p.view(-1) for p in self.net.parameters()])
                self.loss += self.lammy * torch.norm(params, p=1)     # L1 reg
                self.loss.backward()
                self.optimizer.step()  

                # print stat
                running_loss += self.loss.item()

                # if self.verbose == True: # and i % 40 == 39:
                #     print('[%d, %5d] loss: %.5g' %
                #         (epoch + 1, i + 1, running_loss))
                #     running_loss = 0.0

            if self.verbose == True:
                print('[epoch %d] loss: %.5g' %
                        (epoch + 1, running_loss/n))
            # early stopping
            # if best_running_loss is None:
            #     best_running_loss = running_loss
            # elif running_loss < best_running_loss:
            #     best_running_loss = running_loss
            # elif running_loss > 1.3*best_running_loss:
            #     print("Early stopping at epoch #: {}, running_loss={}".format(epoch+1, running_loss))
            #     break
        
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


class AveragingRegressor():
    """ a simple averaging ensemble method. Take mean of base models' predictions as final prediction """

    def __init__(self, models):
        """ constructor
        
        Arguments:
            models {list} -- list of base models
        """

        self.models = models

    def fit(self, X_train, y_train):
        """ train averaging model
        
        Arguments:
            X_train {ndarray} -- X training
            y_train {ndarray} -- y training
        """

        for model in self.models:
            model.fit(X_train, y_train)

    def predict(self, X):
        """ predict by averaging the prediction of base models
        
        Arguments:
            X {ndarray} -- X
        
        Returns:
            ndarray -- predictions
        """

        yhats = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(yhats, axis=1)


class StackingRegressor():
    """ a simple ensemble stacking class """

    def __init__(self, base_models, meta_model):
        """ constructor
        
        Arguments:
            base_models {list} -- list of base models
            meta_model {model} -- meta model which use predictions of base model to predict
        """

        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X_train, y_train):
        """ train stacking regressor
        
        Arguments:
            X_train {ndarray} -- X training
            y_train {ndarray} -- y training
        """

        self._base_model_fit(X_train, y_train)
        Z_train = self._base_model_predict(X_train)

        self.meta_model.fit(Z_train, y_train)

    def predict(self, X):
        """ make prediction with stacking method
        
        Arguments:
            X {ndarray} -- X test
        
        Returns:
            {ndarray} -- predictions
        """

        Z = self._base_model_predict(X)
        yhats = self.meta_model.predict(Z)
        
        return yhats

    def _base_model_fit(self, X_train, y_train):
        """ helper function to train the base models
        
        Arguments:
            X_train {ndarray} -- X training
            y_train {ndarray} -- y training
        """

        for model in self.base_models:
            model.fit(X_train, y_train)

    def _base_model_predict(self, X):
        """ helper function to get prediction from base models
        
        Arguments:
            X {ndarray} -- X test
        
        Returns:
            {ndarray} -- predictions
        """

        Z = np.column_stack([model.predict(X) for model in self.base_models])
        return Z

def neuralNetHyperparamTuning(  hyperparam, hyperparam_vals, X, y, 
                                lr=1e-3, momentum=0.9, lammy=1e-5, batch_size=32, 
                                epochs=100, num_workers=4, err_type='squared', 
                                cross_val=False, valid_size=0.2, n_splits=5, 
                                save_fig=True):
    """ tune NN hyperparameter by varying the specified hyperparameter using the values provided 
    in hyperparam_vals. The specified tuning hyperparameter will only take values from the list (hyperparam_vals),
    and the value from the corresponding keyword arg will not be used
    
    Arguments:
        hyperparam {str} -- string, hyper-parameter to tune {'lr', 'momentum', 'lammy', 'batch_size'}
        hyperparam_vals {list-like, ndarray} -- list of values to tune the hyperparam
        X {ndarray} -- X
        y {ndarray} -- y
    
    Keyword Arguments:
        lr {float} -- learning rate (default: {1e-3})
        momentum {float} -- momentum (default: {0.9})
        lammy {float} -- lamda for regularization (default: {1e-5})
        batch_size {int} -- mini batch size (default: {32})
        epochs {int} -- epochs (default: {100})
        num_workers {int} -- num of sub process for memory transfer (default: {4})
        err_type {str} -- type of error for evaluation (not during training) 'abs', 'squared', 'rmsle' (default: {'squared'})
        cross_val {bool} -- whether to use cross validation (default: {False})
        valid_size {float} -- 0.0 to 1.0, portion of validation set (default: {0.2})
        n_splits {int} -- how many portions are data split into for cross validation. Only used if cross-val is True (default: {5})
        save_fig {bool} -- whether to save figure (default: {True})
    
    Raises:
        NameError -- Wrong hyperparam name

    Returns:
        {ndarray, ndarray} -- training errors, validation errors
    """

    _, num_features = X.shape
    
    num_divs = len(hyperparam_vals)

    # init errors
    errs_tr = np.empty(num_divs)
    errs_va = np.empty(num_divs)

    for i in range(num_divs):
        print("[{}] {} = {}".format(str(i), str(hyperparam), str(hyperparam_vals[i])))
        # randomize random_state seed
        random_state = np.random.randint(0, high=100)
        print("random seed generated {}".format(random_state))

        if hyperparam == 'lr':
            model = NeuralNetRegressor(
                num_features, gpu=True, lr=hyperparam_vals[i], momentum=momentum, lammy=lammy, 
                batch_size=batch_size, epochs=epochs, num_workers=num_workers, 
                verbose=False
            )
        elif hyperparam == 'momentum':
            model = NeuralNetRegressor(
                num_features, gpu=True, lr=lr, momentum=hyperparam_vals[i], lammy=lammy, 
                batch_size=batch_size, epochs=epochs, num_workers=num_workers, 
                verbose=False
            )
        elif hyperparam == 'lammy':
            model = NeuralNetRegressor(
                num_features, gpu=True, lr=lr, momentum=momentum, lammy=hyperparam_vals[i], 
                batch_size=batch_size, epochs=epochs, num_workers=num_workers, 
                verbose=False
            )
        elif hyperparam == 'batch_size':
            batch_size = int(hyperparam_vals[i])
            model = NeuralNetRegressor(
                num_features, gpu=True, lr=lr, momentum=momentum, lammy=lammy, 
                batch_size=batch_size, epochs=epochs, num_workers=num_workers, 
                verbose=False
            )
        else:
            raise NameError("Hyperparam not found")

        errs_tr[i], errs_va[i] = evaluate_model(
            model, X, y, valid_size=valid_size, verbose=True, cross_val=cross_val, 
            n_splits=n_splits, random_state=random_state, err_type=err_type)

    if save_fig:
        plt.figure()
        plt.plot(hyperparam_vals, errs_tr, label="training errors")
        plt.plot(hyperparam_vals, errs_va, label='validation errors')
        plt.xlabel('{}'.format(hyperparam))
        plt.ylabel('mean [{}] errors'.format(err_type))
        plt.legend()
        plt.grid()
        plt.title('Hyperparam tuning: {}'.format(hyperparam))
        fname = os.path.join('..', 'figs', '{}_{}_err.png'.format(hyperparam, str(err_type)))
        plt.savefig(fname)
        # plt.show(block=False)

    return errs_tr, errs_va

if __name__ == "__main__":
    df = pd.pandas.read_csv("train.csv")
    df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])

    #Initialize ChiSquare Class
    cT = ChiSquare(df)

    #Feature Selection
    testColumns = ['Embarked','Cabin','Pclass','Age','Name','dummyCat']
    for var in testColumns:
        cT.TestIndependence(colX=var,colY="Survived" )  