import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()
        #### WRITE YOUR CODE HERE!
        self.conv2d1 = nn.Conv2d(input_channels, 6, 3, padding=1)
        self.conv2d2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #### WRITE YOUR CODE HERE!
        #x = torch.from_numpy(x)
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = x.reshape((x.shape[0], -1))
        x = F.relu(torch.Tensor(self.fc1(x)))
        x = F.relu(torch.Tensor(self.fc2(x)))
        return self.fc3(x)


class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super(MyViT, self).__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        def get_positional_embeddings(sequence_length, d):
            result = torch.ones(sequence_length, d)
            for i in range(sequence_length):
                for j in range(d):
                    if j % 2 == 0:
                        result[i,j] = np.sin(i / (10000 ** ((2 * j) / d)))
                    else:
                        result[i,j] = np.cos(i / (10000 ** ((2 * j) / d)))

            return result
        
        class MyMSA(nn.Module):
            def __init__(self, d, n_heads=2):
                super(MyMSA, self).__init__()
                self.d = d
                self.n_heads = n_heads

                assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
                d_head = int(d / n_heads)
                self.d_head = d_head

                self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
                self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
                self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

                self.softmax = nn.Softmax(dim=-1)

            def forward(self, sequences):
                result = []
                for sequence in sequences:
                    seq_result = []
                    for head in range(self.n_heads):

                        # Select the mapping associated to the given head.
                        q_mapping = self.q_mappings[head]
                        k_mapping = self.k_mappings[head]
                        v_mapping = self.v_mappings[head]

                        seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                        # Map seq to q, k, v.
                        q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                        m = nn.Softmax(dim=1)
                        attention =  self.softmax((q @ k.T) / np.sqrt(self.d_head))
                        seq_result.append(attention @ v)
                    result.append(torch.hstack(seq_result))
                return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
            
            
        class MyViTBlock(nn.Module):
            def __init__(self, hidden_d, n_heads, mlp_ratio=4):
                super(MyViTBlock, self).__init__()
                self.hidden_d = hidden_d
                self.n_heads = n_heads

                self.norm1 = nn.LayerNorm(hidden_d) 
                self.mhsa = MyMSA(hidden_d, n_heads)
                self.norm2 = nn.LayerNorm(hidden_d)  
                self.mlp = nn.Sequential( 
                    nn.Linear(hidden_d, mlp_ratio * hidden_d),
                    nn.GELU(),
                    nn.Linear(mlp_ratio * hidden_d, hidden_d)
                )

            def forward(self, x):
                out = x + self.mhsa(self.norm1(x))
                out = out + self.mlp(self.norm2(out))
                return out

        self.chw = chw 
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) 

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = get_positional_embeddings(n_patches ** 2 + 1, hidden_d) ### WRITE YOUR CODE HERE

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        
        def patchify(images, n_patches):
            n, c, h, w = images.shape

            assert h == w # Only square images are supported


            patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
            patch_size = h // n_patches

            for i, image in enumerate(images):
                for j in range(n_patches):
                    for k in range(n_patches):
                        temp = image[:, j * patch_size : (j+1) * patch_size,  k * patch_size : (k+1) * patch_size]
                        patches[i, j * n_patches + k] = temp.flatten()

            return patches
        
        n, c, h, w = x.shape

        # Divide images into patches.
        patches = patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        preds = tokens + self.positional_embeddings.repeat(n,1,1)

        # Transformer Blocks
        for block in self.blocks:
            preds = block(preds)

        # Get the classification token only.
        preds = preds[:, 0]

        # Map to the output distribution.
        preds = self.mlp(preds)
        
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        #### WRITE YOUR CODE HERE!
        self.model.train()
        for it, batch in enumerate(dataloader):
            x, y = batch
            y = y.long()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        #### WRITE YOUR CODE HERE!
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch
                #print(len(batch))
                pred = self.model(x)
                pred_labels.append(pred.argmax(dim=1))
        return torch.cat(pred_labels)
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
