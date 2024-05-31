import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.utils import accuracy_fn as accuracy

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.
    It should not use any convolutional layers.
    """
    
    def __init__(self, input_size, n_classes, hidden_size1=50, hidden_size2=100):
        """
        Initialize the network.
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            hidden_size (int): number of units in the hidden layer
        """
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size1
        self.hidden_size_2 = hidden_size2
        self.n_classes = n_classes
        
        # Initialize weights and biases for each layer
        self.w1 = nn.Parameter(torch.randn(input_size, hidden_size1) / torch.sqrt(torch.tensor(input_size, dtype=torch.float32)))
        self.b1 = nn.Parameter(torch.zeros(hidden_size1))
        self.w2 = nn.Parameter(torch.randn(hidden_size1, hidden_size2) / torch.sqrt(torch.tensor(hidden_size1, dtype=torch.float32)))
        self.b2 = nn.Parameter(torch.zeros(hidden_size2))
        self.w3 = nn.Parameter(torch.randn(hidden_size2, n_classes) / torch.sqrt(torch.tensor(hidden_size2, dtype=torch.float32)))
        self.b3 = nn.Parameter(torch.zeros(n_classes))
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.
        
        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # First layer
        z1 = x @ self.w1 + self.b1  # Linear transformation
        a1 = F.relu(z1)  # Activation function

        # Second layer
        z2 = a1 @ self.w2 + self.b2  # Linear transformation
        a2 = F.sigmoid(z2)  # Activation function
        
        # Output layer
        z3 = a2 @ self.w3 + self.b3  # Linear transformation
        
        return z3
        

class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv2d1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv2d3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv2d4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)


        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)


        # Dropout layer
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.bn1(self.conv2d1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2d2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv2d3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv2d4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        return x



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
                self.output_mapping = nn.Linear(d, d)

            def forward(self, sequences):
                """
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
            """
                batch_size, seq_len, d = sequences.shape
                sequences = sequences.view(batch_size, seq_len, self.n_heads, self.d_head)
                sequences = sequences.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_head)

                q = torch.cat([mapping(sequences[:, i]) for i, mapping in enumerate(self.q_mappings)], dim=2)
                k = torch.cat([mapping(sequences[:, i]) for i, mapping in enumerate(self.k_mappings)], dim=2)
                v = torch.cat([mapping(sequences[:, i]) for i, mapping in enumerate(self.v_mappings)], dim=2)

                attention_scores = self.softmax(torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_head))
                attention_output = torch.matmul(attention_scores, v)

                attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_len, d)
                return self.output_mapping(attention_output)
            
        class MyViTBlock(nn.Module):
            def __init__(self, hidden_d, n_heads, mlp_ratio=4, dropout_rate = 0.1):
                super(MyViTBlock, self).__init__()
                self.hidden_d = hidden_d
                self.n_heads = n_heads

                self.norm1 = nn.LayerNorm(hidden_d) 
                self.mhsa = MyMSA(hidden_d, n_heads)
                self.dropout1 = nn.Dropout(dropout_rate)   #TO CHECK    
                
                self.norm2 = nn.LayerNorm(hidden_d)  
                self.mlp = nn.Sequential( 
                    nn.Linear(hidden_d, mlp_ratio * hidden_d),
                    nn.GELU(),
                    nn.Linear(mlp_ratio * hidden_d, hidden_d),
                    nn.Dropout(dropout_rate)   #TO CHECK
                )

            def forward(self, x):
                out = x + self.dropout1(self.mhsa(self.norm1(x)))
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
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches) 

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        self.conv = nn.Conv2d(chw[0], hidden_d, kernel_size=self.patch_size, stride=self.patch_size)  #TO CHECK

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, 1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = nn.Parameter(get_positional_embeddings(n_patches ** 2 + 1, hidden_d)) #TO CHECK

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_d),
            nn.Linear(self.hidden_d, out_d),
            #nn.Softmax(dim=-1)
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
            x = x.view(-1, 1, 28, 28)
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

        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        
        
        # Divide images into patches.
        #patches = patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        #tokens = self.linear_mapper(patches)

        class_tokens = self.class_token.expand(n, 1, -1) 
        
        # Add classification token to the tokens.
        x = torch.cat((class_tokens, x), dim=1)
        

        # Add positional embedding.
        #preds = tokens + self.positional_embeddings.repeat(n,1,1)

        x = x + self.positional_embeddings
        
        # Transformer Blocks
        for block in self.blocks:
            #preds = block(preds)
            x = block(x)

        # Get the classification token only.
        x = x[:, 0]

        # Map to the output distribution.
        preds = self.mlp_head(x)
        
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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')

        self.best_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0

    def train_all(self, dataloader, val_dataloader):
        """
        Fully train the model over the epochs. 

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)
            val_loss = self.validate_one_epoch(val_dataloader)
            print(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print("Early stopping triggered")
                break

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
        for (it, batch) in enumerate(dataloader):
            # print(batch[0].shape, batch[1].shape)
            self.optimizer.zero_grad()   #APPARENTLY BETTER TO PUT IT AT THE BEGINNING OF THE LOOP
            x, y = batch[0], batch[1]
            y = y.long()
            x = x.view(-1, 1, 28, 28)
            logit = self.model(x)
            loss = self.criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            #self.writer.add_scalar('Loss/train', loss, ep * len(dataloader) + it)
            self.optimizer.step()
            
            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}'.format(ep + 1, self.epochs, it + 1, len(dataloader), loss), end='')
        #self.writer.flush()


    def softmax(self, x):
        """
        Compute the softmax of a batch of scores.

        Arguments:
            x (tensor): input batch of shape (N, C)
        Returns:
            (tensor): softmax of the input batch of shape (N, C)
        """
        return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

    def validate_one_epoch(self, dataloader):
        """
        Validate the model for ONE epoch.

        Arguments:
            dataloader (DataLoader): dataloader for validation data
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0], batch[1]
                y = y.long()
                x = x.view(-1, 1, 28, 28)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                val_loss += loss.item()
        val_loss /= len(dataloader)
        #self.writer.add_scalar('Loss/val', val_loss)
        return val_loss

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
                x = batch[0]
                x = x.view(-1, 1, 28, 28)
                pred = self.model(x)
                #pred = self.softmax(pred) #TO CHECK, APPARENTLY IT IS NOT NEEDED BECAUSE  nn.CrossEntropyLoss() ALREADY DOES IT
                pred_labels.append(pred.argmax(dim=1))
        return torch.cat(pred_labels)

    def fit(self, training_data, training_labels, validation_data, validation_labels):
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

        # Print the dimensions of the tensors in the TensorDataset
        # print("Training data dimensions: ", train_dataset.tensors[0].size())
        # print("Training labels dimensions: ", train_dataset.tensors[1].size())

        # Print the batch size of the DataLoader
        # print("DataLoader batch size: ", train_dataloader.batch_size)

        # Prepare validation data for pytorch
        val_dataset = TensorDataset(torch.from_numpy(validation_data).float(),
                                    torch.from_numpy(validation_labels))
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_all(train_dataloader, val_dataloader)

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