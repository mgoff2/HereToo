### Neural Networks, Language Embedders, Autocoding-- Oh, my!

In the aforementioned-then-subsequently-ignored interest of brevity, I've tried to strip these example builds down to only what's necessary to grasp the implementation strategy; all of the reading-in of files, mapping of labels, importing of modules, and doing of the essential-but-obvious-stuff is excised. I've linked, however, to related models, notebooks, articles, and so on, so building and customizing -- if that's of interest-- shouldn't be too hard. Likewise, the outputs, too, are just noted briefly or shown at the bottom, since each of these takes >45 minutes to process on a Kaggle GPU.

Preliminary Concepts I won't get too in the weeds labeling and describing these things-- in no small part because there are many parts that I am myself quite hazy about, at least in terms of 'why-this-not-that' and 'what-difference-does-it-make.' This isn't a textbook on neural networks, and I wouldn't try to author one. If you're completely unfamiliar with deep learning/neural networks, there is an exhaustive universe of free resources, which I'll leave to you, dear reader, to explore. A paid resource that I found very helpful is here: [Link](https://www.amazon.com/Deep-Learning-Approach-Andrew-Glassner/dp/1718500726/ref=pd_bxgy_sccl_1/131-4168850-7449845?pd_rd_w=LWxP9&content-id=amzn1.sym.7757a8b5-874e-4a67-9d85-54ed32f01737&pf_rd_p=7757a8b5-874e-4a67-9d85-54ed32f01737&pf_rd_r=SDWF9ERCTD48CSTV0RR8&pd_rd_wg=iNJtJ&pd_rd_r=53d0eb02-2c69-4e0d-b4ae-08f002af1f0d&pd_rd_i=1718500726&psc=1). Briefly, I'll touch on some of the code-elements that might be unfamiliar. All of these language-processing neural nets have (in model-appropriate variations) the following:

If you're completely unfamiliar with neural networks, very little of what follows will make sense. But for context, a neural network is a configuration of nodes grouped into layers. These nodes contribute weights to parts of the input data, which are cumulatively fed to nodes in subsequent layers. The subsequent layer's nodes activate if and only if the input from the previous layer passes a certain threshold. This ultimately arrives at an output layer made of -- at least for classification tasks -- as many nodes as there are classes in the dependent variable. The activation at this level is the model's prediction. Other layers have other functions, the input layer, for example, just recieves the data; a 'dense' layer is a replicate layer that in which each node recieves input from all nodes in the previous layer; and so on.

### Model Framework

In order for neural networks to work, they have to execute *A Lot* of calculations as the number of weights to be learned increase exponentially with each additional layer in the model. Note that when I say 'a lot,' I mean a lot even for computers. Thus configuring the data types and optimizing their implementation on various hardward components requires a good bit of computer engineering know-how. Thankfully, there are open-source libraries that facilitate all of this for us. The two biggest players in this space are indisputably Tensorflow and Torch. Tensorflow is usually used through Keras' API, while Pytorch is a Python-friendly Torch spinoff that kind of does it all. Since both TF and Pytorch utilize tensors and/or .tsv files, there are methods employed in the following demos that may look a bit unfamiliar; largely these are processes converting the dataset into TF/Torch-amenable formats. Likewise, since these require sequences of a common length, all of the models include some varations of the following:

- Max length: the maximum size of any given input text
- Truncation: rules specifying to cut off the ends of any overly long texts.
- Padding: rules specifying 'pad' tokens to append to texts that aren't *sufficiently* long.
- Data Loader: wraps an iterable around the prepped dataset to facilitate access to the contents. It is responsible for selection of items that go into a given batch.
- Dataset: a torch class used to define custom datasets
- Dict(): We frequently have to 'return_dict' or 'dict(list())' and so on because of the way organize our dataframes into tensors.
- .tsv: tab-separated values. As with dict() and dataloader and all, this is invoked to create the tensors needed for these models and the GPUs they run on.
- Return Tensors = 'pt': make Pytorch-amenable tensors.
- 'tf' + anything: TensorFlow
- Def forward(): A function defining how the model should proceed.

You'll also see:

- model.train()/.eval() -- setting the model to training/evaluation mode to inform whether or not it should update. When in eval, the inputs are read-only.
- with __ no_grad() -- set during evaluation to instruct the model not to update the gradients
- metrics-- library of evaluation methods for use with neural networks.
- nn -- to instantiate a neural network from a base class.
- tensor -- to build a tensor data matrix.
- Autotune -- an algorithm that optimizes processor allocation.
- Verbose -- specifies how much of the training progress/information to make available to the analyst.

### CUDA 

Everything related to CUDA deals with configurations vis-a-vis GPU processing. If run on Kaggle, you can use their GPUs, otherwise you can a) buy one, b) use one you didn't know you had (!), c) use another host like Google Colab or Paperspace Gradient, or d) go harangue some wealthier fellow data scientist and convince them to let you 'borrow' theirs. Should you decide to try and implement any configuration on your own device, be forewarned: it is *not* (in the author's humble opinion) a straightforward, intuitive project...

- Set Seed -- used to make the model reproducible despite using randomly generated values in training
- to_device -- defining a variable 'device' as either the GPU or CPU, to_device ensures that model and data are sent to the same place.
- n_jobs -- specifies how many core processors to allocate to the process. There are plenty of great resources out there for learning deeply about deep learning. On the Tensorflow-Keras-(Py)Torch topic, here's a good resource (Tip o' the Hat to Dr. Rogel-Salazar): [Link.](https://www.dominodatalab.com/blog/tensorflow-pytorch-or-keras-for-deep-learning)

### Hyperparameters and User Specifications

Model implementation parameters: whatever kind of model we pass, the following helper functions and hyperparameters must be specified:

- Number-of-Epochs -- how many times we fully pass through the dataset in training
- Batch Size -- how many text-items to process at a time; more will speed-up training, but can easily exhast memory.
- Iteration -- a full forward-and-backward pass through a batch, equal to N/batch-size. Each time data passes forwards, it is analyzed and predicted; based on its performance, the backward-pass (aka 'backpropagation') is where the weights are adjusted based on the model's performance. If you've heard of the gradient descent algorithm, the backward pass is when it's implemented, essentially sending feedback through the network, and moving the weights in such a way as to reduce the loss contributed to the model. (Sometimes called a 'gradient update.')
- Loss Function -- basically what metric we are using to 'score' how well the model is doing. At different weights, the loss (error-- the output of the loss function given that particular weight) of the model goes up or down. We want the loss to go down as much as possible ('descent').
- Objective/Objective Function -- this is the function that the model is attempting to optimize. If a loss-function is defined, the loss function is the objective; if, on the other hand, the objective is accuracy, then we will attempt to *maximize* it.
- Learning Rate -- In seeking to minimize loss, the model 'descends' down the gradient of the loss function, by taking 'steps.' The size of the steps it takes is the model's learning rate. If the learning rate is very high, the model takes giant leaps in reconfiguration, and doesn't steadily descend. If the learning rate is very low, the model is so cautious that it doesn't make adequate progress.
  - Momentum: Specification of the *direction* the optimizer should head. 
  - Warmup Steps: Specification of some number of steps in which the learning/updating is inactive. Sort of a 'let-the-model-get-its-bearings' period preceding the gradient updating.
  - Decay/Weight Decay: for modifying the learning rate -- the 'step-size' -- as we get further along in the analysis. Initially, we may want to take rather large steps, since we began haphazardly. As the model improves, however, we want to slow down and take more cautious steps, since we've likely made progress towards the minima of the loss function.
  - Learning Rate Scheduler: defines trigger to implement learning rate decay.
- Regularization: A range of tools used to limit the model's capacity to overfit. Generally, these take the form of some sort of penalty for added complexity. We want the model to balance *bias* and *variance,* meaning we want the model to have sufficient complexity to make usable predictions, but not *such* model-specific detail that it treats anything not in the training set as a non-positive observation. Popular tools in this respect are:
  - L1 (aka Lasso): Imposition of a penalty on less-informative features, which can include sending the features all the way to zero (i.e., eliminating them from the model).
  - L2 (aka Ridge): Like L1, but never sending any feature's coefficient to zero. This makes the model more accurate, while still giving a conditional/contextual sense of the importance of individual features.
  - Elastic Net: a combination of L1 and L2.
  - Early Stopping: Saving the weight configurations from each separate epoch, such that if continued adjustment produces diminishing returns, we can simply deploy an earlier version of the network.
  - Dropout: Randomly excluding some observations from exposure to the model in training. This forces the model to assign weights with fewer input features in mind, making the model more generalizable.
  - Batch Normalization: transforming the outputs of a given batch such that they are normally distributed.
  - Label-Smoothing: A noise-introducing technique that deliberately misshapes the inputs to the final layer so that the model doesn't become 'overconfident' in its predictions at an early stage.
- Activation Function -- a function that defines the threshold a previous layer's inputs must meet in order for a (subsequent layer's) node to activate. The Rectified Linear Unit ('ReLU') is very common-- it returns 0 if the input is <=0, and 1 otherwise, where 0 = 'don't activate' and 1 = 'activate.'
- Optimizer / Optimization Function -- defines the specific implementation of the gradient descent algorithm to use. 'Adam' -- Adaptive Momentum -- is very common.
- Criterion -- the function used to define boundary thresholds a prediction must meet in order for an item to be predicted a member of class X.
- Entropy -- refers to the extent of heterogeneity within any relevant subset of the data. Since informative features are those that *distinguish* class A from class B, we want less entropy (i.e., we don't want to evaluate on the basis of features members of class A and class B have in common). If I'm describing a suspect to a police sketch artist, telling him that the person was between 3 and 6 feet tall, had two arms, and was covered in skin isn't useful, since literally every human meets fits that description. There would be, in that case, a lot of entropy between the classes of 'The perpetrator' and 'Not the perpetrator.'
  - Categorical Cross-Entropy -- a criterion function that's based on the degree of entropy between the different classes, in non-binary classification tasks. (Class *C* ought to be as dissimilar as possible from classes A *and* B.)

- Shuffle -- to remix the batch items after each epoch. Since a random drawing of items may not be representative of the dataset as a whole, re-feeding the same sets over and over could lead to overfitting.
- Step/Step-Size -- equivalent to 'iteration' and 'learning rate,' respectively.
- Argmax: For a function, f, and a set of inputs, X, the argmax is the member of set X, x_i, that produces the greatest output ('y'-value) of all the items in set X.
- Softmax: an activation function that is essentially the extension of the sigmoid to multi-class, multi-dimensional problems.
- Logits -- the log-probabilities output by the model; the softmax function takes these as inputs and transforms them such that they sum to one.
- Normalization: sending all of the values to a predetermined range (say, -1 to 1) via an information-preserving transformation.
- Standardization: sending all of the values to a standard normal distribution.

### Model Details

The specific models themselves, listed below, could each be the subject of entire books, and I won't attempt to explain them in any detail here. Briefly:

- Recurrent Neural Network (RNN): a NN that retains a node's 'state' in memory for later use, then 'forgets' it at a subsequent step. This type of network is used for sequence-processing, as item N contributes to the meaningfulness of item N+1, and so on. More below. -LSTM: Long Short-Term Memory. A specific sub-type RNN.
- Convolutional Neural Network (CNN): It's a bit of a challenge to concisely explain anything useful about CNNs. One analogy I found helpful is to think of putting a page from a book under a microscope. The narrowness of the lens requires iteratively sliding the page backwards and forwards, up and down, to read the whole text. In a convolutional NN, the nodes on the convolutional layers scan the input by sliding it back and forth like the book's pages. These nodes are pattern detecting filters: just as what we see through a microscope is a 'filtered' image (in that it is magnified considerably), the nodes are seeing through *their* filters. These filters overlay a patterned outline, and activate if the input matches the pattern they are looking for. In this way, composition, simplification, and complexity can all be improved.
  - Stride: the size of the 'slides' that the nodes make (in terms of square units).
    - Pooling/Pooled Inputs, Outputs: When a convolutional layer matches a pattern, it can indicate a 'details-invariant' match -- an indication that it has matched the pattern, even if the pattern is idiosyncratically distinct from other matches. In this way, it can 'downsample' the features, essentially simplifying its result to Yes or No. Analogically, when you search for something on Amazon, say, then apply a filter to limit results to a certain brand, there will still be a diversity of items returned, but all will 'match' with the brand. A pooling layer summarizes features into matches and non-matches.

(Tip o' the Hat to: [Link1.](https://www.youtube.com/watch?v=YRhxdVk_sIs) [Link2.](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/))

### Architecture

Since building and training a deep learning network from scratch is a herculean task, almost everyone uses a high-ish level architecture and/or a pretrained model customized to their needs. For NLP tasks, Recurrent Neural Networks or Transformers are most commonly used, as these network-designs permit effective analysis of *sequences*, which is very important in processing language (after all, 'the cat sat on the mat' is a very different sentence from 'the mat sat on the cat.'). The following items deal with parameters, specifications, and so on apply specifically to formatting and compatibility with the model. Hugging Face is a library from which cutting-edge transformers can be imported, downloaded, customized, etc. 'Bert' (Bidirectional Encoder Representations from Transformers) is a SOA model developed by Google that you can access there, and 'DeBerta' and 'RoBerta' are a couple of their models, though there are many more variations on the core 'Bert' architecture. Distilbert is a stripped-down version that allows for speedier training.

- General Features:
- Tokenization: As above, this is just the division of the sentences into discrete words. There are, however, different approaches to this that are optimized in a model-specific way, so we often specify a model-specific tokenizer.
- Encoding: One of the key features and strengths of these models is their use of encode-decode layers. In this context, encoding means something more like 'compressing' than merely converting types. The model attempts to condense the input to the minimum requisite information required to capture what's being said in the example, and the decode layer does the opposite. Through this process, the model can learn to capture the 'essence' of the input text in the smallest possible dimensionality. This allows the model to build the optimal representations, reducing redundancy and noise.
- Embedding: Like vectorization, but more similar to the doc2vec approach. The space into which the vectors are placed is arranged such that similar items are 'near' to one another, as discerned by the aformentioned tokenization/encoding process.
- Attention, Attention Heads: Because 1) Language is presented in a sequence: 'home is where the heart is' != 'the heart is where home is.'
  \2) Order and context not only matter, they change the meaning of words: 'after pushing the canoe into the water, he turned and approached the bank' suggests a very different sort of bank than that in 'after finsihing the canoe trip, he remembered he had to stop by the bank.' 3) Ambiguity arises that can only be resolved by order: "I like muffins more than you" could mean, "I like muffins better than I like you" or "I like muffins more than you like muffins", and only what it's preceded or followed by can clarify that.
- Bidirectional: Permitting attention and sequential analyis for elements that both precede *and* follow a given input. This is what attention mechanisms accomplish. From the training corpus, the model learns probabilities of two words appearing in a sequence, probabilities of the word in position + 1 equalling X given that position - 2 = P and position - 1 = Q, and so on. Multi-head attention means simply that the model has a variety of different such features that it weighs when estimating the likelihood of meaning/a subsequent word/a masked word/etc. For example, in addition to positional attention mechanisms, it may have mechanisms that look at parts-of-speech, mechanisms that look for noun-phrases, and so forth.
- Model Specific Features:
- base/large: two different scales of Bert-- base has 12 layers, 768 hidden-nodes, 12 attention-heads, 110 million parameters; large uses 24 layers, 1024 hidden nodes, 16 attention heads, and 340 million parameters. Both of these took literally *days* to train, using multiple TPUs and processing the entire text of Wikipedia!
- Special Tokens: Bert expects certain 'special' tokens to be added to the text it is analyzing. These are used as signals to indicate the beginning and end of a given text item ('CLS' and 'SEP', respectively (with 'CLS' being specific to classification tasks)), token-position in terms of the sentence ('POS'), and training-specific 'hints' that help with discerning where to pay attention ('MASK' and 'NSP').
  - Input_ids and Input_masks: variables specifying whether a given token-position consists of an actual token or if it is actually padding that has no informative meaning.
- cased/uncased: just what it sounds like-- whether to use a model optimized for text that has conventional uppercase and lowercase usage, or to use a model optimized for data that's been standardized to lowercase.
- Autotokenizer: a tokenizer (Python) class that builds a model-specific tokenizer based on the model of Bert (or other transformer) you're using.
- Report_to: integrated platforms on which to save the model
- Save_strategy: whether to save a model after a given iteration, epoch, or whatever.
- Evaluation_strategy: schedule for performing updating An excellent introduction to Bert and related architectures here (Tip o' the Hat to Samia Khalid): [Link.](https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c)

Caveat Lector: Exceptions and qualifications exist for just about everything printed in this cell, and the recommended/demonstrated tools are certainly not the only options. There are tons and tons of resources and models and frameworks and configs out there. Those that I've used here vary in terms of ease, efficiency, and architecture, but they are all widely used and respected, and multitudinous tutorials, documentation entries, and examples can be found online.

```python
#Dependencies List - All Models
#General Purpose
import os
import csv
from tqdm import tqdm
import warnings
import datasets
from datasets import load_dataset, Dataset, DatasetDict
#Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
#Transformers
import transformers
from transformers import DistilBertTokenizer
from transformers import TFBertModel
from transformers import TFDistilBertForSequenceClassification 
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#Torch
import torch 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
#Keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
#Sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
```

Bert + Torch v.1

Tip o' the Hat:

[Link1.](https://medium.com/analytics-vidhya/bert-multi-class-text-classification-e3822836ee0d) [Link2.](https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894)

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in BertPrep['Label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in BertPrep['Text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, 
                                     attention_mask=mask,
                                     return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, 
                                                   batch_size=2, 
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, 
                                                 batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), 
                     lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
            
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                
def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
        
    
np.random.seed(112)
df_train, df_val, df_test = np.split(BertPrep.sample(frac=1, random_state=42), 
                                     [int(.8*len(BertPrep)), int(.9*len(BertPrep))])


EPOCHS = 5
model = BertClassifier()
LR = 1e-6
              
train(model, df_train, df_val, LR, EPOCHS)


evaluate(model, df_test)
```

BERT + Torch v.2

Tip o' the Hat:

[Link.](https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/)

```python
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels =3)
desc = model.device

batch_size = 32          
learning_rate = 1e-5           
epochs = 4            
max_len = 0        

for sent in feature_sentences.Token:
    input_ids = tokenizer.encode(sent, add_special_tokens = True)
    max_len = max(max_len, len(input_ids))

max_len = 250 

input_ids = []
attention_masks = []

for sent in feature_sentences.Token:
    encoded_dict = tokenizer.encode_plus(sent, 
                                        add_special_tokens = True,
                                        max_length = max_len,
                                         truncation = True,
                                         padding = 'max_length',
                                         return_attention_mask = True,
                                         return_tensors = 'pt',
                                        )
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(feature_sentences['Labels'])

train_dataset = TensorDataset(input_ids[train_idx], attention_masks[train_idx], labels[train_idx])
val_dataset = TensorDataset(input_ids[val_idx], attention_masks[val_idx], labels[val_idx])
test_dataset= TensorDataset(input_ids[test_idx], attention_masks[test_idx], labels[test_idx])

train_dataloader = DataLoader(train_dataset, 
                             sampler = RandomSampler(train_dataset),
                             batch_size = batch_size
                             )

validation_dataloader = DataLoader(val_dataset, 
                                  sampler = SequentialSampler(val_dataset),
                                  batch_size = batch_size)

optimizer = AdamW(model.parameters(),
                 lr = learning_rate, 
                 eps = 1e-8
                 )

total_steps = len(train_dataloader)*epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                           num_warmup_steps = 0, 
                                           num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded))

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, epochs):

    total_train_loss = 0
    
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            
            print(' Batch  {:>5, } of {:>5, }.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                  
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        
        result = model(b_input_ids,
                      token_type_ids = None,
                      attention_mask = b_input_mask,
                      labels = b_labels, 
                      return_dict = True)
        
        loss = result.loss
        logits = result.logits
        
        total_train_loss += loss.item()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        
        optimizer.step()
        
        scheduler.step()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    training_time = format_time(time.time() - t0)

    
    t0 = time.time()
    
    model.eval()
    
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    for batch in validation_dataloader: 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            
            result= mode(b_input_ids,
                        token_type_ids = None, 
                        attention_mask = b_input_mask, 
                         labels = b_labels, 
                         return_dict = True)
            
            
        loss = result.loss
        logits = result.logits
        
        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        total_eval_accuarcy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    training_stats.append(
        {
            'epoch' : epoch_i + 1, 
            'Training Loss' : avg_train_loss,
            'Valid. Loss' : avg_val_loss, 
            'Valid. Accur.' : avg_val_accuracy,
            'Training Time' : training_time,
            'Validation Time': validation_time
        }
    )
                      

pd.set_option('precision', 2)
KFP_stats = pd.DataFrame(data = training_stats)
KFP_stats = df_stats.set_index('epoch')
df = df.style.set_table_styles([dict(selector = 'th', props = [('max-width', '70px')])])
```

Tensorflow + Distilbert

Tips o' the Hat:

[Link1.](https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894) [Link2.](https://www.kaggle.com/code/bhavikardeshna/transformer-bert-tensorflow/notebook) [Link3.](https://towardsdatascience.com/hugging-face-transformers-fine-tuning-distilbert-for-binary-classification-tasks-490f1d192379)

```python
tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')

X_input_ids = np.zeros((len(BText2), 256))
X_attn_masks = np.zeros((len(BText2), 256))

def encode_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['Text'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256, 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

X_input_ids, X_attn_masks = encode_data(BText2, X_input_ids, X_attn_masks, tokenizer)

labels = np.zeros((len(BText2), 3))
labels[np.arange(len(BText2)), BText2['target'].values]= 1

def DatasetMapFunction(input_ids, attn_masks, labels):
    return {
        \
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))
dataset = dataset.map(DatasetMapFunction)    
dataset = dataset.shuffle(10000).batch(16, drop_remainder=True)

p = 0.8
train_size = int((len(BText2)//16)*p)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

model = TFBertModel.from_pretrained('distilbert-base-cased') 

input_ids = tf.keras.layers.Input(shape=(256, ), name = 'input_ids', dtype = 'int32')
attn_masks = tf.keras.layers.Input(shape = (256, ), name = 'attention_mask', dtype = 'int32')
bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1] 
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(3, activation='softmax', name='output_layer')(intermediate_layer)
feedback_model = tf.keras.Model(inputs = [input_ids, attn_masks], outputs = output_layer)
feedback_model.summary()

feedback_model.compile(optimizer=Adam(learning_rate=1e-5, decay=1e-6), 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

history = feedback_model.fit(train_dataset, 
                            steps_per_epoch = 200, 
                            validation_data = val_dataset,
                            epochs = 5)
```

Tensorflow + Autotune + Bert

Tip o' the Hat:

[Link1.](https://www.kaggle.com/code/imvision12/tensorflow-feedback-bert-baseline) [Link2.](https://www.tensorflow.org/tutorials/keras/text_classification)

```python
AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 500

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for text in texts:
        token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length',
                         add_special_tokens=True)
        input_ids.append(token['input_ids'])
        token_type_ids.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
    
    return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)

tokenizer = transformers.BertTokenizer.from_pretrained('../input/huggingface-bert-variants/bert-base-cased/bert-base-cased')
tokenizer.save_pretrained('.')

sep = tokenizer.sep_token
sep

BertPrep['inputs'] = BertPrep['Text']

new_label = {"Label": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
BertPrep = BertPrep.replace(new_label)

X_train, X_valid, y_train, y_valid = train_test_split(BertPrep['inputs'], BertPrep['Label'], test_size=0.12, random_state=42)

X_train = bert_encode(X_train.astype(str), tokenizer)
X_valid = bert_encode(X_valid.astype(str), tokenizer)

y_train = y_train.values
y_valid = y_valid.values

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

def build_model(bert_model, max_len=MAX_LEN):    
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    clf_output = sequence_output[:, 0, :]
    clf_output = Dropout(.1)(clf_output)
    out = Dense(3, activation='softmax')(clf_output)
    
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

transformer_layer = (TFBertModel.from_pretrained('../input/huggingface-bert-variants/bert-base-cased/bert-base-cased'))
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

train_history = model.fit(
    train_dataset,
    steps_per_epoch=200,
    validation_data=valid_dataset,
    epochs=5
)
```

TensorFlow + DistilBert2

Tips o' the Hat:

[Link1.](https://medium.com/geekculture/hugging-face-distilbert-tensorflow-for-custom-text-classification-1ad4a49e26a7) [Link2.](https://www.sunnyville.ai/fine-tuning-distilbert-multi-class-text-classification-using-transformers-and-tensorflow/) [Link3.](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)

```python
feature_sentences['Label_cats'] = feature_sentences['Labels'].astype('category').cat.codes

pd.set_option('display.max_colwidth', None)
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
BATCH_SIZE = 16
N_EPOCHS = 3

data_texts = feature_sentences['Token'].to_list()

data_labels = feature_sentences['Label_cats'].to_list()

train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0)

# Keep some data for inference (testing)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(feature_sentences['Token'], feature_sentences['Labels'], test_size = 0.3)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(list(train_texts),
                           truncation = True,
                           padding = True)

val_encodings = tokenizer(list(val_texts),
                          truncation = True,
                          padding = True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))

test_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 3)

O_ptimizer = tf.keras.optimizers.Adam(learning_rate = 5e-5)

l0ss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)

model.compile(optimizer = O_ptimizer, loss = l0ss, metrics = ['accuracy'])

model.fit(train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE), epochs = N_EPOCHS, batch_size = BATCH_SIZE)
```

Keras + LSTM 1

Tip o' the Hat:

[Link.](https://www.tensorflow.org/tutorials/keras/text_classification)

```python
vocab_size = 10000
embedding_dim = 64
max_length = 250
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

items = PreppedText
labels = PreppedLabels

train_size = int(len(PreppedText)* training_portion)
train_items = items[0: train_size]
train_labels = labels[0: train_size]

validation_items = items[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_items)
word_index = tokenizer.word_index

dict(list(word_index.items())[10:20])

train_sequences = tokenizer.texts_to_sequences(train_items)
train_padded = pad_sequences(train_sequences, maxlen= max_length, padding = padding_type, truncating = trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation_items)
validation_padded = pad_sequences(validation_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type )

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation = 'relu'),
    tf.keras.layers.Dense(6, activation = 'softmax')
])
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

num_epochs = 10
history = model.fit(train_padded, 
                    training_label_seq, 
                    epochs = num_epochs,
                    validation_data = (validation_padded,
                                      validation_label_seq),
                    verbose = 2)
```

LSTM v. 2 Tips o' the Hat:

- [Link1](https://adventuresinmachinelearning.com/keras-lstm-tutorial/).
- [Link2](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17).

```python
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 400
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(KFP_df['Token'].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(KFP_df['Token'].values)
X = pad_sequences(X, maxlen = MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(KFP_df['Labels']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)


model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length = X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.1, callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3, min_delta = 0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n Loss: {:0.3f}\n Accuracy: {:0.3f}'.format(accr[0], accr[1]))
```

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 400, 100)          5000000   
                                                                 
 spatial_dropout1d (SpatialD  (None, 400, 100)         0         
 ropout1D)                                                       
                                                                 
 lstm (LSTM)                 (None, 100)               80400     
                                                                 
 dense (Dense)               (None, 3)                 303       
                                                                 
=================================================================
Total params: 5,080,703
Trainable params: 5,080,703
Non-trainable params: 0
_________________________________________________________________
None

Epoch 1/5
414/414 [==============================] - 1188s 3s/step - loss: 0.8526 - accuracy: 0.6225 - val_loss: 0.7720 - val_accuracy: 0.6560
Epoch 2/5
414/414 [==============================] - 1144s 3s/step - loss: 0.7044 - accuracy: 0.6924 - val_loss: 0.7799 - val_accuracy: 0.6482
Epoch 3/5
414/414 [==============================] - 1296s 3s/step - loss: 0.6040 - accuracy: 0.7445 - val_loss: 0.8278 - val_accuracy: 0.6363
Epoch 4/5
414/414 [==============================] - 1422s 3s/step - loss: 0.5146 - accuracy: 0.7878 - val_loss: 0.9018 - val_accuracy: 0.6237

230/230 [==============================] - 79s 344ms/step - loss: 0.8855 - accuracy: 0.6290
Test set
 Loss: 0.886
 Accuracy: 0.629
```

Five-Fold DeBERTa + AutoModel

Tips o' the Hat:

[Link1.](https://www.kaggle.com/code/tdh512194/simple-5-foldcv-deberta-baseline/notebook)

[Link2.](https://towardsdatascience.com/three-out-of-the-box-transformer-models-4bc4880bc992)

```python
def set_seed(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
seed_everything(512)


model_name = '../input/debertav3base'

tokenz = AutoTokenizer.from_pretrained(model_name)
sep = tokenz.sep_token

df = train
df['inputs'] = df.Text
new_label = {
    "Label": {
        "Ineffective": 0, "Adequate": 1, "Effective": 2
    }
}
df = df.replace(new_label)
df = df.rename(columns = {'Label': 'label'})
ds = Dataset.from_pandas(df)


def tokenize_function(x):
    return tokenz(x['inputs'], truncation=True, 
                max_length=256, 
                padding='max_length')

def score(preds):
    return {'log loss': log_loss(
        preds.label_ids, 
        F.softmax(torch.Tensor(preds.predictions))
    )}

tokens_ds = ds.map(tokenize_function, 
                batched=True, 
                remove_columns=('discourse_text',
                                'discourse_type', 
                                'inputs',
                                'discourse_id',
                                'essay_id'))


kfold = StratifiedGroupKFold(n_splits=5)

dds_folds = []
for train_idxs, val_idxs in kfold.split(df.index, df.label, df.essay_id):
    dds_folds.append(
        DatasetDict({
        'train':tokens_ds.select(trn_idxs), 
        'test': tokens_ds.select(val_idxs)
    }))

learning_rate = 3e-5
batch_size =  16
weight_decay = 0.01
epochs =  2

for fold in range(5):
    print(f'Training fold {fold}')
    args = TrainingArguments(
        output_dir = f'outputs/fold_{fold}',
        learning_rate = learning_rate, 
        warmup_ratio = 0.1, 
        lr_scheduler_type ='cosine', 
        num_train_epochs = epochs, 
        weight_decay = weight_decay, 
        report_to = 'none',
        evaluation_strategy = "epoch", 
        save_strategy = "no",
        label_smoothing_factor = 0.05,
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=df.label.nunique())
    trainer = Trainer(model = model, 
                      args = args, 
                   train_dataset = dds_folds[fold]['train'], 
                      eval_dataset = dds_folds[fold]['test'], 
                  compute_metrics=score)

    trainer.train()
    model.save_pretrained(f'outputs/fold_{fold}')
```

This model was the highest-performing of all the models (and quite a few more) that I've included in this overview. 