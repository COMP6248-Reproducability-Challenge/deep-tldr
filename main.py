"""
This is a re-implementation of the paper:'A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION'
    - https://openreview.net/pdf?id=HkAClQgA-

"""
import linecache

import gc
import matplotlib; matplotlib.use('module://backend_interagg')
import os
import tracemalloc
import matplotlib.pyplot as plt
from dataset import Dataset
import math
import rouge
import numpy as np
import torch
import torch.nn as nn
from config import *
# from torchviz import make_dot
import log
import nltk

nltk.download('punkt')


def display_top(snapshot, key_type='lineno', limit=10):
    """Takes snapshot of memory usage, displaying the top 10 (i.e. limit) components with the highest memory
     allocation. To work, this method need to be placed straight after the import statements. """
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


device = "cuda:0" if CUDA and torch.cuda.is_available() else "cpu"
torch.manual_seed(125)
logger = log.logger


class LSTMCell(nn.Module):
    """
        Module representing a 'vinilla' LSTM cell.
        Follows structure as shown in 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/'
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_parameters()

        # Initialise the cells learnable layers
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.forget_linear = nn.Linear(input_size, hidden_size)
        self.cell_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(input_size, hidden_size)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, article_batch_t, hidden):
        """
            Calculates the hidden state hy (also known as the short term memory) and context vector c(y) also known
            as the long term memory for 1 pass of a LSTM cell.
            Notation for variables correlates to notation from blog:
                - 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/'
        """
        hx, cx = hidden  # output and long term memory of previous timestep (t-1)
        cx = cx.to(device)

        article_batch_t = article_batch_t.to(device)
        hx = hx.to(device)
        gates = torch.cat((article_batch_t, hx), dim=1)  # concatanate the input with the previous output

        gates.to(device)

        in_gate, forget_gate, cell_gate, out_gate = gates, gates, gates, gates

        in_gate = torch.sigmoid(self.in_linear(in_gate))
        forget_gate = torch.sigmoid(self.forget_linear(forget_gate))
        cell_gate = torch.tanh(self.cell_linear(cell_gate))
        out_gate = torch.sigmoid(self.out_linear(out_gate))

        # long term memory for timestep t
        cy = torch.mul(cx, forget_gate) + torch.mul(in_gate, cell_gate)
        # short term memory of timestep t
        hy = torch.mul(out_gate, torch.tanh(cy))

        return hy, cy


class Encoder(nn.Module):
    """
        Represents a bi-directional LSTM encoder
        See section 2 in paper
    """

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm_f = LSTMCell(input_dim, hidden_dim)
        self.lstm_b = LSTMCell(input_dim, hidden_dim)

    def forward(self, article_batch):
        # Initialize hidden state with zeros
        batch_size = article_batch.size(0)
        hn_f = torch.zeros(batch_size, self.hidden_dim)
        hn_b = torch.zeros(batch_size, self.hidden_dim)

        # Initialize cell state
        cn_f = torch.zeros(batch_size, self.hidden_dim)
        cn_b = torch.zeros(batch_size, self.hidden_dim)

        h_f = []
        c_f = []
        h_b = []
        c_b = []

        '''
        for each article in the batch, calculate the hidden states and context vectors, and append to the respective
        lists 
        '''
        for index in range(article_batch.size(1)):
            # forward processes the article left to right
            x_f = article_batch[:, index]
            # backward processes the article right to left
            x_b = article_batch[:, -index - 1]  # - 1 because -0 is still index 0
            # do a forward pass on the forwards and backwards lstm cells
            hn_f, cn_f = self.lstm_f(x_f, (hn_f, cn_f))
            hn_b, cn_b = self.lstm_b(x_b, (hn_b, cn_b))
            h_f.append(hn_f)
            c_f.append(cn_f)
            h_b.append(hn_b)
            c_b.append(cn_b)

        # combine all the hidden states and context vectors for each article and return as a batch
        return torch.cat((torch.stack(h_f, dim=1), torch.stack(h_b, dim=1)), dim=2), \
               torch.cat((torch.stack(c_f, dim=1), torch.stack(c_b, dim=1)), dim=2)


class EncoderAttention(nn.Module):
    """
        Inter temporal attention - see section 2.1 in paper for details
    """

    def __init__(self, h_d_input_size, h_e_input_size):
        super(EncoderAttention, self).__init__()
        # needs to output scalar
        self.bilinear = nn.Bilinear(h_d_input_size, h_e_input_size, 1, bias=False)
        self.sum_e = None

    def forward(self, h_d, h_e):
        """
        Perform attention over encoder hidden states
        Calculates the attention weights for all tokens in a time step and the updated context vector
        """
        # make h_d into a matrix s.t it can be concatenated with each encoder hidden state
        e_t = self.bilinear(h_d.unsqueeze(1).repeat(1, h_e.size(1), 1), h_e)  # eq (2)

        # eq (3)
        if self.sum_e is None:
            e_t_prime = torch.exp(e_t)
            self.sum_e = torch.exp(e_t)
        else:
            e_t_prime = torch.exp(e_t) / (self.sum_e + EPSILON)
            self.sum_e = self.sum_e + torch.exp(e_t)

        # dividing by scalar sum of words for each sentence
        alpha_t = e_t_prime.squeeze() / (torch.sum(e_t_prime, dim=1) + EPSILON)  # eq (4)
        # apply attention weights to each corresponding hidden encoder state 'vector' and sum column-wise
        c_t = torch.matmul(alpha_t, h_e)  # eq (5)

        return alpha_t.view(alpha_t.size(0), -1), c_t.sum(dim=1)


class DecoderAttention(nn.Module):
    """
        Inter decoder attention - see section 2.2 in paper for details
    """

    def __init__(self, input_size):
        super(DecoderAttention, self).__init__()
        # needs to output scalar
        self.bilinear = nn.Bilinear(input_size, input_size, 1, bias=False)
        self.h_d_all = None

    def forward(self, h_d_t):
        """
            Perform attention over decoder hidden states
            Returns the updated context vector
        """
        if self.h_d_all is None:
            self.h_d_all = h_d_t.unsqueeze(1)
            return torch.zeros(h_d_t.size()).to(device)

        e_t = self.bilinear(h_d_t.unsqueeze(1).repeat(1, self.h_d_all.size(1), 1), self.h_d_all)  # eq (6)

        alpha_t = torch.exp(e_t).squeeze(2) / (torch.sum(e_t, dim=1) + EPSILON)  # eq (7)
        # apply attention weights to each corresponding hidden encoder state 'vector' and sum column-wise
        c_t = torch.matmul(alpha_t.unsqueeze(2).permute(0, 2, 1), self.h_d_all)  # eq (8)

        # update with new timestep hidden state
        self.h_d_all = torch.cat((self.h_d_all.to(device), h_d_t.unsqueeze(1).to(device)), dim=1)

        return c_t.squeeze(1)


class Decoder(nn.Module):
    """
        Represents a bi-directional LSTM decoder
        See section 2 in paper
    """

    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()

        self.lstm = LSTMCell(input_size, hidden_size)

        # Modify the inputs to the linear layer accordingly depending on if intra-decoder attention is being used
        if INTRA_DECODER:
            self.linear_u = nn.Linear(3 * hidden_size, 1, bias=True)
        else:
            self.linear_u = nn.Linear(2 * hidden_size, 1, bias=True)

        # 5 is the vocab size when dealing with the dummy dataset
        self.vocab_size = 5 if USE_D1 else len(dataset.processor.text.vocab.itos)

        # Modify the inputs to the linear layer accordingly depending on if intra-decoder attention is being used
        if INTRA_DECODER:
            self.linear_out = nn.Linear(3 * hidden_size, self.vocab_size, bias=True)
        else:
            self.linear_out = nn.Linear(2 * hidden_size, self.vocab_size, bias=True)

        self.encoder_attention = EncoderAttention(hidden_size, hidden_size)  # length of input data vectors
        self.decoder_attention = DecoderAttention(hidden_size)  # length of input data vectors

        self.encoder_states = None
        self.hidden_state = None
        self.context_vector = None
        self.article = None
        self.article_indices = None

    def init_hidden(self, h_e, article, article_indices):
        self.encoder_states = h_e

        # Initialize hidden state with zeros
        self.hidden_state = h_e[:, -1]

        # Initialize cell state
        self.context_vector = torch.zeros(self.hidden_state.size())

        # current article
        self.article = article

        # current article with each word replaced with their vocab index
        self.article_indices = article_indices

    def forward(self, last_predicted_word):
        """
            Decoder forward pass, calculating the next time step t
        """
        self.hidden_state, self.context_vector = self.lstm(last_predicted_word,
                                                           (self.hidden_state, self.context_vector))

        input_word_probs, temporal_attention = self.encoder_attention(self.hidden_state, self.encoder_states)

        # Modify the inputs to the layers accordingly depending on if intra-decoder attention is being used
        if INTRA_DECODER:
            intra_decoder_attention = self.decoder_attention(self.hidden_state)
            cat_input = torch.cat((self.hidden_state, temporal_attention, intra_decoder_attention), dim=1)
        else:
            cat_input = torch.cat((self.hidden_state, temporal_attention), dim=1)

        use_pointer_prob = torch.sigmoid(self.linear_u(cat_input))  # eq (11)

        # Token generation - eq (9)
        generated_tokens = torch.softmax(self.linear_out(cat_input), dim=1)

        '''
        need a probability distribution over the entire vocab corpus that contains probability of each word being
        chosen for this article
        '''
        vocab_probs = torch.zeros((last_predicted_word.size(0), self.vocab_size)).to(
            device)
        # incorporates pointer mechanism probabilities (eq (12))
        vocab_probs = vocab_probs.scatter_add(1, self.article_indices, input_word_probs * use_pointer_prob)
        # incorporates token generation probabilities (eq (12))
        vocab_probs = vocab_probs + generated_tokens * (
                1 - use_pointer_prob)

        # normalise just to make the values probabilities
        vocab_probs = vocab_probs / (torch.sum(vocab_probs, dim=1).unsqueeze(0).t() + EPSILON)

        # need to return a word for each batch
        word_index = torch.multinomial(vocab_probs, 1)

        word = vocabulary[word_index] if USE_D1 else dataset.processor.text.vocab.vectors.data[word_index]

        return word, vocab_probs, word_index


class Seq2Seq(nn.Module):
    """
        Combines the encoder and decoder models to create a connected seq-2-seq model
    """

    def __init__(self, embedding_size, hidden_size, use_teacher_forcing):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = Encoder(embedding_size + hidden_size, hidden_size)

        # *2 because bidirectional encoder cats two hidden states together each time
        self.decoder = Decoder(embedding_size + 2 * hidden_size, 2 * hidden_size)
        self.use_teacher_forcing = use_teacher_forcing

    def forward(self, article_batch, article_indices_batch, target_summary_batch, target_summary_indices_batch):
        self.reset_new_batch()

        # stores all hidden states from the encoder
        hidden_states_e, context_vectors_e = self.encoder(article_batch)

        self.decoder.init_hidden(hidden_states_e, article_batch, article_indices_batch)
        if USE_D1:
            last_word = torch.Tensor([[-1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]]).to(
                device)  # "start of article"
        else:
            last_word = target_summary_batch[:, 0].to(device)

        max_length = target_summary_indices_batch.size(1)

        tldr = []
        tldr_vocab_probs = []
        tldr_indices = []

        for word_index in range(max_length):
            last_word, vocab_probs, ind = self.decoder(last_word)
            last_word = last_word.squeeze(1)

            tldr.append(last_word)
            tldr_vocab_probs.append(vocab_probs)
            tldr_indices.append(ind)

            if self.training:
                # only use teacher forcing 25% of the time - see paper appendix B
                if np.random.rand() < 0.25:
                    # pretend the model got it right for teacher forcing
                    last_word = target_summary_batch[:, word_index]

        return tldr, tldr_vocab_probs, tldr_indices

    def reset_new_batch(self):
        # must have this line to prevent gradient graph from blowing up exponentially
        self.decoder.encoder_attention.sum_e = None
        self.decoder.decoder_attention.h_d_all = None


def save_training_checkpoint(epoch, model, optimizer, loss, path):
    """
        Saves training state for the given epoch to a file
    """
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

        print("Training checkpoint saved for epoch {}".format(epoch))
    except RuntimeError as e:
        logger.error("Checkpoint NOT saved for epoch {}".format(epoch))
        logger.exception(e)
        logger.error("Continuing without saving...")


def load_training_checkpoint(model, optimizer, path):
    """
        Load a checkpoint file containing a previous training state from a certain epoch
    """
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1  # start from the next uncheckpointed epoch
        loss = checkpoint['loss']
        print("Checkpoint parameters loaded successfully")

    # if there is no checkpoint saved then assume a fresh training run is being done
    except FileNotFoundError:
        print("Checkpoint file does not exist. Starting fresh run.")
        epoch = 0
        loss = None

    return epoch, loss


def print_summary(processor, train_indices, gt_indices, output_indices):
    """
        Prints the summaries  to the console
        Summaries include: original, our summary, ground truth and rouge scores for R-1, R-2, and R-N
    """
    logger.info("Calculating summary")
    output_indices = torch.stack(output_indices).to('cpu')  # send back to cpu for memory savings
    summary_for_output = output_indices.permute(1, 0, 2)

    for i in range(summary_for_output.shape[0]):
        line = summary_for_output[i].to('cpu')
        original = train_indices[i].to('cpu')
        gt = gt_indices[i].to('cpu')
        original_words = [processor.text.vocab.itos[w.item()] for w in original]
        gt_words = [processor.text.vocab.itos[w.item()] for w in gt]
        words = [processor.text.vocab.itos[w.item()] for w in line]
        out = " ".join(words).replace("<pad>", "").replace("  ", " ")
        orig_out = " ".join(original_words).replace("<pad>", "").replace("  ", " ")
        gt_out = " ".join(gt_words).replace("<pad>", "").replace("  ", " ")

        out_str = "\nOriginal: \n" + orig_out + " \n\nOur summary: \n" + out + "\n\nGT Summary: \n" + gt_out
        logger.info(out_str)
        rouge_scores = calc_rouge(orig_out, gt_out)
        rouge_str = "\n".join([str(i) for i in rouge_scores.items()])
        logger.info("ROUGE scores: " + rouge_str)
    logger.info("Summary complete")


def calculate_loss(gt_summ_indices, vocab_prob):
    """
        Manually calculate the loss using the negative log loss
    """
    l = torch.tensor(0.0).to(device)
    for index, word_probs in zip(gt_summ_indices.t(), vocab_prob):
        # hacky way to ignore padding
        # word_probs[:, 1] = 1

        probs = torch.gather(word_probs, 1, index.unsqueeze(1))
        # epsilon  to avoid log of 0
        l = l - torch.sum(torch.log(probs + EPSILON))
    return l


def plot_epoch_loss(plt_data, epoch):
    """
        Plot the epoch loss graph
    """
    if epoch % PRINT_SUMMARY_FREQUENCY is 0 or epoch is 5:
        plt.clf()
        plt.ylabel('Total loss')
        plt.xlabel('epoch')
        data_x, data_y = zip(*plt_data)
        plt.scatter(data_x, data_y)
        plt.plot(data_x, data_y)
        plt.draw()
        plt.pause(0.002)
        logger.info("Updated plot")


def memory_snapshot():
    """
        Take a snapshot of the memory
    """
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    print("\n\n\n")


def calc_rouge(our_summary: str, gt_summary: str):
    """
        Calculates the rouge scores for R-1, R-2 and R-L
        Will contain a dict with the f1, recall and precision scores for each rouge metric
    """
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2
                            )
    scores = evaluator.get_scores([our_summary], [gt_summary])
    return scores


if __name__ == '__main__':
    '''
    STEP 1: LOAD DATASET
    '''
    dataset = Dataset()
    embedding_size = dataset.get_embedding_size()
    hidden_size = dataset.get_hidden_size()

    if USE_D1:
        training_data_str = dataset.get_training_data_str()
        vocabulary = dataset.get_vocabulary()
        training_data = dataset.get_training_data()
        gt_summaries = dataset.get_gt_summaries()
        training_data_indices, gt_summaries_indices = dataset.get_indices()
    print()

    '''
    STEP 2: INSTANTIATE MODEL CLASS
    '''
    model = Seq2Seq(embedding_size, hidden_size, use_teacher_forcing=True).to(device)

    '''
    STEP 3: INSTANTIATE OPTIMIZER CLASS
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.2) # weight decay -> adds regularisation

    '''
    STEP 4: TRAIN THE MODEL
    '''
    start_epoch = 0
    loss = 0
    if not USE_D1 and not USE_D2:
        start_epoch, loss = load_training_checkpoint(model, optimizer, CHEKPOINT_PATH)

    plt.ion()  # Turns on interactive mode so graph will update
    plt.axis('auto')  # auto scale the axes
    plt_data = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        losses = []
        logger.info("\n---------------EPOCH {}--------------------".format(epoch))
        if not USE_D1:
            training_data_indices, gt_summaries_indices = dataset.get_indices()
            training_data_indices = training_data_indices.transpose(0, 1)
            gt_summaries_indices = gt_summaries_indices.transpose(0, 1)

        optimizer.zero_grad()
        for i in range(0, BATCH_SIZE, ARTICLES_ON_DEVICE):
            if i % 1 == 0:
                logger.info("Processing article {}".format(i))
            end_i = max(i + ARTICLES_ON_DEVICE, BATCH_SIZE)
            training_data_indices_device = training_data_indices[i:end_i].detach().to(device)
            gt_summaries_indices_device = gt_summaries_indices[i:end_i].detach().to(device)
            training_data_device = dataset.get_training_data(training_data_indices=training_data_indices_device).to(
                device)
            gt_summaries_device = dataset.get_gt_summaries(gt_summaries_indices=gt_summaries_indices_device).to(device)
            summary, vocab_probs, summary_indices = model(training_data_device, training_data_indices_device,
                                                          gt_summaries_device, gt_summaries_indices_device)

            # Print summaries
            if epoch % PRINT_SUMMARY_FREQUENCY == 0 and epoch is not 0 and not USE_D1:
                print_summary(dataset.processor, training_data_indices_device, gt_summaries_indices_device,
                              summary_indices)

            # Calculate loss
            loss = calculate_loss(gt_summaries_indices_device, vocab_probs)
            losses.append(loss.detach().item())
            loss.backward(retain_graph=True)
            gc.collect()
            torch.cuda.empty_cache()
        avg_loss = sum(losses) / len(losses)
        logger.info("Average loss: {}".format(avg_loss))
        plt_data.append((epoch, avg_loss))
        plot_epoch_loss(plt_data, epoch)

        optimizer.step()

        # only save checkpoints for models based on the proper dataset
        if not USE_D1 and not USE_D2:
            save_training_checkpoint(epoch, model, optimizer, loss, CHEKPOINT_PATH)

        del loss  # free up memory

    plt.ioff()
    plt.show()
