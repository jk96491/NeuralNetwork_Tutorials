from Advanced.Transformer.Network_NN import *
from Advanced.Transformer import Utils as utils

import matplotlib.pyplot as plt

enc_seq_len = 6
dec_seq_len = 2
output_sequence_length = 1

dim_val = 10
dim_attn = 5
lr = 0.002
epochs = 20

n_heads = 1

n_decoder_layers = 3
n_encoder_layers = 3

batch_size = 15

#init network and optimizer
transformer = Transformer(dim_val, dim_attn, 1, dec_seq_len,  output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

#keep track of loss for graph
losses = []

# build live matplotlib fig
fig = plt.figure()

ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

for e in range(epochs):
    out = []

    for b in range(-10 - enc_seq_len, 10 - enc_seq_len):
        optimizer.zero_grad()
        X, Y = utils.get_data(batch_size, enc_seq_len, output_sequence_length)

        # Forward pass and calculate loss
        net_out = transformer(X)
        # print(net_out.shape,Y.shape)
        loss = torch.mean((net_out - Y) ** 2)

        # backwards pass
        loss.backward()
        optimizer.step()

        # Track losses and draw rgaph
        out.append([net_out.detach().numpy(), Y])
        losses.append(loss)

        ax.clear()
        ax.plot(losses)
        ax.set_title("Mean Squared Error")
        fig.canvas.draw()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

o = []
x = [torch.sigmoid(torch.arange(-10, -1).float()).unsqueeze(-1).numpy().tolist()]

# Draw graph comparing to sigmoid
for i in range(-10, 10, output_sequence_length):
    o.append([torch.sigmoid(torch.tensor(i).float())])
    q = torch.tensor(x).float()

    if (output_sequence_length == 1):
        x[0].append([transformer(q).detach().squeeze().numpy()])
    else:
        for a in transformer(q).detach().squeeze().numpy():
            x[0].append([a])

ax.clear()
ax.plot(x[0], label='Network output')
ax.plot(o, label='Sigmoid function')
ax.set_title("")
ax.legend(loc='upper left', frameon=False)