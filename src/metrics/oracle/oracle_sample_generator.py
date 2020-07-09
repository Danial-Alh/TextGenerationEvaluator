from metrics.oracle.oracle_lstm import Oracle_LSTM

oracle = Oracle_LSTM(num_emb=5000, batch_size=128, emb_dim=3200, hidden_dim=32, sequence_length=20)
samples = oracle.generate(15000)
add_space = lambda x: " ".join(map(str, x))
samples_astext = "\n".join(map(add_space, samples))

with open('samples.txt', 'w') as fd:
    fd.write(samples_astext)
