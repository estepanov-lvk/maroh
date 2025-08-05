import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LeakyReLU
from tensorflow.keras import Model

class GATLayer(Layer):
    def __init__(self, out_features, num_heads, adj, kernel_initializer, activation='relu', dropout_rate=0.15):
        super(GATLayer, self).__init__()
        self.out_features = out_features  # Output feature dimension
        self.num_heads = num_heads  # Number of attention heads
        self.activation = tf.keras.activations.get(activation)
        self.adj = adj

        # Linear transformation for node features
        self.W = Dense(out_features * num_heads, use_bias=False,
                       kernel_initializer=kernel_initializer)

        # Attention mechanism
        self.attn_weights = Dense(1, use_bias=False,
                                  kernel_initializer=kernel_initializer)
        self.leaky_relu = LeakyReLU(alpha=0.2)
        # self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, h, adj=None):
        """
        Forward pass for the GAT layer.
        :param h: Node features, shape [N, in_features]
        :param adj: Adjacency matrix, shape [N, N]
        :return: Updated node features, shape [N, out_features * num_heads]
        """
        if adj == None:
            adj = self.adj
        N = tf.shape(h)[0]  # Number of nodes

        # Linear transformation
        Wh = self.W(h)  # [N, out_features * num_heads]
        Wh = tf.reshape(Wh, [N, self.num_heads, self.out_features])  # [N, num_heads, out_features]

        # Compute attention scores
        
        # Wh_repeated = tf.expand_dims(Wh, 1)  # [N, 1, num_heads, out_features]
        # Wh_broadcasted = tf.expand_dims(Wh, 0)  # [1, N, num_heads, out_features]
        
        # Wh_repeated: Expand and tile Wh along the second dimension
        Wh_repeated = tf.expand_dims(Wh, 1)  # [N, 1, num_heads, out_features]
        Wh_repeated = tf.tile(Wh_repeated, [1, tf.shape(adj)[1], 1, 1])  # [N, N, num_heads, out_features]
        # Wh_broadcasted: Expand and tile Wh along the first dimension
        Wh_broadcasted = tf.expand_dims(Wh, 0)  # [1, N, num_heads, out_features]
        Wh_broadcasted = tf.tile(Wh_broadcasted, [tf.shape(adj)[0], 1, 1, 1])  # [N, N, num_heads, out_features]
        
        concat = tf.concat([Wh_repeated, Wh_broadcasted], axis=-1)  # [N, N, num_heads, 2 * out_features]

        # Compute attention coefficients
        e = self.attn_weights(concat)  # [N, N, num_heads]
        # e = self.attn_weights(tf.reshape(concat, [N * N, self.num_heads, 2 * self.out_features]))  # [N * N, num_heads]
        # print("AAA", tf.shape(e), )
        # for _ in range(1000000000000000000):
        #     1
        e = tf.reshape(e, [N, N, self.num_heads])  # [N, N, num_heads]
        # print("BBB", tf.shape(e))
        e = self.leaky_relu(e)

        # Masked attention (apply adjacency matrix)
        mask = tf.expand_dims(adj, -1)  # [N, N, 1]
        e = tf.where(mask == 0, tf.constant(-1e9, dtype=e.dtype), e)  # [N, N, num_heads]

        # Normalize attention coefficients
        alpha = tf.nn.softmax(e, axis=1)  # [N, N, num_heads]

        # alpha = self.dropout(alpha)

        # Aggregate messages
        h_prime = tf.einsum('ijh,jhf->ihf', alpha, Wh)  # [N, num_heads, out_features]
        h_prime = tf.reshape(h_prime, [N, self.num_heads * self.out_features])  # [N, num_heads * out_features]

        return self.activation(h_prime)


class GAT(Model): # Not used right now
    def __init__(self, hidden_features, out_features, num_heads, message_iterations, kernel_initializer):
        super(GAT, self).__init__()
        self.gats = []
        # k-1 iterations
        for _ in (range(max(1, message_iterations - 1))):
            self.gats.append(GATLayer(hidden_features, num_heads, kernel_initializer=kernel_initializer))
        self.gat_final = GATLayer(out_features, 1, kernel_initializer=kernel_initializer)  # Single head for final layer

    def call(self, h, adj):
        """
        Forward pass for the GAT model.
        :param h: Node features, shape [N, in_features]
        :param adj: Adjacency matrix, shape [N, N]
        :return: Output node features, shape [N, out_features]
        """
        for gat_i in self.gats:
            h = gat_i(h, adj)
            h = tf.nn.relu(h)
        h = self.gat_final(h, adj)
        return h
    
    def build(self, input_shape=None):
        self.gats[0].build(input_shape=input_shape)
        for i in range(len(self.gats)):
            self.gats[i].build(input_shape=input_shape)
        self.gat_final.build(input_shape=input_shape)
        self.built = True



# Example usage
if __name__ == "__main__":
    # Define model parameters
    in_features = 2  # Input feature dimension
    hidden_features = 8  # Hidden layer feature dimension
    out_features = 2  # Output feature dimension
    num_heads = 2  # Number of attention heads

    # Initialize model
    model = GAT(in_features, hidden_features, out_features, num_heads)

    # Example input
    N = 8  # Number of nodes
    h = tf.random.normal([N, in_features])  # Random node features
    # Adj. matrix must represent graph structure
    adj = tf.random.uniform([N, N], minval=0, maxval=2, dtype=tf.int32)  # Random adjacency matrix

    # Forward pass
    output = model(h, adj)
    print("Output shape:", output.shape)  # Should be [N, out_features]