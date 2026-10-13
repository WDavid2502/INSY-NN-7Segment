#ifndef XOR_H
#define XOR_H

/* Automatisch erzeugt von nn.py - nicht von Hand ändern. */

#define XOR_NUM_LAYERS 2
#define XOR_NUM_INPUTS 2
#define XOR_NUM_OUTPUTS 1

static const int XOR_LAYER_SIZES[3] = {2, 2, 1};

/* Layer 1: 2 Neuronen, je 2 Inputs */
static const float XOR_L1_WEIGHTS[2][2] = {
    {6.24268546f, -6.19726464f},
    {-5.33566512f, 5.04714329f},
};
static const float XOR_L1_BIASES[2] = {
    -3.56318018f, -2.80511131f
};

/* Layer 2: 1 Neuronen, je 2 Inputs */
static const float XOR_L2_WEIGHTS[1][2] = {
    {8.09413199f, 8.32569933f},
};
static const float XOR_L2_BIASES[1] = {
    -4.05102283f
};

#endif /* XOR_H */
