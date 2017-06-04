DATASET_PATH='../../charades/data'

# Constants
NUM_ACTIONS=157 + 1
EPOCHS=50
TORCH_DEVICE=0
INTERMEDIATE_TEST=0
TEST_FREQ=1
LOG=False
USE_GPU=True

# Experiment options
FEATURE_SIZE=2048
HIDDEN_SIZE=512
LAMBDA=0
PREDICTION_LOSS='MSE'
USE_LSTM=True#False

# Optimizer options
OPTIMIZER='ADAM'
LR=0.001
MOMENTUM=0.9
CLIP_GRAD=False#True



def print_config():
    print("# Constants")
    print("NUM_ACTIONS=%d"%(NUM_ACTIONS))
    print("EPOCHS=%d"%(EPOCHS))
    print("TORCH_DEVICE=%d"%(TORCH_DEVICE))
    print("INTERMEDIATE_TEST=%d"%(INTERMEDIATE_TEST))
    print("TEST_FREQ=%d"%(TEST_FREQ))
    print("LOG=%s"%(LOG))
    print("USE_GPU=%s"%(USE_GPU))

    print("# Experiment options")
    print("FEATURE_SIZE=%d"%(FEATURE_SIZE))
    print("LAMBDA=%d"%(LAMBDA))
    print("PREDICTION_LOSS=%s"%(PREDICTION_LOSS))
    print("USE_LSTM=%s"%(USE_LSTM))

    print("# Optimizer options")
    print("OPTIMIZER=%s"%(OPTIMIZER))
    print("LR=%f"%(LR))
    print("MOMENTUM=%f"%(MOMENTUM))
    print("CLIP_GRAD=%s"%(CLIP_GRAD))
