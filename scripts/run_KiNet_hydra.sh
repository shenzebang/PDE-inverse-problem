CUDA_VISIBLE_DEVICES=0
python main.py\
    train.batch_size=50000\
    neural_network.hidden_dim=32\
    neural_network.layers=1\
    train.optimizer.learning_rate.initial=1e-2\
    pde_instance.total_evolving_time=5\
    train.optimizer.learning_rate.scheduling=cosine