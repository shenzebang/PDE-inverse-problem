CUDA_VISIBLE_DEVICES=0
python main.py\
    pde_instance.domain_dim=4\
    pde_instance.name=Kinetic-Fokker-Planck\
    train.batch_size=250000\
    solver.train.sample_per_time=1000\
    solver.train.n_time_stamps=250\
    solver.train.batch_size_init=250000\
    solver.train.batch_size_terminal=250000\
    solver.train.batch_size_0T=250000\
    solver.train.sample_mode=random_time\
    neural_network.hidden_dim=32\
    neural_network.layers=2\
    train.optimizer.learning_rate.initial=1e-2\
    pde_instance.total_evolving_time=1\
    train.optimizer.learning_rate.scheduling=cosine\
    backend.use_pmap_train=True