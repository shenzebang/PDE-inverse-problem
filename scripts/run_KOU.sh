CUDA_VISIBLE_DEVICES=2
python main.py\
    pde_instance.domain_dim=4\
    pde_instance.name=Kinetic-Fokker-Planck\
    train.batch_size=250000\
    solver.train.sample_per_time=250\
    solver.train.n_time_stamps=100\
    solver.train.batch_size_init=2500\
    solver.train.batch_size_terminal=2500\
    solver.train.batch_size_0T=250000\
    solver.train.sample_mode=grid_time\
    neural_network.hidden_dim=32\
    neural_network.layers=2\
    train.optimizer.learning_rate.initial=1e-2\
    pde_instance.total_evolving_time=2\
    train.optimizer.learning_rate.scheduling=cosine\
    backend.use_pmap_train=False